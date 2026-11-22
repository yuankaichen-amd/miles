"""Check the aka38 PyTorch baselines against the AKA float64 references on one GPU.

For every task: correctness cases under the task's own compare bounds (AKA arena_eval inputs, generators
and comparators), CUDA-graph capture and replay of the smallest case, and the replay time of the largest
performance case. Each task runs in a fresh subprocess inside a throwaway AKA-style workspace.

    python3 check_baselines.py --aka ~/hip-cc/AgentKernelArena --out /tmp/aka38-check.jsonl [--only gelu]
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

INNER = r'''
import json, sys, time, traceback
import torch
sys.path.insert(0, ".")
from _aka_eval.contract import load_contract
from _aka_eval import values as V
from _aka_eval.compare import compare_tensor
import importlib.util

spec = importlib.util.spec_from_file_location("baseline_kernel", "baseline/kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
run = mod.run
contract = load_contract()
dev = torch.device("cuda")
report = {"cases": []}
max_cases = int(sys.argv[1])

def call(kernel, outputs):
    run(**kernel, **outputs)

def numel(case):
    s = contract.symbols(case)
    return sum(int(torch.Size(contract.shape(d, s)).numel()) for d in contract.inputs.values() if d["kind"] == "tensor")

corr = [c for c in contract.cases if "correctness" in c.checks]
corr = sorted(corr, key=numel)[:max_cases]
for case in corr:
    row = {"case": case.case_id, "axes": case.axes}
    try:
        inputs = V.make_inputs(contract, case, case.seeds[0], dev)
        ref = V.reference_outputs(contract, inputs.logical)
        outs = V.allocate_outputs(contract, case, dev)
        V.poison(outs, 1)
        call(V.clone_inputs(inputs.kernel), outs)
        torch.cuda.synchronize()
        res = {}
        for name in contract.checked_tensors:
            r = compare_tensor(contract.compare[name], outs[name], ref[name],
                               dtype_name=contract.outputs[name]["dtype"], baseline=outs[name],
                               element_order=contract.element_order)
            m = r.to_mapping()
            res[name] = dict(m, passed=bool(r.passed))
        row["compare"] = res
        row["passed"] = all(v["passed"] for v in res.values())
        del inputs, ref, outs
    except torch.OutOfMemoryError:
        row["oom"] = True
    except Exception:
        row["error"] = traceback.format_exc()[-1500:]
    torch.cuda.empty_cache()
    report["cases"].append(row)

def graph_check(case, time_it):
    inputs = V.make_inputs(contract, case, case.seeds[0], dev)
    kernel = V.clone_inputs(inputs.kernel)
    outs = V.allocate_outputs(contract, case, dev)
    call(kernel, outs); torch.cuda.synchronize()
    eager = {k: v.clone() for k, v in outs.items()}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(kernel, outs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call(kernel, outs)
    V.poison(outs, 2)
    g.replay(); torch.cuda.synchronize()
    same = all(torch.equal(outs[k], eager[k]) or torch.allclose(outs[k].double(), eager[k].double(), rtol=1e-5, atol=1e-6)
               for k in outs)
    out = {"case": case.case_id, "replay_matches_eager": bool(same)}
    if time_it:
        g.replay(); torch.cuda.synchronize()
        t = time.perf_counter(); n = 5
        for _ in range(n):
            g.replay()
        torch.cuda.synchronize()
        out["replay_ms"] = (time.perf_counter() - t) / n * 1e3
    return out

perf = [c for c in contract.cases if "performance" in c.checks]
try:
    report["graph_small"] = graph_check(corr[0], False)
except Exception:
    report["graph_small"] = {"error": traceback.format_exc()[-1500:]}
torch.cuda.empty_cache()
try:
    report["graph_large"] = graph_check(max(perf, key=numel), True)
except torch.OutOfMemoryError:
    report["graph_large"] = {"oom": True}
except Exception:
    report["graph_large"] = {"error": traceback.format_exc()[-1500:]}
print("AKA38_REPORT " + json.dumps(report, default=str))
'''


def tasks(aka_tasks: Path):
    for yaml_path in sorted(aka_tasks.glob("*/*/*/task.yaml")):
        yield yaml_path.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aka", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, default=HERE / "baselines")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-cases", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    import yaml
    sys.path.insert(0, str(args.aka))
    from src import task_contract as tc
    tc.BASELINE_LANGUAGES = tc.BASELINE_LANGUAGES + ("python",)

    aka_tasks = args.aka / "tasks" / "hip"
    eval_pkg = args.aka / "src" / "tools" / "arena_eval"
    root = Path(tempfile.mkdtemp(prefix="aka38ws-"))
    with args.out.open("w") as fh:
        for task_dir in tasks(aka_tasks):
            rel = task_dir.relative_to(aka_tasks)
            name = f"{rel.parts[1]}_{rel.parts[2]}"
            if args.only and args.only not in str(rel):
                continue
            ws = root / name
            shutil.copytree(task_dir, ws, ignore=shutil.ignore_patterns("baseline", "__pycache__"))
            (ws / "baseline").mkdir()
            shutil.copy(args.baselines / rel / "kernel.py", ws / "baseline" / "kernel.py")
            shutil.copytree(eval_pkg, ws / "_aka_eval", ignore=shutil.ignore_patterns("__pycache__"))
            raw = yaml.safe_load((ws / "task.yaml").read_text())
            raw["baseline"] = {**raw["baseline"], "language": "python", "entry": "baseline/kernel.py"}
            workloads = json.loads((ws / "workloads.json").read_text())
            contract = tc.validate_task_contract(raw, workloads, task_id=f"t2h/aka/{name}")
            (ws / "_aka_eval" / "contract.json").write_text(json.dumps(contract.resolved_contract("triton")))
            row = {"task": str(rel), "name": name}
            try:
                proc = subprocess.run([sys.executable, "-c", INNER, str(args.max_cases)], cwd=ws,
                                      capture_output=True, text=True, timeout=args.timeout)
                line = next((l for l in proc.stdout.splitlines() if l.startswith("AKA38_REPORT ")), None)
                if line:
                    row.update(json.loads(line[len("AKA38_REPORT "):]))
                else:
                    row["error"] = (proc.stderr or proc.stdout)[-2000:]
            except subprocess.TimeoutExpired:
                row["error"] = f"timeout {args.timeout}s"
            cases = row.get("cases", [])
            row["ok"] = (bool(cases) and all(c.get("passed") or c.get("oom") for c in cases)
                         and row.get("graph_small", {}).get("replay_matches_eager") is True)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{'OK  ' if row['ok'] else 'FAIL'} {rel} "
                  f"cases={sum(bool(c.get('passed')) for c in cases)}/{len(cases)} "
                  f"graph={row.get('graph_small', {}).get('replay_matches_eager')} "
                  f"ms={row.get('graph_large', {}).get('replay_ms')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
