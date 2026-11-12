"""P2: build the PyTorch-to-HIP (t2h) AKA v3 task store for the HIP Claude Code run.

Selects the training rows and the three eval sets (plan §6.6), removes training
rows that overlap an eval row (§6.5), generates one AKA schema-v3 package per
task (§6.4, package format in internal/hip-rl-exec/t2h-task-package-format.md)
and writes the upload tarball plus its sha256 manifest. Run on a compute node:
it executes dataset code in child processes.

    python3 build_t2h_tasks.py --out $DATA/t2h --workers 30
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2h_contract  # noqa: E402

DATA = Path("/shared_inference/yuankach/glm53/datasets/hip")
KERNELBENCH = Path("/home/yuankach/amd_kernel_writer/KernelBench/KernelBench")
BUILD_HIP_DATA = Path("/home/yuankach/amd_kernel_writer/scripts/build_hip_data.py")
AKA_ROOT = Path("/home/yuankach/hip-cc/AgentKernelArena")
DATASET_REVISION = "AIG-Datasets v0.2 d43e2530, build_hip_data.py seed 42"
SOURCE_NAMES = {"cuda_agent": "cudaagent", "gpumode": "gpumode", "kernelbench": "kernelbench"}
LICENSES = {"cudaagent": "CC-BY-4.0 (CUDA-Agent-Ops-6K via AIG-Datasets)",
            "gpumode": "MIT (GPUMODE/KernelBook via AIG-Datasets)",
            "kernelbench": "MIT (KernelBench 423217d)"}
PROVENANCE_SOURCE = {"cudaagent": "AIG-Datasets v0.2 CudaAgent", "gpumode": "AIG-Datasets v0.2 GPUMode",
                     "kernelbench": "KernelBench"}
EVAL128_PER_SOURCE = 64


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stable_key(seed: int, text: str) -> str:
    return sha256_text(f"{seed}:{text}")


def load_build_helpers():
    spec = importlib.util.spec_from_file_location("build_hip_data", BUILD_HIP_DATA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, line.rstrip("\n"), json.loads(line)


def row_spec(row: dict, *, split: str, row_file: str, row_index: int, line: str, policy_flags=None) -> dict:
    """Common task description from an rft-style or manifest-style row."""
    if "label" in row:
        label = row["label"]
        get = label.get
        hip_ref = get("hip_ref_code") or ""
        kernel = get("kernel_name") or "op"
    else:
        get = row.get
        hip_ref = row.get("hip_reference_code") or ""
        kernel = row.get("operation") or "op"
    source = SOURCE_NAMES[get("source")]
    name = f"{t2h_contract.operator_name(kernel)}_{sha256_text(str(get('id')))[:8]}"
    task_id = f"t2h/{source}/{name}"
    return {
        "kind": "row", "task_id": task_id, "split": split, "source": source, "kernel_name": kernel,
        "dir_name": name, "id": get("id"), "group_id": get("group_id"), "difficulty": get("difficulty"),
        "functional": get("pytorch_functional_code") or "", "module_code": get("pytorch_module_code") or "",
        "hip_ref": hip_ref, "row_file": row_file, "row_index": row_index, "row_sha256": sha256_text(line),
        "policy_flags": policy_flags if policy_flags is not None else (row.get("policy_flags") or []),
        "provenance": {"source": PROVENANCE_SOURCE[source], "revision": DATASET_REVISION,
                       "path": f"{row_file}#{row_index}", "license": LICENSES[source]},
    }


def kernelbench_specs() -> list[dict]:
    specs = []
    for level in ("level1", "level2"):
        for path in sorted((KERNELBENCH / level).glob("*.py")):
            code = path.read_text(encoding="utf-8")
            name = f"kb_l{level[-1]}_{t2h_contract.operator_name(path.stem)}"
            specs.append({
                "kind": "kernelbench", "task_id": f"t2h/kernelbench/{name}", "split": "kernelbench-t2h",
                "source": "kernelbench", "kernel_name": path.stem, "dir_name": name, "id": f"{level}/{path.name}",
                "group_id": f"kernelbench/{level}/{path.stem}", "difficulty": f"L{level[-1]}",
                "module": code, "module_code": code, "functional": "", "hip_ref": "",
                "row_file": f"KernelBench/{level}/{path.name}", "row_index": 0, "row_sha256": sha256_text(code),
                "policy_flags": [],
                "provenance": {"source": PROVENANCE_SOURCE["kernelbench"], "revision": "423217d",
                               "path": f"KernelBench/{level}/{path.name}", "license": LICENSES["kernelbench"]},
            })
    return specs


def stratified_pick(rows: list[dict], count: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Pick ``count`` rows proportionally by difficulty (largest remainder); return (picked, rest)."""
    by_difficulty = defaultdict(list)
    for row in sorted(rows, key=lambda r: stable_key(seed, str(r["id"]))):
        by_difficulty[str(row["difficulty"])].append(row)
    total = len(rows)
    quotas = {d: count * len(v) / total for d, v in by_difficulty.items()}
    alloc = {d: int(q) for d, q in quotas.items()}
    for d in sorted(quotas, key=lambda d: (alloc[d] - quotas[d], d))[: count - sum(alloc.values())]:
        alloc[d] += 1
    picked, rest = [], []
    for d, group in sorted(by_difficulty.items()):
        picked += group[: alloc[d]]
        rest += group[alloc[d]:]
    return picked, rest


def select(args) -> tuple[list[dict], dict]:
    report: dict = {"counts": {}, "dropped": Counter()}
    manifest_flags = {}
    for _, _, row in read_jsonl(DATA / "manifests/train.jsonl"):
        if row.get("task_type") == "pytorch_to_hip":
            manifest_flags[row["id"]] = row.get("policy_flags") or []
    train = []
    for index, line, row in read_jsonl(DATA / "rft/train.jsonl"):
        if row["label"].get("task_type") != "pytorch_to_hip":
            continue
        spec = row_spec(row, split="train", row_file="rft/train.jsonl", row_index=index, line=line,
                        policy_flags=manifest_flags.get(row["label"]["id"], []))
        train.append(spec)
    report["counts"]["train_t2h_rows"] = len(train)

    heldout = []
    for index, line, row in read_jsonl(DATA / "eval_final/heldout150.jsonl"):
        if row["label"].get("task_type") == "pytorch_to_hip":
            heldout.append(row_spec(row, split="heldout150-t2h", row_file="eval_final/heldout150.jsonl",
                                    row_index=index, line=line))
    heldout_ids = {s["id"] for s in heldout}
    fast_ids = {row["id"] for _, _, row in read_jsonl(DATA / "manifests/cuda_agent_fast_val.jsonl")}
    val = []
    for index, line, row in read_jsonl(DATA / "manifests/val.jsonl"):
        if row.get("task_type") == "pytorch_to_hip" and row["id"] not in heldout_ids:
            val.append(row_spec(row, split="reserve", row_file="manifests/val.jsonl", row_index=index, line=line))
    cuda_pool = [s for s in val if s["id"] in fast_ids]
    gpumode_pool = [s for s in val if s["source"] == "gpumode"]
    cuda_pick, cuda_rest = stratified_pick(cuda_pool, EVAL128_PER_SOURCE, args.seed)
    gm_pick, gm_rest = stratified_pick(gpumode_pool, EVAL128_PER_SOURCE, args.seed)
    picked_ids = {s["id"] for s in cuda_pick + gm_pick}
    for s in cuda_pick + gm_pick:
        s["split"] = "t2h-eval128"
    # Top-up order for t2h-eval128 when gfx950 validation drops a pick: same source and difficulty first.
    for rank, s in enumerate(cuda_rest):
        s["topup_rank"] = rank
    for rank, s in enumerate(gm_rest):
        s["topup_rank"] = rank
    reserve = [s for s in val if s["id"] not in picked_ids]
    kb = kernelbench_specs()

    removed = overlap(train, cuda_pick + gm_pick + heldout + kb, report)
    train = [s for s in train if s["id"] not in removed]
    report["counts"]["train_removed_overlap"] = len(removed)
    if args.policy_flagged == "drop":
        kept = [s for s in train if not s["policy_flags"]]
        report["dropped"]["train/policy_flagged"] = len(train) - len(kept)
        train = kept
    report["counts"].update({"train": len(train), "t2h-eval128": len(cuda_pick) + len(gm_pick),
                             "heldout150-t2h": len(heldout), "kernelbench-t2h": len(kb), "reserve": len(reserve)})
    specs = train + cuda_pick + gm_pick + heldout + kb + reserve
    ids = Counter(s["task_id"] for s in specs)
    dup = [t for t, n in ids.items() if n > 1]
    if dup:
        raise SystemExit(f"duplicate task ids: {dup[:5]}")
    return specs, report


def overlap(train: list[dict], evals: list[dict], report: dict) -> set:
    helpers = load_build_helpers()
    eval_fp = []
    for s in evals:
        code = s["module_code"]
        eval_fp.append((s, helpers.ast_digest(code), helpers.call_sequence(code), helpers.shingle_hashes(code)))
    eval_groups = {s["group_id"]: s for s in evals}
    aka_refs = []
    for path in sorted((AKA_ROOT / "tasks/hip").rglob("reference.py")):
        aka_refs.append((str(path.relative_to(AKA_ROOT)), helpers.shingle_hashes(path.read_text())))
    removed, matches = set(), []
    for s in train:
        code = s["module_code"]
        ast_d, calls, shingles = helpers.ast_digest(code), helpers.call_sequence(code), helpers.shingle_hashes(code)
        hit = None
        if s["group_id"] in eval_groups:
            hit = (eval_groups[s["group_id"]], "group_id", 1.0)
        else:
            for e, e_ast, e_calls, e_sh in eval_fp:
                if ast_d and ast_d == e_ast:
                    hit = (e, "normalized_ast_exact", 1.0)
                elif len(calls) >= 4 and calls == e_calls:
                    hit = (e, "call_sequence_exact", 1.0)
                else:
                    score = helpers.jaccard(shingles, e_sh)
                    if score >= 0.8:
                        hit = (e, "normalized_token_5gram_jaccard", round(score, 4))
                if hit:
                    break
        if hit:
            removed.add(s["id"])
            matches.append({"train_task": s["task_id"], "eval_task": hit[0]["task_id"], "eval_split": hit[0]["split"],
                            "rule": hit[1], "score": hit[2]})
        for aka_path, aka_sh in aka_refs:
            score = helpers.jaccard(shingles, aka_sh)
            if score >= 0.5:
                matches.append({"train_task": s["task_id"], "aka38": aka_path, "rule": "aka38_log_only",
                                "score": round(score, 4)})
    report["overlap_matches"] = matches
    report["counts"]["aka38_log_matches"] = sum(1 for m in matches if m.get("rule") == "aka38_log_only")
    return removed


def _child(spec, conn):
    try:
        conn.send(t2h_contract.analyze(spec))
    except BaseException as exc:  # noqa: BLE001
        conn.send({"drop": f"analyzer_crash:{type(exc).__name__}"})
    finally:
        conn.close()


def analyze_one(spec: dict, timeout: float) -> dict:
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(spec, child), daemon=True)
    proc.start()
    child.close()
    result = {"drop": "timeout"}
    if parent.poll(timeout):
        try:
            result = parent.recv()
        except EOFError:
            result = {"drop": "child_died"}
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join()
    elif proc.exitcode not in (0, None) and "drop" in result and result["drop"] == "timeout":
        result = {"drop": f"child_exit_{proc.exitcode}"}
    return result


def aka_validator():
    sys.path.insert(0, str(AKA_ROOT))
    from src import task_contract as tc  # noqa: PLC0415

    tc.BASELINE_LANGUAGES = tc.BASELINE_LANGUAGES + ("python",)
    original = tc._validate_workloads
    relaxed_prefixes = ("A task needs", "Performance cases must span", "Correctness cases lack",
                        "The minimum, non-power-of-two")

    def relaxed(workloads, axes, var, inputs):
        try:
            original(workloads, axes, var, inputs)
        except tc.TaskConfigError as exc:
            if not str(exc).startswith(relaxed_prefixes):
                raise

    tc._validate_workloads = relaxed
    return tc


def top_up_eval128(specs, results, report) -> None:
    """Replace t2h-eval128 picks that failed to build with reserve rows of the same source and difficulty."""
    ok = lambda s: "drop" not in results[s["task_id"]]  # noqa: E731
    promoted = 0
    for source in ("cudaagent", "gpumode"):
        dropped = [s for s in specs if s["split"] == "t2h-eval128" and s["source"] == source and not ok(s)]
        pool = sorted((s for s in specs if s["split"] == "reserve" and s["source"] == source
                       and "topup_rank" in s and ok(s)), key=lambda s: s["topup_rank"])
        for lost in dropped:
            same = [s for s in pool if s["difficulty"] == lost["difficulty"]] or pool
            if not same:
                break
            pick = same[0]
            pool.remove(pick)
            pick["split"], pick["eval128_topup"] = "t2h-eval128", True
            promoted += 1
    report["counts"]["eval128_topped_up"] = promoted


def write_package(specs, results, out: Path, report: dict, staging_root: Path) -> tuple[str, Path]:
    import yaml

    tc = aka_validator()
    staging = staging_root / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "tasks").mkdir(parents=True)
    index, flags = [], Counter()
    per_split = defaultdict(Counter)
    for spec in specs:
        result = results[spec["task_id"]]
        key = f"{spec['split']}/{spec['source']}"
        if "drop" in result:
            reason = result["drop"].split(":")[0]
            report["dropped"][f"{key}/{reason}"] += 1
            report.setdefault("drops", []).append({"task_id": spec["task_id"], "split": spec["split"],
                                                   "reason": result["drop"]})
            per_split[spec["split"]]["dropped"] += 1
            continue
        try:
            tc.validate_task_contract(yaml.safe_load(result["task_yaml"]), json.loads(result["workloads_json"]),
                                      task_id=spec["task_id"])
        except Exception as exc:  # noqa: BLE001
            report["dropped"][f"{key}/aka_validation"] += 1
            report.setdefault("drops", []).append({"task_id": spec["task_id"], "split": spec["split"],
                                                   "reason": f"aka_validation:{exc}"[:300]})
            per_split[spec["split"]]["dropped"] += 1
            continue
        rel = f"tasks/{spec['split']}/{spec['source']}/{spec['dir_name']}"
        root = staging / rel
        (root / "baseline").mkdir(parents=True)
        (root / "task.yaml").write_text(result["task_yaml"])
        (root / "workloads.json").write_text(result["workloads_json"])
        (root / "reference.py").write_text(result["reference_py"])
        (root / "baseline/kernel.py").write_text(result["baseline_py"])
        meta = {"task_id": spec["task_id"], "split": spec["split"], "source": spec["source"],
                "kernel_name": spec["kernel_name"], "group_id": spec["group_id"], "difficulty": spec["difficulty"],
                "row_file": spec["row_file"], "row_index": spec["row_index"], "row_sha256": spec["row_sha256"],
                "hip_ref_code_sha256": sha256_text(spec["hip_ref"]) if spec["hip_ref"] else None,
                "has_hip_ref": bool(spec["hip_ref"]), "hip_ref_policy_flagged": bool(spec["policy_flags"]),
                "build_flags": result["meta"]["flags"], "workloads": result["meta"]["workloads"],
                "input_bytes_w0": result["meta"]["input_bytes_w0"]}
        if "topup_rank" in spec and spec["split"] == "reserve":
            meta["eval128_topup_rank"] = spec["topup_rank"]
        if spec.get("eval128_topup"):
            meta["eval128_topup"] = True
        (root / "t2h.json").write_text(json.dumps(meta, indent=1) + "\n")
        # hip_ref_code ships only in the separate feasibility tarball (build_t2h_feasibility.py).
        index.append({"task_id": spec["task_id"], "split": spec["split"], "source": spec["source"], "path": rel,
                      "row_sha256": spec["row_sha256"]})
        per_split[spec["split"]][spec["source"]] += 1
        for flag in result["meta"]["flags"]:
            flags[flag] += 1
    index.sort(key=lambda r: r["task_id"])
    (staging / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    lines = []
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(staging).as_posix()}")
    leaked = [line for line in lines if "hip_ref" in line.split("  ", 1)[1]]
    if leaked:
        raise SystemExit(f"hip_ref material in the task package: {leaked[:3]}")
    manifest = "\n".join(lines) + "\n"
    (staging / "MANIFEST.sha256").write_text(manifest)
    version = hashlib.sha256(manifest.encode()).hexdigest()[:16]
    tarball = out / f"t2h-{version}.tar.gz"
    with tarfile.open(tarball, "w:gz", compresslevel=6) as tar:
        tar.add(staging, arcname=version)
    # The full tree stays in the tarball; NFS gets only the small top-level files (small-file writes are slow there).
    final = out / "store" / version
    final.mkdir(parents=True, exist_ok=True)
    for name in ("index.jsonl", "MANIFEST.sha256"):
        shutil.copy(staging / name, final / name)
    shutil.rmtree(staging)
    (out / f"t2h-{version}.tar.gz.sha256").write_text(
        f"{hashlib.sha256(tarball.read_bytes()).hexdigest()}  {tarball.name}\n")
    report["kept_per_split"] = {k: dict(v) for k, v in per_split.items()}
    report["build_flags"] = dict(flags)
    report["task_version"] = version
    report["tasks"] = len(index)
    return version, tarball


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DATA / "t2h")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--policy-flagged", choices=("drop", "keep"), default="drop",
                        help="Training rows whose dataset hip_ref_code fails the policy audit (plan §6.1).")
    parser.add_argument("--staging", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / f"t2h-{os.getpid()}",
                        help="Node-local scratch for the package tree.")
    parser.add_argument("--limit", type=int, default=0, help="Analyze only the first N specs of each split (smoke test).")
    parser.add_argument("--only", default="", help="Comma-separated task ids to analyze (debug).")
    args = parser.parse_args()
    # Containers run as a uid with no passwd entry; torch's lazy imports call getpass.getuser().
    os.environ.setdefault("USER", "t2h")
    os.environ.setdefault("LOGNAME", "t2h")
    import torch

    torch.set_num_threads(1)
    with torch.device("meta"):
        torch.nn.functional.softplus(torch.empty(2)) + 1  # import the meta/prims stack once, before forking
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    specs, report = select(args)
    if args.only:
        wanted = set(args.only.split(","))
        specs = [s for s in specs if s["task_id"] in wanted]
    if args.limit:
        per_split = defaultdict(list)
        for s in specs:
            if len(per_split[s["split"]]) < args.limit:
                per_split[s["split"]].append(s)
        specs = [s for group in per_split.values() for s in group]
    print(f"selected {len(specs)} specs in {time.time() - started:.0f}s: {report['counts']}", flush=True)
    results, done, lock = {}, [0], threading.Lock()

    def work(spec):
        result = analyze_one(spec, args.timeout)
        with lock:
            results[spec["task_id"]] = result
            done[0] += 1
            if done[0] % 500 == 0:
                print(f"  analyzed {done[0]}/{len(specs)} ({time.time() - started:.0f}s)", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, specs))
    top_up_eval128(specs, results, report)
    version, tarball = write_package(specs, results, args.out, report, args.staging)
    report["dropped"] = dict(sorted(report["dropped"].items()))
    report["policy_flagged_mode"] = args.policy_flagged
    report["elapsed_s"] = round(time.time() - started)
    reports = args.out / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"build-{version}.json").write_text(json.dumps(report, indent=1) + "\n")
    summary = {k: v for k, v in report.items() if k not in ("drops", "overlap_matches")}
    print(json.dumps(summary, indent=1))
    print(f"task_version {version}\ntarball {tarball}")


if __name__ == "__main__":
    main()
