"""Build the aka38 t2h task package: the 38 AKA HIP tasks with trusted PyTorch baselines.

Each AKA task keeps its task.yaml (baseline switched to python), workloads.json, reference.py and, where
the task declares a generator, inputs.py. AKA's HIP baseline and README are not shipped. Every task lands
in split ``aka38-t2h``, source ``aka``, as ``t2h/aka/<operator>_<scenario>``.

    python3 build_aka38_package.py --out /shared_inference/yuankach/glm53/datasets/hip/t2h-aka38
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
AKA_ROOT = Path("/home/yuankach/hip-cc/AgentKernelArena")
SPLIT, SOURCE = "aka38-t2h", "aka"
GPUMODE_DERIVED = {"gelu", "conv1x1_ffn_gelu", "feedforward_vstack_mlp", "ffn_relu_residual_layernorm",
                   "gate_gru_selection", "inner_prod", "linear_log_softmax", "cross_entropy_label_smoothing"}
SHIPPED = ("task.yaml", "workloads.json", "reference.py", "inputs.py")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def aka_validator(aka_root: Path):
    sys.path.insert(0, str(aka_root))
    from src import task_contract as tc  # noqa: PLC0415

    tc.BASELINE_LANGUAGES = tc.BASELINE_LANGUAGES + ("python",)
    return tc


def output_bytes(raw: dict, workloads: dict, tc) -> int:
    """Largest total output size over the task's cases."""
    interface = raw["interface"]
    worst = 0
    for case in workloads["cases"]:
        symbols = {n: a["value"] if a["kind"] == "const" else case["axes"][n] for n, a in interface["axes"].items()}
        for name, expr in interface.get("derived_axes", {}).items():
            symbols[name] = _eval_dim(expr, symbols)
        total = 0
        for out in interface["outputs"].values():
            n = 1
            for dim in out["shape"]:
                n *= _eval_dim(dim, symbols)
            total += n * _itemsize(out["dtype"])
        worst = max(worst, total)
    return worst


def _eval_dim(expr, symbols):
    if isinstance(expr, int):
        return expr
    env = dict(symbols, ceil_div=lambda a, b: -(-a // b), min=min, max=max)
    return int(eval(str(expr), {"__builtins__": {}}, env))  # noqa: S307 - trusted AKA task files


def _itemsize(dtype: str) -> int:
    return {"float64": 8, "int64": 8, "float32": 4, "int32": 4, "bfloat16": 2, "float16": 2, "int16": 2}.get(dtype, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aka", type=Path, default=AKA_ROOT)
    ap.add_argument("--baselines", type=Path, default=HERE / "baselines")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tc = aka_validator(args.aka)
    revision = subprocess.run(["git", "-C", str(args.aka), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"
    aka_tasks = args.aka / "tasks" / "hip"
    staging = Path(tempfile.mkdtemp(prefix="aka38pkg-")) / "staging"
    index, report = [], {"tasks": [], "aka_revision": revision, "split": SPLIT, "source": SOURCE}
    for yaml_path in sorted(aka_tasks.glob("*/*/*/task.yaml")):
        task_dir = yaml_path.parent
        category, op, scenario = task_dir.relative_to(aka_tasks).parts
        name = f"{op}_{scenario}"
        task_id = f"t2h/{SOURCE}/{name}"
        raw = yaml.safe_load(yaml_path.read_text())
        original_baseline = raw["baseline"]
        raw["baseline"] = {"language": "python", "entry": "baseline/kernel.py",
                           "provenance": {"source": "AgentKernelArena", "revision": revision,
                                          "path": f"tasks/hip/{category}/{op}/{scenario}",
                                          "license": original_baseline.get("provenance", {}).get("license", "MIT")}}
        workloads = json.loads((task_dir / "workloads.json").read_text())
        tc.validate_task_contract(raw, workloads, task_id=task_id)
        baseline = (args.baselines / category / op / scenario / "kernel.py").read_bytes()

        rel = f"tasks/{SPLIT}/{SOURCE}/{name}"
        root = staging / rel
        (root / "baseline").mkdir(parents=True)
        for fname in SHIPPED:
            if (task_dir / fname).exists() and fname != "task.yaml":
                shutil.copy(task_dir / fname, root / fname)
        (root / "task.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, width=110))
        (root / "baseline" / "kernel.py").write_bytes(baseline)
        row_sha = sha256(yaml_path.read_bytes())
        hip = task_dir / original_baseline["entry"]
        meta = {"task_id": task_id, "split": SPLIT, "source": SOURCE, "kernel_name": name,
                "group_id": f"aka/{category}/{op}", "difficulty": "aka", "row_file": f"tasks/hip/{category}/{op}/{scenario}",
                "row_index": 0, "row_sha256": row_sha,
                "hip_ref_code_sha256": sha256(hip.read_bytes()) if hip.exists() else None,
                "has_hip_ref": hip.exists(), "hip_ref_policy_flagged": False,
                "gpumode_derived": op in GPUMODE_DERIVED, "aka_category": category,
                "build_flags": [], "workloads": len(workloads["cases"]),
                "max_output_bytes": output_bytes(raw, workloads, tc)}
        (root / "t2h.json").write_text(json.dumps(meta, indent=1) + "\n")
        index.append({"task_id": task_id, "split": SPLIT, "source": SOURCE, "path": rel, "row_sha256": row_sha})
        report["tasks"].append({k: meta[k] for k in ("task_id", "gpumode_derived", "aka_category", "workloads",
                                                     "max_output_bytes")})
    index.sort(key=lambda r: r["task_id"])
    (staging / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    lines = [f"{sha256(p.read_bytes())}  {p.relative_to(staging).as_posix()}"
             for p in sorted(p for p in staging.rglob("*") if p.is_file())]
    if any("hip_ref" in line.split("  ", 1)[1] or line.endswith(".hip") for line in lines):
        raise SystemExit("HIP reference material in the task package")
    manifest = "\n".join(lines) + "\n"
    (staging / "MANIFEST.sha256").write_text(manifest)
    version = sha256(manifest.encode())[:16]
    args.out.mkdir(parents=True, exist_ok=True)
    tarball = args.out / f"t2h-{version}.tar.gz"
    with tarfile.open(tarball, "w:gz", compresslevel=6) as tar:
        tar.add(staging, arcname=version)
    (args.out / f"t2h-{version}.tar.gz.sha256").write_text(f"{sha256(tarball.read_bytes())}  {tarball.name}\n")
    shutil.copy(staging / "index.jsonl", args.out / f"index-{version}.jsonl")
    report.update(task_version=version, n_tasks=len(index), tarball=str(tarball),
                  gpumode_derived=sorted(t["task_id"] for t in report["tasks"] if t["gpumode_derived"]))
    (args.out / f"report-{version}.json").write_text(json.dumps(report, indent=1) + "\n")
    shutil.rmtree(staging.parent)
    print(json.dumps({k: report[k] for k in ("task_version", "n_tasks", "tarball")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
