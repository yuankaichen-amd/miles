"""Phase 0a task sample: 5 train tasks with a fixed seed, rendered into their own directory.

3 GPUMode (one each of L1, L2, L3) and 2 CudaAgent (two different difficulties picked by the same
seed), all train tasks that passed all runs in the given manifest and have a gfx950-feasible
hip_ref (``no_known_gfx950_solution`` is false). Writes ``<out>/tasks.jsonl`` (driver rows) and
``<out>/sample.json`` (seed, manifest, picks).

    python3 sample_tasks.py --manifest <manifest json> --out $DATA/t2h/phase0a [--seed 20261113]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import t2h_manifest  # noqa: E402

GPUMODE_LEVELS = ("L1", "L2", "L3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--version", default=t2h_manifest.VERSION)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    runs = int((manifest.get("rules") or {}).get("runs", 3))
    package = t2h_manifest.DATA / "t2h" / f"t2h-{args.version}.tar.gz"
    meta = t2h_manifest.topup_meta(package, splits=("train",))
    eligible = sorted(t["task_id"] for t in manifest["tasks"]
                      if t.get("split") == "train" and t.get("no_known_gfx950_solution") is False
                      and t2h_manifest.passed_all_runs(t, runs) and t["task_id"] in meta)
    rng = random.Random(args.seed)
    by = {}
    for task_id in eligible:
        by.setdefault((meta[task_id]["source"], meta[task_id]["difficulty"]), []).append(task_id)
    picks = [(rng.choice(by[("gpumode", level)]), level) for level in GPUMODE_LEVELS]
    cuda_levels = sorted(level for source, level in by if source == "cudaagent")
    picks += [(rng.choice(by[("cudaagent", level)]), level) for level in sorted(rng.sample(cuda_levels, 2))]

    from cc_workspace import render_workspace  # noqa: PLC0415

    out = Path(args.out)
    rows = []
    for task_id, level in picks:
        workspace = out / "workspaces" / task_id.rsplit("/", 1)[-1]
        if not (workspace / "solution" / "kernel.hip").is_file():
            render_workspace(package, task_id, workspace, task_version=args.version, split="train")
        rows.append({"prompt": task_id, "label": {"task_id": task_id, "split": "train",
                                                  "source": meta[task_id]["source"], "difficulty": level,
                                                  "task_version": args.version, "workspace_dir": str(workspace)}})
    (out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    record = {"seed": args.seed, "manifest": str(args.manifest), "manifest_generated_at": manifest.get("generated_at"),
              "eligible": len(eligible), "cudaagent_levels_available": cuda_levels,
              "tasks": [{"task_id": t, "difficulty": d} for t, d in picks]}
    (out / "sample.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
