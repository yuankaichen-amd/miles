"""Phase 0b training sample: N validated train tasks, stratified, dealt into the four ablation cells.

The stratification rule is a parameter (``--strata``, ``--allocation``, ``--gfx950``); none has a
default, so the rule is always stated. Tasks come from the final train list written by
``t2h_manifest.py workspaces``; difficulty comes from each task's ``t2h.json`` in the P2 package and
``no_known_gfx950_solution`` from the Crusoe manifest. Tasks used in Phase 0a are excluded.

The picks of each stratum are dealt round-robin, from a seeded shuffle, into the cells
{counters on/off} x {/kb on/off}, so every cell gets N/4 tasks with the same stratum mix (to within
one task per stratum). Each row's label carries ``counters`` and ``kb``, which ``hip_cc_agent``
applies per task.

    python3 phase0b/sample_tasks.py --strata source,difficulty --allocation proportional --gfx950 any \
        [--n 512] [--seed 20261117] [--exclude <tasks.jsonl> ...] --dry-run
    python3 phase0b/sample_tasks.py ... --out $DATA/t2h/phase0b      (writes train512.jsonl, sample.json)

Run in the Miles image (reads the P2 tarball).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import t2h_manifest  # noqa: E402

STRATA_KEYS = ("source", "difficulty", "gfx950")
ALLOCATIONS = ("proportional", "equal")
GFX950_FILTERS = ("any", "known")
CELLS = ({"counters": True, "kb": True}, {"counters": True, "kb": False},
         {"counters": False, "kb": True}, {"counters": False, "kb": False})
DEFAULT_EXCLUDES = (
    t2h_manifest.DATA / "t2h" / "phase0a" / "tasks.jsonl",
    Path("/shared_inference/yuankach/glm53/hip-cc/phase0/0a-568026/check5-tasks.jsonl"),
)


def cell_name(cell: dict) -> str:
    return f"counters_{'on' if cell['counters'] else 'off'}-kb_{'on' if cell['kb'] else 'off'}"


def allocate(sizes: dict, n: int, allocation: str) -> dict:
    """Tasks per stratum. ``proportional``: largest remainder on stratum size. ``equal``: the same
    share per stratum, with what a small stratum can't take spread over the others (water-filling)."""
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"asked for {n} tasks, only {total} eligible")
    if allocation == "proportional":
        quotas = {k: n * size / total for k, size in sizes.items()}
        out = {k: int(q) for k, q in quotas.items()}
        for k in sorted(quotas, key=lambda k: (-(quotas[k] - out[k]), str(k)))[: n - sum(out.values())]:
            out[k] += 1
        return out
    if allocation == "equal":
        out = dict.fromkeys(sizes, 0)
        left = n
        open_keys = sorted(sizes, key=str)
        while left:
            open_keys = [k for k in open_keys if out[k] < sizes[k]]
            share, extra = divmod(left, len(open_keys))
            for i, k in enumerate(open_keys):
                take = min(share + (1 if i < extra else 0), sizes[k] - out[k])
                out[k] += take
                left -= take
        return out
    raise ValueError(f"unknown allocation {allocation!r}")


def stratum(task: dict, keys) -> tuple:
    return tuple(task[k] for k in keys)


def sample(tasks: list[dict], n: int, keys, allocation: str, gfx950: str, seed: int, exclude=()) -> dict:
    """``tasks``: {task_id, source, difficulty, gfx950, row}. Returns picks with cells and the record."""
    exclude = set(exclude)
    eligible = [t for t in tasks if t["task_id"] not in exclude and (gfx950 == "any" or t["gfx950"] == "known")]
    by = collections.defaultdict(list)
    for t in sorted(eligible, key=lambda t: t["task_id"]):
        by[stratum(t, keys)].append(t)
    quota = allocate({k: len(v) for k, v in by.items()}, n, allocation)
    rng = random.Random(seed)
    picks = []
    offset = 0
    for key in sorted(by, key=str):
        chosen = rng.sample(by[key], quota[key])
        for i, t in enumerate(chosen):
            picks.append({**t, "cell": CELLS[(offset + i) % len(CELLS)]})
        offset += len(chosen)
    per_cell = collections.Counter(cell_name(p["cell"]) for p in picks)
    record = {
        "rule": {"strata": list(keys), "allocation": allocation, "gfx950": gfx950, "n": n, "seed": seed},
        "eligible": len(eligible), "excluded": sorted(exclude & {t["task_id"] for t in tasks}),
        "strata": {"/".join(map(str, k)): {"eligible": len(by[k]), "picked": quota[k]} for k in sorted(by, key=str)},
        "cells": dict(sorted(per_cell.items())),
        "cells_by_stratum": {
            "/".join(map(str, k)): dict(sorted(collections.Counter(
                cell_name(p["cell"]) for p in picks if stratum(p, keys) == k).items()))
            for k in sorted(by, key=str)},
    }
    return {"picks": picks, "record": record}


def load_tasks(train_list: Path, manifest: dict, package: Path) -> list[dict]:
    flags = {t["task_id"]: t.get("no_known_gfx950_solution") for t in manifest["tasks"]}
    meta = t2h_manifest.topup_meta(package, splits=("train",))
    tasks = []
    for line in train_list.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row["label"]["task_id"]
        flag = flags.get(task_id)
        tasks.append({"task_id": task_id, "source": row["label"]["source"],
                      "difficulty": (meta.get(task_id) or {}).get("difficulty"),
                      "gfx950": "known" if flag is False else ("none" if flag is True else "unknown"),
                      "row": row})
    return tasks


def excluded_ids(paths) -> set[str]:
    ids = set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                ids.add((row.get("label") or {}).get("task_id") or row["prompt"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", required=True, help=f"comma list from {STRATA_KEYS}")
    parser.add_argument("--allocation", required=True, choices=ALLOCATIONS)
    parser.add_argument("--gfx950", required=True, choices=GFX950_FILTERS,
                        help="known: only tasks with a known gfx950 solution (no_known_gfx950_solution false)")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--version", default=t2h_manifest.VERSION)
    parser.add_argument("--manifest", default=None, help="Crusoe manifest (default: the latest)")
    parser.add_argument("--train", default=str(t2h_manifest.DATA / "t2h" / "workspaces" / "train" / "manifest.jsonl"))
    parser.add_argument("--exclude", nargs="*", default=[str(p) for p in DEFAULT_EXCLUDES])
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the allocation; write nothing")
    args = parser.parse_args()
    keys = tuple(k for k in args.strata.split(",") if k)
    if not keys or set(keys) - set(STRATA_KEYS):
        parser.error(f"--strata must be a comma list from {STRATA_KEYS}")
    if args.n % len(CELLS):
        parser.error(f"--n must be a multiple of {len(CELLS)}")
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run")
    manifest_path = Path(args.manifest) if args.manifest else (
        t2h_manifest.DATA / "t2h" / "manifests" / f"manifest-{args.version}-latest.json")
    manifest = json.loads(manifest_path.read_text())
    package = t2h_manifest.DATA / "t2h" / f"t2h-{args.version}.tar.gz"
    train = Path(args.train)
    tasks = load_tasks(train, manifest, package)
    missing = [t["task_id"] for t in tasks if "difficulty" in keys and t["difficulty"] is None]
    if missing:
        raise SystemExit(f"{len(missing)} train tasks have no difficulty, e.g. {missing[:3]}")
    result = sample(tasks, args.n, keys, args.allocation, args.gfx950, args.seed, excluded_ids(args.exclude))
    record = {**result["record"], "manifest": str(manifest_path), "manifest_generated_at": manifest.get("generated_at"),
              "train_list": str(train), "train_list_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
              "train_tasks": len(tasks)}
    if args.dry_run:
        print(json.dumps(record, indent=1))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(result["picks"], key=lambda p: p["task_id"]):
        label = {**p["row"]["label"], "difficulty": p["difficulty"], "cell": cell_name(p["cell"]), **p["cell"]}
        rows.append({"prompt": p["task_id"], "label": label})
    target = out / f"train{args.n}.jsonl"
    if target.exists():
        raise SystemExit(f"{target} exists; the sample is written once")
    target.write_text("".join(json.dumps(r) + "\n" for r in rows))
    record["output"] = str(target)
    record["output_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    record["tasks"] = [{"task_id": r["prompt"], "difficulty": r["label"]["difficulty"], "cell": r["label"]["cell"]}
                       for r in rows]
    (out / "sample.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "tasks"}, indent=1))


if __name__ == "__main__":
    main()
