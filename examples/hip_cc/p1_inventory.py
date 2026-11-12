"""P1 data inventory for the HIP Claude Code run (plan §10 P1).

Streams the source JSONL files under $DATA and prints a JSON report. Run on a
compute node; manifests/train.jsonl alone is ~900 MB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def fields(row: dict) -> dict:
    """Normalize rft-style ({prompt, label}) and manifest-style rows."""
    if "label" in row:
        label = row["label"]
        return {
            "id": label.get("id"),
            "source": label.get("source"),
            "task_type": label.get("task_type"),
            "functional": label.get("pytorch_functional_code") or "",
            "group_id": label.get("group_id"),
            "difficulty": label.get("difficulty"),
            "eval_split": label.get("eval_split"),
        }
    return {
        "id": row.get("id"),
        "source": row.get("source"),
        "task_type": row.get("task_type"),
        "functional": row.get("pytorch_functional_code") or "",
        "group_id": row.get("group_id"),
        "difficulty": row.get("difficulty"),
        "policy_flags": row.get("policy_flags") or [],
    }


def count_file(path: Path) -> tuple[dict, set[str]]:
    by_type: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    empty_func: Counter[str] = Counter()
    difficulty: Counter[str] = Counter()
    flagged: Counter[str] = Counter()
    flag_kinds: Counter[str] = Counter()
    t2h_ids: set[str] = set()
    total = 0
    for row in rows(path):
        f = fields(row)
        total += 1
        by_type[f["task_type"]] += 1
        if f["task_type"] != "pytorch_to_hip":
            continue
        t2h_ids.add(f["id"])
        by_source[f["source"]] += 1
        difficulty[f"{f['source']}/{f['difficulty']}"] += 1
        if not f["functional"].strip():
            empty_func[f["source"]] += 1
        if f.get("policy_flags"):
            flagged[f["source"]] += 1
            for flag in f["policy_flags"]:
                flag_kinds[flag.split(":")[0]] += 1
    report = {
        "rows": total,
        "task_type": dict(by_type),
        "t2h_by_source": dict(by_source),
        "t2h_empty_functional": dict(empty_func),
        "t2h_difficulty": dict(sorted(difficulty.items())),
    }
    if flagged:
        report["t2h_policy_flagged"] = dict(flagged)
        report["t2h_policy_flag_kinds"] = dict(flag_kinds)
    return report, t2h_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("/shared_inference/yuankach/glm53/datasets/hip"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    d = args.data
    files = [
        "rft/train.jsonl",
        "rft/train_mt_func.jsonl",
        "manifests/train.jsonl",
        "manifests/val.jsonl",
        "manifests/cuda_agent_fast_val.jsonl",
        "manifests/test.jsonl",
        "eval_final/heldout150.jsonl",
        "eval_final/candidates.jsonl",
    ]
    report: dict = {"files": {}}
    ids: dict[str, set[str]] = {}
    for name in files:
        report["files"][name], ids[name] = count_file(d / name)

    rft_t2h, man_t2h = ids["rft/train.jsonl"], ids["manifests/train.jsonl"]
    flagged_ids = {
        f["id"]
        for f in map(fields, rows(d / "manifests/train.jsonl"))
        if f["task_type"] == "pytorch_to_hip" and f.get("policy_flags")
    }
    report["checks"] = {
        "rft_train_t2h_equals_manifest_train_t2h": rft_t2h == man_t2h,
        "policy_flagged_t2h_inside_rft_train_t2h": len(flagged_ids & rft_t2h),
        "policy_flagged_t2h_total": len(flagged_ids),
        "train_mt_func_t2h_subset_of_rft_train_t2h": ids["rft/train_mt_func.jsonl"] <= rft_t2h,
        "heldout150_t2h_inside_val_t2h": len(ids["eval_final/heldout150.jsonl"] & ids["manifests/val.jsonl"]),
        "fast_val_inside_val_t2h": len(ids["manifests/cuda_agent_fast_val.jsonl"] & ids["manifests/val.jsonl"]),
        "val_t2h_overlap_train_t2h_ids": len(ids["manifests/val.jsonl"] & rft_t2h),
    }

    sources = sorted(p for p in d.rglob("*") if p.is_file() and "verification" not in p.parts and "t2h" not in p.parts)
    manifest = [f"{sha256(p)}  {p.relative_to(d)}" for p in sources]
    report["sha256_manifest"] = manifest
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sha256_manifest"}, indent=2))


if __name__ == "__main__":
    main()
