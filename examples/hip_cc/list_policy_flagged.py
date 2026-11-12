"""List the t2h training rows whose dataset hip_ref_code fails the policy audit (plan §6.1).

These rows are left out of the training pool (``build_t2h_tasks.py --policy-flagged drop``);
the list is kept so they can be added back with ``--policy-flagged keep``.

    python3 list_policy_flagged.py --out <file.md>
"""

import argparse
import hashlib
from collections import Counter

from build_t2h_tasks import DATA, read_jsonl

FRONTMATTER = '---\ncursor:\n  subagentId: "bc-4fd51009-c224-56ab-8289-5bd17efb0a2b"\n---\n\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    flags = {}
    for _, _, row in read_jsonl(DATA / "manifests/train.jsonl"):
        if row.get("task_type") == "pytorch_to_hip" and row.get("policy_flags"):
            flags[row["id"]] = row["policy_flags"]
    rows, train_t2h = [], 0
    for index, _, row in read_jsonl(DATA / "rft/train.jsonl"):
        label = row["label"]
        if label.get("task_type") != "pytorch_to_hip":
            continue
        train_t2h += 1
        if label["id"] in flags:
            rows.append((label["id"], label.get("source", ""), index, flags[label["id"]]))
    by_source = Counter(r[1] for r in rows)
    by_flag = Counter(f for r in rows for f in r[3])
    ids_sha = hashlib.sha256("".join(f"{r[0]}\n" for r in rows).encode()).hexdigest()
    lines = [
        FRONTMATTER,
        "# Policy-flagged t2h training rows left out of the pool\n\n",
        f"Decision (Yuankai, 2026-11-12): these {len(rows):,} rows of `$DATA/rft/train.jsonl` "
        f"(of {train_t2h:,} pytorch_to_hip rows) are not used for now, because their dataset "
        "`hip_ref_code` fails the policy audit in `manifests/train.jsonl` (`policy_flags`). "
        "That leaves 5,651 training rows before eval-overlap removal and gfx950 validation. "
        "To add them back, rebuild with `build_t2h_tasks.py --policy-flagged keep`.\n\n",
        f"- `$DATA` = `{DATA}`\n",
        f"- By source: {dict(by_source)}\n",
        f"- By flag (a row can have several): {dict(by_flag)}\n",
        f"- sha256 of the id list (one id per line, in this order): `{ids_sha}`\n\n",
        "| id | source | row in rft/train.jsonl | policy_flags |\n|---|---|---|---|\n",
    ]
    lines += [f"| `{i}` | {s} | {n} | {', '.join(f)} |\n" for i, s, n, f in rows]
    with open(args.out, "w") as fh:
        fh.writelines(lines)
    print(f"{len(rows)} rows, {dict(by_source)}, {dict(by_flag)}, sha {ids_sha}")


if __name__ == "__main__":
    main()
