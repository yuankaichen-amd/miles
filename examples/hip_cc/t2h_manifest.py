"""The gfx950-validated task lists for rollouts (phase 0a start condition 4).

    python3 t2h_manifest.py fetch [--version V]      save GET /tasks/manifest?version=V
    python3 t2h_manifest.py topup [--manifest F]      plan the t2h-eval128 top-up from the ranked reserve
    python3 t2h_manifest.py workspaces [--manifest F] render validated tasks and write
                                                      $DATA/t2h/workspaces/<split>/manifest.jsonl

``workspaces`` keeps only tasks the Crusoe manifest marks valid with every one of the manifest's
``rules.runs`` baseline runs present (on as many timing nodes), in the order of the P2 index.

``topup`` replaces each dropped t2h-eval128 pick with a reserve task of the same source, same
difficulty first: the first task in the P2 ``eval128_topup_rank`` order that has passed all runs.
``$DATA/t2h/eval128-known-drops.txt`` lists tasks Worker B reports as dropped where the manifest
still says pending (they count as dropped). It writes ``$DATA/t2h/eval128-topup-latest.json``;
``workspaces`` moves the promoted tasks into the t2h-eval128 list. ``freeze-eval128`` checks the
list (128 tasks, 64 per source, all passed, plan complete), records its sha256 in
``$DATA/t2h/eval128-frozen.json`` and makes it read-only; later ``workspaces`` runs leave it alone.
Each one is rendered once with ``cc_workspace.render_workspace`` (the only path from the task
store to an agent) into ``$DATA/t2h/workspaces/<split>/<name>/``. Manifest rows are Miles
prompt-data rows: ``{"prompt": task_id, "label": {task_id, split, source, task_version,
workspace_dir}}``. Run on a compute node, in the Miles image for ``workspaces`` (it needs PyYAML
and AKA); ``fetch`` reads the token in-process from ``$HIP_SANDBOX_TOKEN_FILE``.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA = Path(os.environ.get("DATA", "/shared_inference/yuankach/glm53/datasets/hip"))
VERSION = "6e0402dbba69efa9"
URL_FILE = os.environ.get("HIP_SF_URL_FILE", "/shared_inference/yuankach/glm53/hip-cc/crusoe_relay_url.txt")
TOKEN_FILE = os.environ.get("HIP_SANDBOX_TOKEN_FILE") or os.environ.get(
    "HIP_SF_TOKEN_FILE", os.path.expanduser("~/.hip_cc/crusoe_grader_token"))
VALID_STATUS = ("valid",)
TOPUP_PLAN = DATA / "t2h" / "eval128-topup-latest.json"
KNOWN_DROPS = DATA / "t2h" / "eval128-known-drops.txt"
EVAL128_FROZEN = DATA / "t2h" / "eval128-frozen.json"
EVAL128_PER_SOURCE = 64


def known_drops() -> set[str]:
    if not KNOWN_DROPS.exists():
        return set()
    return {line.split("#", 1)[0].strip() for line in KNOWN_DROPS.read_text().splitlines()} - {""}


def passed_all_runs(task: dict | None, runs: int) -> bool:
    if not task or task.get("status") not in VALID_STATUS:
        return False
    baseline = task.get("baseline") or {}
    return (len(task.get("timing_nodes") or []) >= runs and bool(baseline)
            and all(len(w.get("runs_ms") or []) >= runs for w in baseline.values()))


def topup_meta(package: Path, splits=("t2h-eval128", "reserve")) -> dict[str, dict]:
    """difficulty and eval128_topup_rank from each task's t2h.json; no other member is read."""
    import re  # noqa: PLC0415
    import tarfile  # noqa: PLC0415

    wanted = re.compile(r"(?:^|/)tasks/(%s)/[^/]+/[^/]+/t2h\.json$" % "|".join(map(re.escape, splits)))
    meta = {}
    with tarfile.open(package) as tar:
        for member in tar:
            if member.isfile() and wanted.search(member.name):
                data = json.loads(tar.extractfile(member).read())
                meta[data["task_id"]] = {"difficulty": data.get("difficulty"),
                                         "rank": data.get("eval128_topup_rank"), "source": data.get("source")}
    return meta


def topup(manifest_path: Path, version: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != version:
        raise SystemExit(f"manifest version {manifest.get('version')} != {version}")
    runs = int((manifest.get("rules") or {}).get("runs", 3))
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    meta = topup_meta(DATA / "t2h" / f"t2h-{version}.tar.gz")
    rows = [json.loads(line) for line in (DATA / "t2h" / "store" / version / "index.jsonl").read_text().splitlines()
            if line.strip()]

    drops = known_drops()

    def state(task_id: str) -> str:
        task = tasks.get(task_id) or {}
        if task_id in drops:
            return "dropped"
        if passed_all_runs(task, runs):
            return "passed"
        return "dropped" if task.get("status") == "dropped" else "pending"

    slots, pending_picks = [], []
    for source in ("cudaagent", "gpumode"):
        picks = [r["task_id"] for r in rows if r["split"] == "t2h-eval128" and r["source"] == source]
        lost = [t for t in picks if state(t) == "dropped"]
        pending_picks += [t for t in picks if state(t) == "pending"]
        pool = sorted((r["task_id"] for r in rows if r["split"] == "reserve" and r["source"] == source
                       and meta.get(r["task_id"], {}).get("rank") is not None),
                      key=lambda t: meta[t]["rank"])
        taken: set[str] = set()
        for task_id in lost:
            difficulty = meta.get(task_id, {}).get("difficulty")
            open_pool = [t for t in pool if t not in taken and state(t) == "passed"]
            same = [t for t in open_pool if meta[t]["difficulty"] == difficulty]
            pick = (same or open_pool or [None])[0]
            if pick:
                taken.add(pick)
            slots.append({"replaces": task_id, "source": source, "difficulty": difficulty, "task_id": pick,
                          "rank": meta[pick]["rank"] if pick else None,
                          "same_difficulty": bool(same), "state": state(pick) if pick else "no_candidate"})
    plan = {"version": version, "manifest": manifest_path.name, "generated_at": manifest.get("generated_at"),
            "runs": runs, "slots": slots, "eval128_pending": len(pending_picks),
            "known_drops": sorted(drops),
            "complete": not pending_picks and all(s["state"] == "passed" for s in slots)}
    tmp = TOPUP_PLAN.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=1) + "\n")
    tmp.replace(TOPUP_PLAN)
    counts = collections.Counter(s["state"] for s in slots)
    print(f"eval128 top-up: {len(slots)} dropped picks -> {dict(counts)}; {len(pending_picks)} original picks "
          f"still pending; complete={plan['complete']} -> {TOPUP_PLAN}")
    return plan


def read_token(path: str = TOKEN_FILE) -> str:
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        raise SystemExit(f"token file mode {oct(mode)} is not 0600")
    return Path(path).read_text().strip()


def base_urls(path: str = URL_FILE) -> list[str]:
    return [line.strip().rstrip("/") for line in Path(path).read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def fetch(version: str) -> Path:
    token = read_token()
    last = None
    for base in base_urls():
        request = urllib.request.Request(f"{base}/tasks/manifest?version={version}",
                                         headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read()
            break
        except (urllib.error.URLError, OSError) as exc:
            last = exc
    else:
        raise SystemExit(f"manifest fetch failed on every URL: {type(last).__name__}")
    manifest = json.loads(body)
    out_dir = DATA / "t2h" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"manifest-{version}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out.write_bytes(body)
    (out_dir / f"manifest-{version}-latest.json").write_bytes(body)
    counts = collections.Counter((t.get("split"), t.get("status")) for t in manifest.get("tasks", []))
    print(f"saved {out} ({len(body):,} bytes), generated_at {manifest.get('generated_at')}")
    for (split, status), n in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        print(f"  {split:16s} {status:12s} {n}")
    return out


def workspaces(manifest_path: Path, version: str, splits: set[str] | None = None) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != version:
        raise SystemExit(f"manifest version {manifest.get('version')} != {version}")
    runs = int((manifest.get("rules") or {}).get("runs", 3))
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    status = {t["task_id"]: t.get("status") for t in manifest["tasks"]}
    plan = json.loads(TOPUP_PLAN.read_text()) if TOPUP_PLAN.exists() else {}
    if plan and plan.get("generated_at") != manifest.get("generated_at"):
        print(f"top-up plan is for the manifest of {plan.get('generated_at')}; rerun topup on this one first")
        plan = {}
    promoted = {s["task_id"]: s["replaces"] for s in plan.get("slots", []) if s.get("state") == "passed"}
    drops = known_drops()
    index = DATA / "t2h" / "store" / version / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    from cc_workspace import render_workspace  # noqa: PLC0415

    package = DATA / "t2h" / f"t2h-{version}.tar.gz"
    root = DATA / "t2h" / "workspaces"
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    rendered = 0
    for row in rows:
        if row["task_id"] in drops or not passed_all_runs(tasks.get(row["task_id"]), runs):
            continue
        split = "t2h-eval128" if row["task_id"] in promoted else row["split"]
        if splits and split not in splits:
            continue
        workspace = root / split / row["task_id"].rsplit("/", 1)[-1]
        if not (workspace / "solution" / "kernel.hip").is_file():
            if workspace.exists():
                shutil.rmtree(workspace)
            render_workspace(package, row["task_id"], workspace, task_version=version, split=split)
            rendered += 1
        label = {"task_id": row["task_id"], "split": split, "source": row.get("source"),
                 "task_version": version, "workspace_dir": str(workspace)}
        if row["task_id"] in promoted:
            label.update(eval128_topup=True, replaces=promoted[row["task_id"]])
        by_split[split].append({"prompt": row["task_id"], "label": label})
    print(f"rendered {rendered} new workspaces")
    for split, items in sorted(by_split.items()):
        if split == "t2h-eval128" and EVAL128_FROZEN.exists():
            print(f"t2h-eval128 is frozen ({EVAL128_FROZEN}); not rewritten")
            continue
        out = root / split / "manifest.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(item) + "\n" for item in items))
        tmp.replace(out)
        print(f"{split}: {len(items)} validated tasks -> {out}")
    if not by_split:
        print("no validated tasks yet; statuses seen:", collections.Counter(status.values()).most_common(8))


def freeze_eval128(manifest_path: Path, version: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    runs = int((manifest.get("rules") or {}).get("runs", 3))
    tasks = {t["task_id"]: t for t in manifest["tasks"]}
    plan = json.loads(TOPUP_PLAN.read_text())
    out = DATA / "t2h" / "workspaces" / "t2h-eval128" / "manifest.jsonl"
    raw = out.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    ids = [r["label"]["task_id"] for r in rows]
    per_source = collections.Counter(r["label"]["source"] for r in rows)
    drops = known_drops()
    problems = []
    if plan.get("generated_at") != manifest.get("generated_at") or not plan.get("complete"):
        problems.append("top-up plan is not complete for this manifest")
    if len(ids) != 2 * EVAL128_PER_SOURCE or len(set(ids)) != len(ids):
        problems.append(f"{len(ids)} rows ({len(set(ids))} distinct), want {2 * EVAL128_PER_SOURCE}")
    if any(n != EVAL128_PER_SOURCE for n in per_source.values()) or len(per_source) != 2:
        problems.append(f"per source {dict(per_source)}")
    bad = [t for t in ids if t in drops or not passed_all_runs(tasks.get(t), runs)]
    if bad:
        problems.append(f"{len(bad)} rows not passed: {bad[:5]}")
    if problems:
        raise SystemExit("eval128 not frozen: " + "; ".join(problems))
    record = {"sha256": hashlib.sha256(raw).hexdigest(), "path": str(out), "tasks": len(ids),
              "per_source": dict(per_source), "topped_up": sum(1 for r in rows if r["label"].get("eval128_topup")),
              "task_version": version, "manifest_generated_at": manifest.get("generated_at"),
              "known_drops": sorted(drops), "task_ids": sorted(ids)}
    EVAL128_FROZEN.write_text(json.dumps(record, indent=1) + "\n")
    os.chmod(out, 0o444)
    os.chmod(EVAL128_FROZEN, 0o444)
    print(f"t2h-eval128 frozen: sha256 {record['sha256']}, {len(ids)} tasks {dict(per_source)}, "
          f"{record['topped_up']} from reserve -> {EVAL128_FROZEN}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "topup", "workspaces", "freeze-eval128"])
    parser.add_argument("--splits", default="", help="workspaces: only these splits (comma list)")
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.version)
        return
    path = Path(args.manifest) if args.manifest else DATA / "t2h" / "manifests" / f"manifest-{args.version}-latest.json"
    if args.command == "topup":
        topup(path, args.version)
    elif args.command == "freeze-eval128":
        freeze_eval128(path, args.version)
    else:
        workspaces(path, args.version, {s for s in args.splits.split(",") if s} or None)


if __name__ == "__main__":
    sys.exit(main())
