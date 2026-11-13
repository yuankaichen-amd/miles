"""The gfx950-validated task lists for rollouts (phase 0a start condition 4).

    python3 t2h_manifest.py fetch [--version V]      save GET /tasks/manifest?version=V
    python3 t2h_manifest.py workspaces [--manifest F] render validated tasks and write
                                                      $DATA/t2h/workspaces/<split>/manifest.jsonl

``workspaces`` keeps only tasks the Crusoe manifest marks valid, in the order of the P2 index.
Each one is rendered once with ``cc_workspace.render_workspace`` (the only path from the task
store to an agent) into ``$DATA/t2h/workspaces/<split>/<name>/``. Manifest rows are Miles
prompt-data rows: ``{"prompt": task_id, "label": {task_id, split, source, task_version,
workspace_dir}}``. Run on a compute node, in the Miles image for ``workspaces`` (it needs PyYAML
and AKA); ``fetch`` reads the token in-process from ``$HIP_SANDBOX_TOKEN_FILE``.
"""

from __future__ import annotations

import argparse
import collections
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


def workspaces(manifest_path: Path, version: str) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != version:
        raise SystemExit(f"manifest version {manifest.get('version')} != {version}")
    status = {t["task_id"]: t.get("status") for t in manifest["tasks"]}
    index = DATA / "t2h" / "store" / version / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    from cc_workspace import render_workspace  # noqa: PLC0415

    package = DATA / "t2h" / f"t2h-{version}.tar.gz"
    root = DATA / "t2h" / "workspaces"
    by_split: dict[str, list[dict]] = collections.defaultdict(list)
    rendered = 0
    for row in rows:
        if status.get(row["task_id"]) not in VALID_STATUS:
            continue
        workspace = root / row["split"] / row["task_id"].rsplit("/", 1)[-1]
        if not (workspace / "solution" / "kernel.hip").is_file():
            if workspace.exists():
                shutil.rmtree(workspace)
            render_workspace(package, row["task_id"], workspace, task_version=version, split=row["split"])
            rendered += 1
        label = {"task_id": row["task_id"], "split": row["split"], "source": row.get("source"),
                 "task_version": version, "workspace_dir": str(workspace)}
        by_split[row["split"]].append({"prompt": row["task_id"], "label": label})
    print(f"rendered {rendered} new workspaces")
    for split, items in sorted(by_split.items()):
        out = root / split / "manifest.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(item) + "\n" for item in items))
        tmp.replace(out)
        print(f"{split}: {len(items)} validated tasks -> {out}")
    if not by_split:
        print("no validated tasks yet; statuses seen:", collections.Counter(status.values()).most_common(8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "workspaces"])
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.version)
    else:
        path = Path(args.manifest) if args.manifest else DATA / "t2h" / "manifests" / f"manifest-{args.version}-latest.json"
        workspaces(path, args.version)


if __name__ == "__main__":
    sys.exit(main())
