"""Container part of the workspace isolation test (needs docker; stdlib only, runs on the host).

For each workspace rendered by ``test_workspace_isolation.py`` (``$ISOLATION_OUT``), starts a
rollout container with ``cc_workspace.container_args``, streams the workspace into its tmpfs
``/workspace`` and runs a scanner inside it. The scanner gets only hashes, never hip_ref
text, and fails on:

* any path in the container's filesystem naming ``hip_ref`` or a feasibility tarball;
* any host mount: an nfs filesystem, ``/shared_inference``, ``/shared_nfs`` or ``/home/<user>``;
* any file under /workspace, /tmp, /home, /kb whose content (whole or any line) hashes to
  a hip_ref needle or contains a row_sha256;
* any network path out (relay and grader addresses must be unreachable).

    python3 run_container_isolation.py --out $ISOLATION_OUT --image <image>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cc_workspace  # noqa: E402

SCANNER = r'''
import hashlib, json, os, re, socket, sys
cfg = json.loads(sys.stdin.readline())
lines, whole, rows = set(cfg["line_sha256"]), set(cfg["whole_sha256"]), cfg["row_sha256"]
bad = []
forbidden = re.compile(r"hip_ref|t2h-[0-9a-f]{16}-feasibility", re.I)
for root, dirs, files in os.walk("/", onerror=lambda e: None):
    if root in ("/proc", "/sys", "/dev") or root.startswith(("/proc/", "/sys/", "/dev/")):
        dirs[:] = []
        continue
    for name in dirs + files:
        if forbidden.search(name):
            bad.append(("path", os.path.join(root, name)))
for p in ("/shared_inference", "/shared_nfs", "/home/yuankach"):
    if os.path.exists(p):
        bad.append(("host_path_exists", p))
for line in open("/proc/self/mountinfo"):
    fields = line.split()
    sep = fields.index("-")
    fstype, source, point = fields[sep + 1], fields[sep + 2], fields[4]
    if fstype.startswith("nfs") or "shared_inference" in line or "/home/yuankach" in line:
        bad.append(("mount", f"{point} {fstype} {source}"))
scanned = 0
for top in ("/workspace", "/tmp", "/home", "/kb"):
    for root, dirs, files in os.walk(top, onerror=lambda e: None):
        for name in files:
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > 8 << 20:
                    continue
                data = open(path, "rb").read()
            except OSError:
                continue
            scanned += 1
            text = data.decode("utf-8", "replace")
            if hashlib.sha256(text.encode()).hexdigest() in whole:
                bad.append(("whole_file", path))
            if {hashlib.sha256(l.strip().encode()).hexdigest() for l in text.splitlines()} & lines:
                bad.append(("hip_ref_line", path))
            if any(r in text for r in rows) or "hip_ref" in text:
                bad.append(("marker", path))
reachable = []
for host, port in cfg["endpoints"]:
    s = socket.socket(); s.settimeout(3)
    try:
        s.connect((host, port)); reachable.append(f"{host}:{port}")
    except OSError:
        pass
    finally:
        s.close()
bad += [("network", r) for r in reachable]
ws = sorted(os.path.relpath(os.path.join(r, f), "/workspace") for r, _, fs in os.walk("/workspace") for f in fs)
print(json.dumps({"bad": bad, "scanned_files": scanned, "workspace_files": ws, "uid": os.getuid()}))
'''


def docker(*args, input_bytes=None, check=True, timeout=600):
    proc = subprocess.run(["docker", *args], input=input_bytes, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker {args[:2]} failed: {proc.stderr.decode()[-2000:]}")
    return proc.stdout.decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--image", default="rocm/sgl-dev:miles-rocm10-mi30x-20261001")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    cfg = json.loads((Path(args.out) / "needles.json").read_text())
    relay = Path("/shared_inference/yuankach/glm53/hip-cc/crusoe_relay_url.txt").read_text().strip()
    host, port = relay.split("//")[1].rsplit(":", 1)
    scan_cfg = {k: cfg[k] for k in ("line_sha256", "whole_sha256", "row_sha256")}
    scan_cfg["endpoints"] = [[host, int(port)], ["10.245.149.31", 18300], ["8.8.8.8", 53], ["1.1.1.1", 443]]
    items = list(cfg["workspaces"].items())[: args.limit or None]
    failures, results = 0, []
    for index, (task_id, workspace) in enumerate(items):
        report = scan(f"ccws_iso_{os.getpid()}_{index}", args.image, Path(workspace), scan_cfg)
        ok = not report.get("bad") and report.get("workspace_files") == sorted(cc_workspace.WORKSPACE_FILES)
        failures += not ok
        results.append({"task_id": task_id, "ok": ok, **report})
        print(json.dumps(results[-1])[:1500], flush=True)
    planted = Path(cfg["planted"])
    try:
        control = scan(f"ccws_iso_{os.getpid()}_planted", args.image, planted, scan_cfg)
    finally:
        shutil.rmtree(planted, ignore_errors=True)
    kinds = {kind for kind, _ in control.get("bad", [])}
    control_ok = {"whole_file", "hip_ref_line"} <= kinds
    print(f"positive control (hip_ref planted as kernel.hip): flagged {sorted(kinds)} -> "
          f"{'detected' if control_ok else 'MISSED'}")
    print(f"container isolation: {len(items) - failures}/{len(items)} workspaces clean")
    (Path(args.out) / "container-report.json").write_text(json.dumps(
        {"workspaces": results, "positive_control": control}, indent=1))
    sys.exit(1 if failures or not control_ok else 0)


def scan(name: str, image: str, workspace: Path, scan_cfg: dict) -> dict:
    run_args = cc_workspace.container_args(name, image, os.getuid(), os.getgid())
    try:
        subprocess.run(run_args, check=True, capture_output=True, timeout=600)
        docker("exec", "-i", name, "tar", "-x", "-C", "/workspace", input_bytes=cc_workspace.workspace_tar(workspace))
        out = subprocess.run(["docker", "exec", "-i", name, "python3", "-c", SCANNER],
                             input=(json.dumps(scan_cfg) + "\n").encode(), capture_output=True, timeout=1200)
        if out.returncode != 0:
            return {"bad": [("scanner_failed", out.stderr.decode()[-500:])]}
        return json.loads(out.stdout.decode())
    finally:
        docker("rm", "-f", name, check=False)


if __name__ == "__main__":
    main()
