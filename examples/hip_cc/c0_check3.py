"""C0 check 3 from a useocp compute node: relay health, then (optionally) the task-store upload.

The bearer token is read from ``$HIP_SF_TOKEN_FILE`` (default ~/.hip_cc/crusoe_grader_token)
inside this process; it never appears on argv or in output.

    python3 c0_check3.py [--upload <t2h-<version>.tar.gz>]
"""

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL_FILE = os.environ.get("HIP_SF_URL_FILE", "/shared_inference/yuankach/glm53/hip-cc/crusoe_relay_url.txt")
TOKEN_FILE = os.environ.get("HIP_SF_TOKEN_FILE", os.path.expanduser("~/.hip_cc/crusoe_grader_token"))


def call(url, token=None, data=None, content_type=None, timeout=60):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()
    return status, body.decode(errors="replace"), round(time.time() - started, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", default=None)
    args = parser.parse_args()
    relay = Path(URL_FILE).read_text().strip().rstrip("/")
    mode = stat.S_IMODE(os.stat(TOKEN_FILE).st_mode)
    if mode & 0o077:
        sys.exit(f"token file mode {oct(mode)} is not 0600")
    token = Path(TOKEN_FILE).read_text().strip()
    ok = True

    status, body, dt = call(f"{relay}/healthz")
    print(f"GET /healthz (no auth): {status} in {dt}s: {body[:300]}")
    ok &= status == 200
    status, body, dt = call(f"{relay}/health")
    print(f"GET /health (no auth, expect 401): {status}")
    ok &= status == 401
    status, body, dt = call(f"{relay}/health", token)
    print(f"GET /health (auth): {status} in {dt}s: {body[:600]}")
    try:
        executors = json.loads(body).get("executors") or []
        ok &= status == 200 and bool(executors) and all(e["workers"] == e["healthy"] == 8 for e in executors)
    except ValueError:
        ok = False

    if args.upload:
        tarball = Path(args.upload)
        data = tarball.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected = Path(str(tarball) + ".sha256").read_text().split()[0]
        version = tarball.name.removeprefix("t2h-").removesuffix(".tar.gz").removesuffix("-feasibility")
        print(f"tarball {tarball} {len(data):,} bytes sha256 {digest} (local .sha256 match: {digest == expected})")
        ok &= digest == expected
        query = f"version={version}" + ("&part=feasibility" if tarball.name.endswith("-feasibility.tar.gz") else "")
        status, body, dt = call(f"{relay}/tasks/upload?{query}", token, data, "application/gzip", timeout=1800)
        print(f"POST /tasks/upload?{query}: {status} in {dt}s: {body[:3000]}")
        ok &= status == 200 and json.loads(body).get("verified") is True
        status, body, dt = call(f"{relay}/tasks/manifest?version={version}", token)
        print(f"GET /tasks/manifest: {status}: {body[:1500]}")
    print("C0 check 3:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
