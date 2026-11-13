"""Relay /health from the node, before any container starts (Phase 0a plan, readiness item 1).

Exit 0 only if the authenticated /health answers 200 and every executor it lists is aka_hip with
all of its workers healthy. Prints the reply; the token is read from the 0600 file and never shown.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from c0_check3 import call  # noqa: E402

URL_FILE = os.environ["HIP_SF_URL_FILE"]
TOKEN_FILE = os.environ["HIP_SANDBOX_TOKEN_FILE"]


def main() -> int:
    relay = Path(URL_FILE).read_text().split()[0].rstrip("/")
    if stat.S_IMODE(os.stat(TOKEN_FILE).st_mode) & 0o077:
        print(f"relay health: {TOKEN_FILE} is not 0600")
        return 1
    token = Path(TOKEN_FILE).read_text().strip()
    try:
        status, body, seconds = call(f"{relay}/health", token, timeout=30)
    except OSError as exc:
        print(f"relay health: {relay} unreachable: {type(exc).__name__}: {exc}")
        return 1
    try:
        executors = json.loads(body).get("executors") or []
    except ValueError:
        executors = []
    ok = status == 200 and bool(executors) and all(
        e.get("executor", e.get("name")) == "aka_hip" and e.get("healthy") == e.get("workers") and e.get("workers", 0) > 0
        for e in executors)
    print(f"relay health {relay}: HTTP {status} in {seconds}s: {body[:600]}")
    print("relay health:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
