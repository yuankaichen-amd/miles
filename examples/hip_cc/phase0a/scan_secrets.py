"""Stop-condition scan: no token or Authorization header in any Phase 0a log.

Looks in the given directories (recursively, text files) for the literal values of the grader,
runner and gateway-admin tokens, and for a bearer credential or an Authorization header field.
Prints only file names, line numbers and the kind of match, never the matched text.

    python3 scan_secrets.py --secret-file F [--secret-file F ...] DIR [DIR ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PATTERNS = {
    "bearer_credential": re.compile(rb"[Bb]earer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "authorization_field": re.compile(rb"""["']?authorization["']?\s*[:=]\s*["']?[A-Za-z]""", re.IGNORECASE),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret-file", action="append", default=[])
    parser.add_argument("--max-bytes", type=int, default=4 << 30)
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args()
    secrets = [(Path(p).name, Path(p).read_bytes().strip()) for p in args.secret_file if Path(p).is_file()]
    secrets = [(name, value) for name, value in secrets if len(value) >= 16]
    hits, scanned = [], 0
    for root in args.dirs:
        for path in sorted(Path(root).rglob("*")):
            if not path.is_file() or path.stat().st_size > args.max_bytes:
                continue
            scanned += 1
            with path.open("rb") as handle:
                for number, line in enumerate(handle, 1):
                    for name, value in secrets:
                        if value in line:
                            hits.append({"file": str(path), "line": number, "kind": f"secret:{name}"})
                    for kind, pattern in PATTERNS.items():
                        if pattern.search(line):
                            hits.append({"file": str(path), "line": number, "kind": kind})
    print(json.dumps({"files_scanned": scanned, "secrets_checked": [n for n, _ in secrets], "hits": hits[:200],
                      "hit_count": len(hits)}, indent=1))
    print("SECRET SCAN:", "CLEAN" if not hits else "HITS")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())
