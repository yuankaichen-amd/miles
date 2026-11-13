"""Mock Crusoe HIP grader for the cc-runner dry run (stdlib only).

Routes as in plan §5.2: /healthz, /health, /tool/correctness, /tool/performance, /tool/counters
(501, as the real pool returns today), /grade. Correctness is decided by content: a kernel that
clamps at 6 passes (the relu6 test kernels). ``--fail-first N`` answers the first N tool calls with
502 infrastructure_error to exercise the eval-proxy retry. Requests are logged without code.

    python3 mock_grader.py --port P --token-file F --log L [--fail-first N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class State:
    def __init__(self, token: str, fail_first: int, log: Path):
        self.token = token
        self.fail_left = fail_first
        self.log = log.open("a", buffering=1)
        self.seen: dict[tuple[str, int], dict] = {}
        self.lock = threading.Lock()


def verdict(code: str) -> dict:
    compiled = "out[i] = v;" in code or "out[i] = v\n" not in code
    correct = compiled and "6.f" in code and "hipDeviceSynchronize" not in code
    return {"compiled": compiled, "correct": correct,
            "audit": "hipDeviceSynchronize" not in code}


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _auth(self) -> bool:
            return self.headers.get("Authorization", "") == f"Bearer {state.token}"

        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, {"ok": True})
            if not self._auth():
                return self._send(401, {"error": "unauthorized"})
            if self.path == "/health":
                return self._send(200, {"executors": [{"executor": "aka_hip", "workers": 8, "healthy": 8}]})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if not self._auth():
                return self._send(401, {"error": "unauthorized"})
            code = body.get("code", "")
            entry = {"t": time.time(), "path": self.path, "episode": body.get("episode"), "seq": body.get("seq"),
                     "task_id": body.get("task_id"), "sha": hashlib.sha256(code.encode()).hexdigest()[:16]}
            with state.lock:
                if self.path.startswith("/tool/") and state.fail_left > 0:
                    state.fail_left -= 1
                    entry["injected"] = 502
                    state.log.write(json.dumps(entry) + "\n")
                    return self._send(502, {"infrastructure_error": "injected failure"})
                key = (str(body.get("episode")), int(body.get("seq") or 0))
                if self.path.startswith("/tool/") and key in state.seen:
                    entry["dedupe"] = True
                    state.log.write(json.dumps(entry) + "\n")
                    return self._send(200, state.seen[key])
                state.log.write(json.dumps(entry) + "\n")
            v = verdict(code)
            if self.path == "/tool/counters":
                return self._send(501, {"error": "counters not implemented"})
            if self.path == "/tool/correctness":
                if not v["compiled"]:
                    reply = {"passed": False, "failure_kind": "compile_error", "summary": "compile_error: expected ';'"}
                elif not v["audit"]:
                    reply = {"passed": False, "failure_kind": "policy_violation",
                             "summary": "policy_violation: hipDeviceSynchronize is not allowed"}
                elif not v["correct"]:
                    reply = {"passed": False, "failure_kind": "numerical_mismatch",
                             "summary": "FAIL w0: out max error 3.2 (bound 1.2e-7) at [0, 0, 3, 17]"}
                else:
                    reply = {"passed": True, "summary": "PASS: w0, w1 within bound over 2 draws"}
            elif self.path == "/tool/performance":
                reply = {"summary": "w0: candidate 1.010 ms, baseline 1.420 ms (1.41x)\n"
                                    "w1: candidate 0.260 ms, baseline 0.361 ms (1.39x)\ngeomean speedup 1.40x"}
            elif self.path == "/grade":
                s = 1.4 if v["correct"] else 0.0
                reply = {"compiled": v["compiled"], "correct": v["correct"], "audit": v["audit"],
                         "method_consistent": True, "speedup": s, "score": max(0.0, 1 - 1.05 / s) if s else 0.0}
            else:
                return self._send(404, {"error": "not found"})
            if self.path.startswith("/tool/"):
                with state.lock:
                    state.seen[key] = reply
            return self._send(200, reply)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--fail-first", type=int, default=0)
    args = parser.parse_args()
    state = State(Path(args.token_file).read_text().strip(), args.fail_first, Path(args.log))
    ThreadingHTTPServer((args.host, args.port), make_handler(state)).serve_forever()


if __name__ == "__main__":
    main()
