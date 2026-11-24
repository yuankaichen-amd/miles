"""hip_grader_client timeout handling against a local HTTP server: terminal timeouts for checks,
the unchanged default retries used by final grades, and the per-call timeout on pooled connections.

    PYTHONPATH=<miles> python3 -m pytest -q examples/hip_cc/tests/test_hip_grader_client.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

from hip_grader_client import GraderClient  # noqa: E402

SLOW_S = 1.0


class Handler(BaseHTTPRequestHandler):
    hits: list[str] = []

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        Handler.hits.append(f"{self.server.server_port}{self.path}")
        if self.path.endswith("/slow"):
            time.sleep(SLOW_S)
            status = 200
        elif self.path.endswith("/gw504"):
            status = 504
        else:
            status = 200
        body = json.dumps({"compiled": True, "correct": True}).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def servers():
    started = []
    for _ in range(2):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
    Handler.hits = []
    yield started
    for server in started:
        server.shutdown()
        server.server_close()


def make_client(tmp_path, servers, *, n_urls=1, timeout=900.0):
    token = tmp_path / "token"
    token.write_text("t0k3n")
    os.chmod(token, 0o600)
    urls = tmp_path / "urls"
    urls.write_text("\n".join(f"http://127.0.0.1:{s.server_port}" for s in servers[:n_urls]) + "\n")
    return GraderClient(str(urls), str(token), timeout=timeout)


def test_terminal_timeout_sends_one_request(tmp_path, servers):
    client = make_client(tmp_path, servers, n_urls=2)
    started = time.time()
    result = client.call_with_retry("POST", "/tool/slow", {"x": 1}, retries=1, delay_s=0,
                                    timeout=0.2, terminal_timeout=True)
    assert result.kind == "infra" and result.timed_out and "Timeout" in result.error
    assert len(Handler.hits) == 1, f"no fresh-connection retry, no next URL, no repeat: {Handler.hits}"
    assert time.time() - started < SLOW_S


def test_default_retries_are_unchanged_for_final_grades(tmp_path, servers):
    client = make_client(tmp_path, servers, n_urls=1, timeout=0.2)
    result = client.call_with_retry("POST", "/tool/slow", {"x": 1}, retries=1, delay_s=0)
    assert result.kind == "infra" and result.timed_out
    assert len(Handler.hits) == 4, "fresh-connection retry x call_with_retry repeat, as before"


def test_default_moves_on_to_the_next_url_after_a_timeout(tmp_path, servers):
    client = make_client(tmp_path, servers, n_urls=2, timeout=0.2)
    client.call("POST", "/tool/slow", {"x": 1})
    ports = {hit.split("/")[0] for hit in Handler.hits}
    assert len(Handler.hits) == 4 and len(ports) == 2


def test_504_is_a_terminal_timeout_but_retried_by_default(tmp_path, servers):
    client = make_client(tmp_path, servers)
    result = client.call_with_retry("POST", "/tool/gw504", {}, retries=1, delay_s=0, terminal_timeout=True)
    assert result.kind == "infra" and result.timed_out and result.status == 504
    assert len(Handler.hits) == 1
    Handler.hits = []
    client.call_with_retry("POST", "/tool/gw504", {}, retries=1, delay_s=0)
    assert len(Handler.hits) == 2


def test_per_call_timeout_applies_to_a_pooled_connection(tmp_path, servers):
    client = make_client(tmp_path, servers, timeout=900.0)
    assert client.call("POST", "/tool/fast", {}).kind == "ok"
    started = time.time()
    result = client.call("POST", "/tool/slow", {}, timeout=0.2, terminal_timeout=True)
    assert result.timed_out and time.time() - started < SLOW_S
    # The next call on the pool gets the default timeout back, so a slow reply succeeds.
    assert client.call("POST", "/tool/slow", {}).kind == "ok"


def test_ok_reply_is_unaffected(tmp_path, servers):
    client = make_client(tmp_path, servers)
    result = client.call_with_retry("POST", "/tool/fast", {}, timeout=5.0, terminal_timeout=True)
    assert result.kind == "ok" and not result.timed_out and result.body["correct"] is True
