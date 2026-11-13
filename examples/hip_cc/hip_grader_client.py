"""Client for the Crusoe HIP grader (plan §5.2, §5.4), stdlib only.

Used by the node-local eval-proxy (checks) and by hip_cc_agent.py (final grades).

* The bearer token is read in-process from ``HIP_SANDBOX_TOKEN_FILE`` (mode 0600); it is never
  logged, never put in an exception message and never passed to a subprocess.
* ``HIP_SF_URL_FILE`` is re-read on every attempt. It lists the relay first, then the direct
  gateway URLs; a transport error on one URL moves on to the next.
* Keep-alive connections are pooled per base URL.
* A result is ``ok`` (a reply from the grader, pass or fail), ``rejected`` (a 4xx: our request
  was wrong; never retried), ``unsupported`` (501, e.g. counters before Crusoe ships them) or
  ``infra`` (transport error, 5xx, or an ``infrastructure_error`` body: the caller retries or
  reports "unavailable").
"""

from __future__ import annotations

import http.client
import json
import os
import queue
import stat
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_URL_FILE = "/shared_inference/yuankach/glm53/hip-cc/crusoe_relay_url.txt"
DEFAULT_TOKEN_FILE = os.path.expanduser("~/.hip_cc/crusoe_grader_token")


def url_file() -> str:
    return os.environ.get("HIP_SF_URL_FILE", DEFAULT_URL_FILE)


def token_file() -> str:
    return os.environ.get("HIP_SANDBOX_TOKEN_FILE") or os.environ.get("HIP_SF_TOKEN_FILE") or DEFAULT_TOKEN_FILE


def read_token(path: str | None = None) -> str:
    path = path or token_file()
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        raise PermissionError(f"grader token file mode {oct(mode)} is not 0600")
    return Path(path).read_text().strip()


def read_urls(path: str | None = None) -> list[str]:
    try:
        text = Path(path or url_file()).read_text()
    except OSError:
        return []
    return [line.strip().rstrip("/") for line in text.splitlines() if line.strip() and not line.startswith("#")]


@dataclass
class Result:
    kind: str  # ok | rejected | unsupported | infra
    status: int = 0
    body: dict | None = None
    error: str = ""
    url: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


@dataclass
class Stats:
    calls: int = 0
    ok: int = 0
    infra: int = 0
    rejected: int = 0
    unsupported: int = 0
    consecutive_infra: int = 0
    last_ok: float = 0.0
    last_infra: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, result: Result) -> None:
        with self.lock:
            self.calls += 1
            setattr(self, result.kind, getattr(self, result.kind) + 1)
            if result.kind == "infra":
                self.consecutive_infra += 1
                self.last_infra = time.time()
            else:
                self.consecutive_infra = 0
                self.last_ok = time.time()

    def snapshot(self) -> dict:
        with self.lock:
            return {k: getattr(self, k) for k in ("calls", "ok", "infra", "rejected", "unsupported",
                                                   "consecutive_infra",
                                                   "last_ok", "last_infra")}


class GraderClient:
    def __init__(self, url_path: str | None = None, token_path: str | None = None, *, pool_size: int = 8,
                 timeout: float = 900.0):
        self.url_path = url_path
        self.token_path = token_path
        self.timeout = timeout
        self.pool_size = pool_size
        self._pools: dict[str, queue.LifoQueue] = {}
        self._lock = threading.Lock()
        self.stats = Stats()

    def _connection(self, base: str) -> http.client.HTTPConnection:
        with self._lock:
            pool = self._pools.setdefault(base, queue.LifoQueue(self.pool_size))
        try:
            return pool.get_nowait()
        except queue.Empty:
            parts = urllib.parse.urlsplit(base)
            cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
            return cls(parts.hostname, parts.port, timeout=self.timeout)

    def _release(self, base: str, conn: http.client.HTTPConnection) -> None:
        try:
            self._pools[base].put_nowait(conn)
        except queue.Full:
            conn.close()

    def _once(self, base: str, method: str, path: str, payload: dict | None) -> Result:
        token = read_token(self.token_path)
        headers = {"Authorization": f"Bearer {token}", "Connection": "keep-alive"}
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        prefix = urllib.parse.urlsplit(base).path.rstrip("/")
        started = time.time()
        for fresh in (False, True):
            conn = self._connection(base)
            if fresh:
                conn.close()
            try:
                conn.request(method, prefix + path, body=data, headers=headers)
                response = conn.getresponse()
                raw = response.read()
                status = response.status
                if response.getheader("connection", "").lower() == "close":
                    conn.close()
                else:
                    self._release(base, conn)
                break
            except (OSError, http.client.HTTPException) as exc:
                conn.close()
                if fresh:
                    return Result("infra", error=f"transport: {type(exc).__name__}", url=base,
                                  seconds=time.time() - started)
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {"text": raw[:2000].decode(errors="replace")}
        if not isinstance(body, dict):
            body = {"value": body}
        seconds = time.time() - started
        if status == 501:
            return Result("unsupported", status, body, "HTTP 501", base, seconds)
        infra_body = body.get("infrastructure_error") or body.get("error") == "infrastructure_error"
        if status >= 500 or infra_body:
            return Result("infra", status, body, f"HTTP {status}", base, seconds)
        if status >= 400:
            return Result("rejected", status, body, f"HTTP {status}", base, seconds)
        return Result("ok", status, body, "", base, seconds)

    def call(self, method: str, path: str, payload: dict | None = None) -> Result:
        """One attempt over the URL list: the first URL that answers decides."""
        urls = read_urls(self.url_path)
        if not urls:
            result = Result("infra", error="no grader URLs in the URL file")
            self.stats.record(result)
            return result
        result = None
        for base in urls:
            result = self._once(base, method, path, payload)
            if not (result.kind == "infra" and result.status == 0):
                break
        self.stats.record(result)
        return result

    def call_with_retry(self, method: str, path: str, payload: dict | None = None, *,
                        retries: int = 1, delay_s: float = 5.0, sleep=time.sleep) -> Result:
        result = self.call(method, path, payload)
        for _ in range(retries):
            if result.kind != "infra":
                break
            sleep(delay_s)
            result = self.call(method, path, payload)
        return result
