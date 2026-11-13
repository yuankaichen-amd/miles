"""cc-runner: one per rollout node; runs Claude Code episodes in hip-cc-rollout containers (plan §4.2, §5.4, §7.3).

Stdlib only: it runs on the node's host Python, next to the docker CLI.

Per episode:

* a container from ``cc_workspace.container_args`` with ``--network none`` and a non-root uid
  that is not ours (container threads would otherwise count against our per-user process limit
  on the host); the workspace is streamed into its tmpfs ``/workspace`` with tar;
* a socket directory bound read-only at ``/fwd`` with two unix sockets served here:
  ``gateway.sock`` forwards only ``POST /e/<episode>/v1/messages[...]`` to the cc-gateway and logs
  every other request as a denial; ``eval.sock`` is the eval-proxy;
* inside, ``uds_relay.py`` maps 127.0.0.1:8080 and :8081 to those sockets, and ``claude -p`` runs
  under ``timeout`` with the 2.1.311 settings from plan §2.4;
* on exit (the container calls ``/done`` on the eval-proxy) or at the wall clock, the runner
  collects ``/workspace`` and ``/tmp/cc`` with one ``docker exec tar`` and removes the container.

The eval-proxy owns the check budgets (compile 20, correctness 10, performance 4, counters 1
after correctness passed on the same code), dedupes by code hash and forwards checks to the
Crusoe HIP relay with ``(episode, seq)``. A transport error, a 5xx or an infrastructure_error is
retried once after 5 s, re-reading ``HIP_SF_URL_FILE``; if it still fails the agent is told
"check unavailable (infrastructure); not counted against your budget". 501 (counters before
Crusoe ships them) is "unavailable", never charged, and never feeds the grade.

API (bearer token from ``CC_RUNNER_TOKEN_FILE``):
    POST   /episodes                  start an episode (JSON EpisodeSpec); 202, 409, 503
    GET    /episodes/<id>?wait=<s>    long-poll for the result
    DELETE /episodes/<id>             cancel
    GET    /health                    capacity, running episodes and grader stats
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import difflib
import hashlib
import hmac
import io
import json
import logging
import os
import re
import secrets
import shutil
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cc_http  # noqa: E402
from cc_workspace import check_path_allowed, container_args, workspace_tar  # noqa: E402
from hip_grader_client import GraderClient  # noqa: E402

logger = logging.getLogger("cc_runner")

CONTAINER_UID = int(os.environ.get("CC_CONTAINER_UID", "10001"))
CHECK_ACTIONS = ("correctness", "performance", "counters")
SUMMARY_CAP = {"correctness": 2500, "performance": 2500, "counters": 1500}
DEFAULT_BUDGETS = {"compile": 20, "correctness": 10, "performance": 4, "counters": 1}
TOOLS = "Bash,Read,Write,Edit,Glob,Grep"
DISALLOWED_TOOLS = "Task,Agent,WebFetch,WebSearch,NotebookEdit,TodoWrite,KillShell,BashOutput"
COLLECT_LIMIT = 64 * 1024 * 1024
SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def container_labels() -> dict[str, str]:
    """``cc.role=rollout`` plus ``CC_CONTAINER_LABEL`` (``key=value``), so cc_node.sh can clean up its job."""
    labels = {"cc.role": "rollout"}
    extra = os.environ.get("CC_CONTAINER_LABEL", "")
    if "=" in extra:
        key, value = extra.split("=", 1)
        labels[key] = value
    return labels


def safe_name(text: str) -> str:
    return SAFE.sub("_", text)[:120]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", buffering=1)

    def __call__(self, event: str, **fields) -> None:
        self.handle.write(json.dumps({"t": round(time.time(), 3), "event": event, **fields}) + "\n")

    def close(self) -> None:
        self.handle.close()


@dataclass
class EpisodeSpec:
    episode: str
    task_id: str
    task_version: str
    workspace_dir: str
    gateway_url: str
    gateway_token: str = field(repr=False)
    model: str
    split: str = "train"
    prompt: str = ""
    max_turns: int = 40
    wall_s: int = 1200
    max_output_tokens: int = 16384
    budgets: dict = field(default_factory=dict)
    counters_enabled: bool = True
    kb: bool = True
    effort: str = ""

    @classmethod
    def from_json(cls, body: dict) -> "EpisodeSpec":
        known = set(cls.__dataclass_fields__)
        missing = [k for k in ("episode", "task_id", "task_version", "workspace_dir", "gateway_url",
                               "gateway_token", "model") if not body.get(k)]
        if missing:
            raise cc_http.HTTPError(400, f"missing fields: {missing}")
        spec = cls(**{k: v for k, v in body.items() if k in known})
        check_path_allowed(spec.workspace_dir)
        if not Path(spec.workspace_dir, "solution", "kernel.hip").is_file():
            raise cc_http.HTTPError(400, "workspace_dir is not a rendered workspace")
        if not spec.kb and "/kb" in spec.prompt:
            raise cc_http.HTTPError(400, "a /kb-off episode's prompt must not mention /kb")
        return spec

    def public(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k not in ("gateway_token", "prompt")}


# ---------------------------------------------------------------- eval-proxy


def _passed(body: dict) -> bool:
    for key in ("passed", "correct", "pass"):
        if isinstance(body.get(key), bool):
            return body[key]
    return str(body.get("status", body.get("verdict", ""))).lower() in ("pass", "passed", "ok", "correct")


def grader_error(body: dict) -> str | None:
    """A 200 reply that reports the grader's own failure (e.g. a profiling timeout), not a verdict."""
    candidates = [body]
    for key in ("summary", "text"):
        text = body.get(key)
        if isinstance(text, str) and text.lstrip().startswith("{"):
            try:
                inner = json.loads(text)
            except ValueError:
                continue
            if isinstance(inner, dict):
                candidates.append(inner)
    for item in candidates:
        if str(item.get("status", "")).lower() == "error" and "compiled" not in item and "correct" not in item:
            return str(item.get("detail") or item.get("error") or "grader error")[:200]
    return None


def summarize(action: str, body: dict, cap: int) -> str:
    text = body.get("summary") or body.get("text")
    if not isinstance(text, str):
        shown = {k: v for k, v in body.items() if k not in ("episode", "seq", "task_version", "code")}
        text = json.dumps(shown, separators=(",", ":"))
    return text if len(text) <= cap else text[: cap - 15] + "\n[truncated]"


class EvalProxy:
    def __init__(self, spec: EpisodeSpec, client: GraderClient, log: JsonlLog, pool, *, retry_delay_s: float = 5.0,
                 denial_log: JsonlLog | None = None):
        self.spec = spec
        self.client = client
        self.log = log
        self.denial_log = denial_log
        self.pool = pool
        self.retry_delay_s = retry_delay_s
        self.token = secrets.token_urlsafe(24)
        self.budgets = {**DEFAULT_BUDGETS, **(spec.budgets or {})}
        self.used = dict.fromkeys(self.budgets, 0)
        self.unavailable = dict.fromkeys(CHECK_ACTIONS, 0)
        self.cache: dict[tuple[str, str], str] = {}
        self.passed: set[str] = set()
        self.compiled_ok: set[str] = set()
        self.seq = 0
        self.lock = asyncio.Lock()
        self.done = asyncio.Event()
        self.done_info: dict = {}
        self.denials = 0

    def status_text(self) -> str:
        return "budgets used: " + ", ".join(f"{k} {self.used[k]}/{self.budgets[k]}" for k in self.budgets)

    def deny(self, **fields) -> None:
        self.denials += 1
        self.log("eval_denied", **fields)
        if self.denial_log is not None:
            self.denial_log("eval_denied", episode=self.spec.episode, **fields)

    async def handle(self, request: cc_http.Request) -> bytes:
        if not hmac.compare_digest(request.bearer(), self.token):
            self.deny(method=request.method, path=request.path[:200], reason="token")
            return cc_http.json_response(401, {"ok": False, "text": "unauthorized"})
        if request.method != "POST":
            self.deny(method=request.method, path=request.path[:200], reason="method")
            raise cc_http.HTTPError(405, "POST only")
        body = request.json()
        if request.path == "/compile":
            return cc_http.json_response(200, self.compile_grant(str(body.get("sha256", ""))))
        if request.path == "/compile/result":
            if body.get("ok"):
                self.compiled_ok.add(str(body.get("sha256", "")))
            self.log("compile_result", sha=str(body.get("sha256", ""))[:16], ok=bool(body.get("ok")))
            return cc_http.json_response(200, {"ok": True})
        if request.path == "/status":
            return cc_http.json_response(200, {"ok": True, "text": self.status_text()})
        if request.path == "/check":
            action, code = body.get("action"), body.get("code")
            if action not in CHECK_ACTIONS or not isinstance(code, str):
                raise cc_http.HTTPError(400, "action must be correctness|performance|counters with code")
            ok, text = await self.check(action, code)
            return cc_http.json_response(200, {"ok": ok, "text": text})
        if request.path == "/done":
            self.done_info = {k: body.get(k) for k in ("exit", "reason")}
            self.log("done", **self.done_info)
            self.done.set()
            return cc_http.json_response(200, {"ok": True})
        self.deny(method=request.method, path=request.path[:200], reason="route")
        raise cc_http.HTTPError(404, "unknown route")

    def compile_grant(self, sha: str) -> dict:
        if self.used["compile"] >= self.budgets["compile"]:
            self.log("compile", sha=sha[:16], granted=False)
            return {"ok": False, "text": f"compile budget exhausted ({self.budgets['compile']} used)"}
        self.used["compile"] += 1
        self.log("compile", sha=sha[:16], granted=True, used=self.used["compile"])
        return {"ok": True, "text": f"compile {self.used['compile']}/{self.budgets['compile']}"}

    async def check(self, action: str, code: str) -> tuple[bool, str]:
        async with self.lock:
            return await self._check(action, code)

    async def _check(self, action: str, code: str) -> tuple[bool, str]:
        digest = sha256(code)
        cached = self.cache.get((action, digest))
        if cached is not None:
            self.log("check", action=action, sha=digest[:16], kind="cached", charged=False)
            return True, cached + f"\n(same code as an earlier {action} check; not charged)"
        if action == "counters" and not self.spec.counters_enabled:
            self.unavailable[action] += 1
            return False, "counters unavailable in this run; not counted against your budget"
        if action == "counters" and digest not in self.passed:
            return False, "counters needs a passing correctness check on this exact code first; not charged"
        if self.used[action] >= self.budgets[action]:
            self.log("check", action=action, sha=digest[:16], kind="budget", charged=False)
            return False, f"{action} budget exhausted ({self.budgets[action]} used); not run"
        self.seq += 1
        payload = {"episode": self.spec.episode, "seq": self.seq, "task_id": self.spec.task_id,
                   "task_version": self.spec.task_version, "code": code}
        started = time.time()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.pool, lambda: self.client.call_with_retry("POST", f"/tool/{action}", payload,
                                                           retries=1, delay_s=self.retry_delay_s))
        event = {"action": action, "seq": self.seq, "sha": digest[:16], "kind": result.kind,
                 "status": result.status, "seconds": round(time.time() - started, 2), "error": result.error}
        failure = grader_error(result.body or {}) if result.kind == "ok" else None
        if failure is not None:
            self.unavailable[action] += 1
            self.log("check", charged=False, grader_error=failure, **event)
            return False, f"{action} unavailable ({failure}); not counted against your budget"
        if result.kind == "ok":
            self.used[action] += 1
            passed = _passed(result.body or {}) if action == "correctness" else None
            if passed:
                self.passed.add(digest)
            header = f"{action} check {self.used[action]}/{self.budgets[action]} on MI355X:\n"
            text = header + summarize(action, result.body or {}, SUMMARY_CAP[action] - len(header))
            self.cache[(action, digest)] = text
            self.log("check", charged=True, passed=passed, **event)
            return True, text
        self.unavailable[action] += 1
        self.log("check", charged=False, **event)
        if result.kind == "unsupported":
            return False, f"{action} unavailable on the grader pool right now; not counted against your budget"
        if result.kind == "rejected":
            reason = str((result.body or {}).get("error", ""))[:300]
            return False, f"{action} check rejected (HTTP {result.status}: {reason}); not counted against your budget"
        return False, f"{action} check unavailable (infrastructure); not counted against your budget"

    def metrics(self) -> dict:
        return {"used": dict(self.used), "budgets": dict(self.budgets), "unavailable": dict(self.unavailable),
                "checks_sent": self.seq, "correctness_passed_hashes": len(self.passed)}


# ---------------------------------------------------------------- gateway forwarder

HOP_HEADERS = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "content-length", "host", "te",
               "upgrade", "proxy-authorization"}


class GatewayForwarder:
    """The policy boundary for model traffic: the only route out is the episode's /v1/messages."""

    def __init__(self, spec: EpisodeSpec, log: JsonlLog, denial_log: JsonlLog | None = None):
        self.spec = spec
        self.log = log
        self.denial_log = denial_log
        parts = urlsplit(spec.gateway_url)
        self.host, self.port = parts.hostname, parts.port or 80
        self.prefix = parts.path.rstrip("/")
        self.allowed_prefix = f"/e/{spec.episode}/v1/messages"
        self.requests = 0
        self.denials = 0

    def deny(self, **fields) -> None:
        self.denials += 1
        self.log("gateway_denied", **fields)
        if self.denial_log is not None:
            self.denial_log("gateway_denied", episode=self.spec.episode, **fields)

    def allowed(self, method: str, path: str) -> bool:
        if method != "POST" or ".." in path or "%" in path or "//" in path:
            return False
        return path == self.allowed_prefix or path.startswith(self.allowed_prefix + "/")

    async def __call__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw, method, target, version, headers = await cc_http.read_head(reader)
            path = urlsplit(target).path
            if not self.allowed(method, path):
                self.deny(method=method, path=path[:200], reason="not allowlisted")
                writer.write(cc_http.json_response(403, {"type": "error", "error": {
                    "type": "permission_error", "message": "only the episode's /v1/messages is reachable"}}))
                await writer.drain()
                return
            body = await cc_http.read_body(reader, headers)
            self.requests += 1
            up_reader, up_writer = await asyncio.open_connection(self.host, self.port)
            kept = [(k, v) for k, v in headers if k.lower() not in HOP_HEADERS]
            head = [f"{method} {self.prefix}{target} HTTP/1.1", f"Host: {self.host}:{self.port}",
                    f"Content-Length: {len(body)}", "Connection: close"] + [f"{k}: {v}" for k, v in kept]
            up_writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body)
            await up_writer.drain()
            while True:
                chunk = await up_reader.read(65536)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
            up_writer.close()
        except (cc_http.HTTPError, ConnectionError, asyncio.IncompleteReadError, OSError) as exc:
            if isinstance(exc, cc_http.HTTPError):
                self.deny(reason="malformed request", error=str(exc)[:200])
            else:
                self.log("gateway_error", error=type(exc).__name__)
            try:
                writer.write(cc_http.json_response(502, {"type": "error", "error": {
                    "type": "api_error", "message": "gateway unreachable"}}))
                await writer.drain()
            except ConnectionError:
                pass
        finally:
            writer.close()


# ---------------------------------------------------------------- container helpers


class Docker:
    """Short-lived docker CLI calls, capped so their threads stay under the per-user process limit."""

    def __init__(self, concurrency: int):
        self.sem = asyncio.Semaphore(concurrency)
        self.env = {**os.environ, "GOMAXPROCS": "2"}

    async def run(self, *args: str, stdin: bytes | None = None, timeout: float = 120.0) -> tuple[int, bytes, bytes]:
        async with self.sem:
            proc = await asyncio.create_subprocess_exec(
                "docker", *args, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self.env)
            try:
                out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return 124, b"", b"docker call timed out"
            return proc.returncode, out, err


def claude_env(spec: EpisodeSpec) -> dict[str, str]:
    """Plan §2.4, names checked against the 2.1.311 binary."""
    env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:8080/e/{spec.episode}",
        "ANTHROPIC_MODEL": spec.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "bg-stub",
        "ANTHROPIC_SMALL_FAST_MODEL": "bg-stub",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(spec.max_output_tokens),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
        "CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY": "1",
        "CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS": "8000",
        "ENABLE_TOOL_SEARCH": "false",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_BUG_COMMAND": "1",
        "DISABLE_AUTO_COMPACT": "1",
        "DISABLE_COST_WARNINGS": "1",
        "BASH_DEFAULT_TIMEOUT_MS": "120000",
        "BASH_MAX_TIMEOUT_MS": "120000",
        "BASH_MAX_OUTPUT_LENGTH": "8000",
        "IS_SANDBOX": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "HOME": "/home/agent",
        "USER": "agent",
        "LOGNAME": "agent",
        "CC_EVAL_URL": "http://127.0.0.1:8081",
        "CC_EVAL_TOKEN_FILE": "/home/agent/.cc/eval_token",
    }
    if spec.effort:
        env["CLAUDE_CODE_EFFORT_LEVEL"] = spec.effort
    return env


RUN_SH = r"""#!/bin/bash
# Claude Code inside the rollout container; the tokens are read from tmpfs files, never argv.
set -u
{exports}
export ANTHROPIC_AUTH_TOKEN="$(cat /home/agent/.cc/gw_token)"
mkdir -p /tmp/cc
cd /workspace
timeout -s TERM -k 15 {wall} claude -p --model "$ANTHROPIC_MODEL" --output-format stream-json --verbose \
  --max-turns {max_turns} --dangerously-skip-permissions --tools "{tools}" --disallowedTools "{disallowed}" \
  --no-session-persistence --disable-slash-commands --strict-mcp-config \
  < /home/agent/.cc/prompt.md > /tmp/cc/stream.jsonl 2> /tmp/cc/stderr.txt
rc=$?
reason=exit; [ $rc -eq 124 ] && reason=wall_clock
unset ANTHROPIC_AUTH_TOKEN
python3 - "$rc" "$reason" <<'EOF'
import json, sys, urllib.request
token = open("/home/agent/.cc/eval_token").read().strip()
req = urllib.request.Request("http://127.0.0.1:8081/done", data=json.dumps({{"exit": int(sys.argv[1]), "reason": sys.argv[2]}}).encode(),
                             headers={{"content-type": "application/json", "authorization": "Bearer " + token}})
urllib.request.urlopen(req, timeout=30).read()
EOF
"""


def home_tar(spec: EpisodeSpec, eval_token: str, prompt: str) -> bytes:
    exports = "\n".join(f"export {k}={json.dumps(v)}" for k, v in claude_env(spec).items())
    files = {
        ".cc/gw_token": (spec.gateway_token + "\n", 0o600),
        ".cc/eval_token": (eval_token + "\n", 0o600),
        ".cc/prompt.md": (prompt, 0o644),
        ".cc/run.sh": (RUN_SH.format(exports=exports, wall=int(spec.wall_s), max_turns=int(spec.max_turns),
                                     tools=TOOLS, disallowed=DISALLOWED_TOOLS), 0o755),
        ".claude.json": (json.dumps({"autoCompactEnabled": False, "hasCompletedOnboarding": True,
                                     "bypassPermissionsModeAccepted": True, "autoUpdates": False}), 0o644),
        ".claude/settings.json": (json.dumps({"autoCompactEnabled": False, "includeCoAuthoredBy": False,
                                              "cleanupPeriodDays": 1}), 0o644),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for directory in (".cc", ".claude"):
            info = tarfile.TarInfo(directory)
            info.type, info.mode = tarfile.DIRTYPE, 0o700
            tar.addfile(info)
        for name, (text, mode) in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), mode, int(time.time())
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def stream_metrics(stream: str) -> dict:
    """Turns, tool calls by name and the final result event from Claude Code's stream-json."""
    tools: dict[str, int] = {}
    turns, result, parse_errors = 0, {}, 0
    for line in stream.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            parse_errors += 1
            continue
        if event.get("type") == "assistant":
            turns += 1
            for block in event.get("message", {}).get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tools[block.get("name", "?")] = tools.get(block.get("name", "?"), 0) + 1
        elif event.get("type") == "result":
            result = {k: event.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms", "stop_reason")}
    return {"assistant_messages": turns, "tool_calls": tools, "result": result, "stream_parse_errors": parse_errors}


def workspace_diff(original: Path, files: dict[str, str]) -> str:
    out = []
    names = sorted(set(files) | {p.relative_to(original).as_posix() for p in original.rglob("*") if p.is_file()})
    for name in names:
        before = (original / name).read_text(errors="replace") if (original / name).is_file() else ""
        after = files.get(name, "")
        if before != after:
            out.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), f"a/{name}", f"b/{name}"))
    text = "".join(out)
    return text if len(text) <= 2_000_000 else text[:2_000_000] + "\n[truncated]\n"


# ---------------------------------------------------------------- episodes


@dataclass
class Episode:
    spec: EpisodeSpec
    status: str = "starting"
    result: dict = field(default_factory=dict)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    container: str = ""


class Runner:
    def __init__(self, args):
        self.args = args
        self.docker = Docker(args.docker_concurrency)
        self.client = GraderClient(pool_size=args.grader_pool)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.grader_pool,
                                                          thread_name_prefix="grader")
        self.episodes: dict[str, Episode] = {}
        self.token = Path(args.token_file).read_text().strip() if args.token_file else None
        self.sock_root = Path(args.sock_root)
        self.log_root = Path(args.log_root)
        self.denial_log = JsonlLog(self.log_root / "denials.jsonl")
        self.sock_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.sock_root, 0o711)

    def running(self) -> int:
        return sum(1 for e in self.episodes.values() if not e.finished.is_set())

    async def api(self, request: cc_http.Request) -> bytes:
        if self.token is not None and not hmac.compare_digest(request.bearer(), self.token):
            return cc_http.json_response(401, {"error": "unauthorized"})
        parts = [p for p in request.path.split("/") if p]
        if parts == ["health"]:
            return cc_http.json_response(200, {"ok": True, "running": self.running(), "capacity": self.args.capacity,
                                               "episodes": len(self.episodes), "grader": self.client.stats.snapshot()})
        if parts == ["episodes"] and request.method == "POST":
            spec = EpisodeSpec.from_json(request.json())
            if not spec.kb and not self.args.image_nokb:
                return cc_http.json_response(400, {"error": "this runner has no /kb-off image (--image-nokb)"})
            if spec.episode in self.episodes:
                return cc_http.json_response(409, {"error": "episode exists"})
            if self.running() >= self.args.capacity:
                return cc_http.json_response(503, {"error": "at capacity", "running": self.running()})
            episode = Episode(spec)
            self.episodes[spec.episode] = episode
            episode.task = asyncio.create_task(self.run_episode(episode))
            return cc_http.json_response(202, {"ok": True, "episode": spec.episode})
        if len(parts) == 2 and parts[0] == "episodes":
            episode = self.episodes.get(parts[1])
            if episode is None:
                return cc_http.json_response(404, {"error": "unknown episode"})
            if request.method == "GET":
                wait = min(float(request.query.get("wait", 0) or 0), 600.0)
                if wait > 0 and not episode.finished.is_set():
                    try:
                        await asyncio.wait_for(episode.finished.wait(), wait)
                    except asyncio.TimeoutError:
                        pass
                body = {"episode": parts[1], "status": episode.status}
                if episode.finished.is_set():
                    body["result"] = episode.result
                    if request.query.get("forget") == "1":
                        self.episodes.pop(parts[1], None)
                return cc_http.json_response(200, body)
            if request.method == "DELETE":
                if episode.task and not episode.finished.is_set():
                    episode.task.cancel()
                return cc_http.json_response(200, {"ok": True})
        return cc_http.json_response(404, {"error": "unknown route"})

    async def run_episode(self, episode: Episode) -> None:
        spec = episode.spec
        name = f"cc_{safe_name(spec.episode)}"
        sock_dir = self.sock_root / safe_name(spec.episode)
        log_dir = self.log_root / safe_name(spec.episode)
        log_dir.mkdir(parents=True, exist_ok=True)
        log = JsonlLog(log_dir / "events.jsonl")
        proxy = EvalProxy(spec, self.client, log, self.pool, retry_delay_s=self.args.retry_delay_s,
                          denial_log=self.denial_log)
        forwarder = GatewayForwarder(spec, log, denial_log=self.denial_log)
        servers = []
        started = time.time()
        reason = "error"
        files: dict[str, str] = {}
        try:
            log("start", spec=spec.public(), node=os.uname().nodename)
            if sock_dir.exists():
                shutil.rmtree(sock_dir)
            sock_dir.mkdir()
            os.chmod(sock_dir, 0o755)
            servers.append(await asyncio.start_unix_server(cc_http.serve(proxy.handle), str(sock_dir / "eval.sock")))
            servers.append(await asyncio.start_unix_server(forwarder, str(sock_dir / "gateway.sock")))
            for sock in ("eval.sock", "gateway.sock"):
                os.chmod(sock_dir / sock, 0o666)
            image = self.args.image if spec.kb else self.args.image_nokb
            args = container_args(name, image, CONTAINER_UID, CONTAINER_UID, fwd_dir=str(sock_dir),
                                  sock_root=str(self.sock_root), labels=container_labels())
            await self.docker.run("rm", "-f", name)
            rc, out, err = await self.docker.run(*args[1:])
            if rc != 0:
                raise RuntimeError(f"docker run failed: {err.decode(errors='replace')[:300]}")
            episode.container = name
            if not spec.kb:
                rc, _, _ = await self.docker.run("exec", name, "test", "!", "-e", "/kb")
                if rc != 0:
                    raise RuntimeError("/kb is present in the /kb-off container")
            workspace = Path(spec.workspace_dir)
            prompt = spec.prompt or build_prompt(workspace, with_kb=spec.kb)
            for target, data in (("/workspace", workspace_tar(workspace, with_kb=spec.kb)),
                                 ("/home/agent", home_tar(spec, proxy.token, prompt))):
                rc, _, err = await self.docker.run("exec", "-i", name, "tar", "-x", "--no-same-owner", "-C", target,
                                                   stdin=data)
                if rc != 0:
                    raise RuntimeError(f"tar into {target} failed: {err.decode(errors='replace')[:300]}")
            await self.docker.run("exec", "-d", name, "python3", "/opt/cc/uds_relay.py")
            rc, _, err = await self.docker.run("exec", "-d", name, "bash", "/home/agent/.cc/run.sh")
            if rc != 0:
                raise RuntimeError(f"starting claude failed: {err.decode(errors='replace')[:300]}")
            episode.status = "running"
            log("running", setup_s=round(time.time() - started, 2))
            try:
                await asyncio.wait_for(proxy.done.wait(), spec.wall_s + self.args.wall_grace_s)
                reason = proxy.done_info.get("reason") or "exit"
            except asyncio.TimeoutError:
                reason = "wall_clock_backstop"
            files = await self.collect(name)
        except asyncio.CancelledError:
            reason = "cancelled"
            try:
                files = await asyncio.shield(self.collect(name)) if episode.container else {}
            except Exception:  # noqa: BLE001
                files = {}
        except Exception as exc:  # noqa: BLE001
            reason = "error"
            log("error", error=f"{type(exc).__name__}: {exc}"[:500])
        finally:
            for server in servers:
                server.close()
            if episode.container:
                await self.docker.run("rm", "-f", name)
            shutil.rmtree(sock_dir, ignore_errors=True)
        workspace_files = {k[len("workspace/"):]: v for k, v in files.items() if k.startswith("workspace/")}
        stream = files.get("tmp/cc/stream.jsonl", "")
        (log_dir / "stream.jsonl").write_text(stream)
        (log_dir / "claude.stderr.txt").write_text(files.get("tmp/cc/stderr.txt", ""))
        final_code = workspace_files.get("solution/kernel.hip")
        if final_code is not None:
            (log_dir / "kernel.hip").write_text(final_code)
        try:
            (log_dir / "workspace.diff").write_text(workspace_diff(Path(spec.workspace_dir), workspace_files))
        except OSError:
            pass
        metrics = {"reason": reason, "exit": proxy.done_info.get("exit"), "wall_s": round(time.time() - started, 1),
                   "checks": proxy.metrics(), "gateway_requests": forwarder.requests,
                   "gateway_denials": forwarder.denials, **stream_metrics(stream)}
        (log_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        log("finish", **{k: metrics[k] for k in ("reason", "exit", "wall_s", "gateway_requests", "gateway_denials")})
        log.close()
        ok = reason in ("exit", "wall_clock", "wall_clock_backstop") and final_code is not None
        episode.result = {"ok": ok, "final_code": final_code, "metrics": metrics, "log_dir": str(log_dir)}
        episode.status = "done" if ok else "failed"
        episode.finished.set()

    async def collect(self, name: str) -> dict[str, str]:
        rc, out, _ = await self.docker.run("exec", name, "tar", "-c", "-C", "/", "workspace", "tmp/cc", timeout=120)
        files = {}
        if not out:
            return files
        try:
            with tarfile.open(fileobj=io.BytesIO(out[:COLLECT_LIMIT])) as tar:
                for member in tar:
                    if member.isfile() and member.size <= 8 * 1024 * 1024:
                        files[member.name.lstrip("./")] = tar.extractfile(member).read().decode(errors="replace")
        except tarfile.TarError:
            pass
        return files


def build_prompt(workspace: Path, *, with_kb: bool = True) -> str:
    from cc_prompt import build_prompt as build  # noqa: PLC0415

    return build(workspace, with_kb=with_kb)


async def amain(args) -> None:
    runner = Runner(args)
    server = await asyncio.start_server(cc_http.serve(runner.api), args.host, args.port)
    logger.info("cc-runner on %s:%s image %s (no-/kb %s) capacity %s", args.host, args.port, args.image,
                args.image_nokb, args.capacity)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18500)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-nokb", default=None, help="Pinned image without /kb, for /kb-off episodes.")
    parser.add_argument("--capacity", type=int, default=int(os.environ.get("CC_CONTAINERS_PER_NODE", "32")))
    parser.add_argument("--token-file", default=os.environ.get("CC_RUNNER_TOKEN_FILE"))
    parser.add_argument("--sock-root", default="/tmp/cc-sock")
    parser.add_argument("--log-root", default="/tmp/cc-logs")
    parser.add_argument("--docker-concurrency", type=int, default=4)
    parser.add_argument("--grader-pool", type=int, default=16)
    parser.add_argument("--retry-delay-s", type=float, default=5.0)
    parser.add_argument("--wall-grace-s", type=float, default=90.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
