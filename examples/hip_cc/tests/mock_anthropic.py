"""Scripted Anthropic Messages server for capturing and replaying Claude Code traffic.

Two roles:

* ``--role anthropic``: stands in for api.anthropic.com while ``claude -p`` runs, logging
  every raw request (that log is what the gateway tests replay).
* ``--role session``: stands in for the Miles session server behind cc_gateway. Every
  request is checked with Miles' own Anthropic adapter validation and rejected with 400
  on failure, the way the real session server would.

The served model follows a script picked by ``SCENARIO:<name>`` in the first user message;
the step is the number of assistant turns so far. Other models (Claude Code's background
calls) get a one-word reply.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

SERVED_MODEL = "glm-5.3-flash"
DEFAULT_SCENARIO = "text_only"
LOGGED_HEADERS = ("anthropic-version", "anthropic-beta", "user-agent", "x-app", "content-type")


def _bash(command, description="run"):
    return ("Bash", {"command": command, "description": description})


def scenario_steps(name: str, cwd: str) -> list[list]:
    """Each step is the list of blocks of one assistant turn: ("text", s) or (tool, input)."""
    path = lambda f: f"{cwd.rstrip('/')}/{f}" if cwd else f
    scripts = {
        "bash_basic": [[_bash("ls -la", "list files")], [_bash("echo hello && pwd")], [("text", "Done: listed files.")]],
        "write_read_edit": [
            [("Write", {"file_path": path("hello.txt"), "content": "hello world\nsecond line\n"})],
            [("Read", {"file_path": path("hello.txt")})],
            [("Edit", {"file_path": path("hello.txt"), "old_string": "world", "new_string": "gateway"})],
            [_bash(f"cat {path('hello.txt')}")],
            [("text", "File written, read and edited.")],
        ],
        "errors": [
            [_bash("cat /nonexistent_file_for_mock_test", "fails on purpose")],
            [("Read", {"file_path": path("missing_file.txt")})],
            [_bash("exit 3", "nonzero exit")],
            [("text", "Errors handled.")],
        ],
        "multi_tool": [
            [_bash("echo alpha > a.txt && echo beta > b.txt"), ("Glob", {"pattern": "*.txt"})],
            [("Grep", {"pattern": "alpha", "path": cwd or "."})],
            [("text", "Found alpha in a.txt.")],
        ],
        "long": [[_bash(f"seq 1 {200 * (i + 1)} | tail -n 150", f"step {i}")] for i in range(8)]
        + [[("text", "Long run complete.")]],
        "text_only": [[("text", "Nothing to do; answering directly.")]],
        "hip_like": [
            [("Write", {"file_path": path("solution/kernel.hip"), "content": HIP_STUB})],
            [_bash("ls -la solution && wc -l solution/kernel.hip")],
            [("Read", {"file_path": path("solution/kernel.hip")})],
            [("text", "Kernel written to solution/kernel.hip.")],
        ],
        "hip_episode": [
            [_bash("head -20 README.md && aka-eval status", "read the task")],
            [("Read", {"file_path": "/kb/INDEX.md"})],
            [("Write", {"file_path": path("solution/kernel.hip"), "content": _kernel("relu6_wrong.hip")})],
            [_bash("aka-eval compile", "build")],
            [_bash("aka-eval correctness", "check")],
            [("Write", {"file_path": path("solution/kernel.hip"), "content": _kernel("relu6_good.hip")})],
            [_bash("aka-eval compile && aka-eval correctness", "build and check")],
            [_bash("aka-eval correctness", "same code again")],
            [_bash("aka-eval performance", "time it")],
            [_bash("aka-eval counters; aka-eval status", "counters")],
            [_bash(RED_TEAM, "probe the sandbox")],
            [("text", "ReLU6 kernel done.")],
        ],
    }
    return scripts.get(name, scripts["text_only"])


KERNELS = Path(__file__).resolve().parent / "kernels"
RED_TEAM = ("python3 -c \"import socket\nfor h, p in (('8.8.8.8', 53), ('10.245.147.96', 18310), ('1.1.1.1', 443)):\n"
            "    try:\n        socket.create_connection((h, p), 3); print('REACHED', h, p)\n"
            "    except OSError as e:\n        print('blocked', h, p, type(e).__name__)\"; "
            "curl -s -m 3 -X POST http://127.0.0.1:8080/episodes -d '{}'; echo; "
            "curl -s -m 3 http://127.0.0.1:8080/e/other-episode/v1/messages -d '{}'; echo; "
            "curl -s -m 3 http://127.0.0.1:8080/sessions/s1/samples; echo; "
            "curl -s -m 3 -X POST http://127.0.0.1:8080/flush_cache; echo; "
            "curl -s -m 3 -X POST http://127.0.0.1:8080/update_weights_from_disk -d '{}'; echo; "
            "curl -s -m 3 --path-as-is -X POST \"$ANTHROPIC_BASE_URL/v1/messages/../../../sessions/s1/samples\" -d '{}'; echo; "
            "curl -s -m 3 -X POST http://127.0.0.1:8081/grade -d '{}'; echo; "
            "curl -s -m 3 -H \"Authorization: Bearer $(cat $CC_EVAL_TOKEN_FILE)\" -X POST http://127.0.0.1:8081/tasks/upload -d '{}'; echo; "
            "curl -s -m 3 -H \"Authorization: Bearer $(cat $CC_EVAL_TOKEN_FILE)\" http://127.0.0.1:8081/health; echo; "
            "ls /shared_inference /home/yuankach 2>&1 | head -3; ls -la /fwd; whoami; id; "
            "[ -e /kb ]; echo kb_exists_rc=$?; grep -c /kb ~/.cc/prompt.md README.md")


def _kernel(name: str) -> str:
    return (KERNELS / name).read_text()


HIP_STUB = """#include <hip/hip_runtime.h>
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
"""


def _texts(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            out.append(block.get("text", ""))
        elif block.get("type") == "tool_result":
            out.append(_texts(block.get("content")))
    return "\n".join(out)


def _system_text(body: dict) -> str:
    system = body.get("system")
    return system if isinstance(system, str) else "\n".join(b.get("text", "") for b in system or [])


def plan_reply(body: dict, thinking: bool) -> list[dict]:
    messages = body.get("messages", [])
    first_user = next((m for m in messages if m.get("role") == "user"), {})
    found = re.search(r"SCENARIO:(\w+)", _texts(first_user.get("content")))
    scenario = found.group(1) if found else DEFAULT_SCENARIO
    cwd_match = re.search(r"[Ww]orking directory: (\S+)", _system_text(body))
    steps = scenario_steps(scenario, cwd_match.group(1) if cwd_match else "")
    step = sum(1 for m in messages if m.get("role") == "assistant")
    blocks = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": f"Scenario {scenario}, step {step}.",
                       "signature": "mock-signature"})
    if step >= len(steps):
        return blocks + [{"type": "text", "text": "All steps done."}]
    for kind, payload in steps[step]:
        if kind == "text":
            blocks.append({"type": "text", "text": payload})
        else:
            blocks.append({"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": kind,
                           "input": payload})
    return blocks


def _approx_tokens(body: dict) -> int:
    return max(1, len(json.dumps(body)) // 4)


def build_message(body: dict, thinking: bool) -> dict:
    model = body.get("model", "")
    if model != SERVED_MODEL:
        blocks = [{"type": "text", "text": "ok"}]
    elif not body.get("tools"):
        blocks = [{"type": "text", "text": "Mock summary."}]
    else:
        blocks = plan_reply(body, thinking)
    stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
    out_tokens = max(1, len(json.dumps(blocks)) // 4)
    return {"id": f"msg_mock_{uuid.uuid4().hex[:20]}", "type": "message", "role": "assistant", "model": model,
            "content": blocks, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": _approx_tokens(body), "output_tokens": out_tokens}}


def sse_body(message: dict) -> bytes:
    events = [("message_start", {"type": "message_start",
                                 "message": {**message, "content": [], "stop_reason": None,
                                             "usage": {**message["usage"], "output_tokens": 1}}})]
    for index, block in enumerate(message["content"]):
        kind = block["type"]
        if kind == "text":
            start, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
        elif kind == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": block["thinking"]},
                      {"type": "signature_delta", "signature": block.get("signature", "")}]
        else:
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
        events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": start}))
        events += [("content_block_delta", {"type": "content_block_delta", "index": index, "delta": d}) for d in deltas]
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                                     "usage": {"output_tokens": message["usage"]["output_tokens"]}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()


def _import_adapter_without_gpu() -> None:
    """sglang's anthropic ``utils`` pulls in GPU kernels at import time; validation only needs
    the request model in ``protocol``, so on a CPU node ``utils`` is replaced by an empty module."""
    if "miles.rollout.session.anthropic_adapter" in sys.modules:
        return
    try:
        import sglang.srt.entrypoints.anthropic.utils  # noqa: F401
    except Exception:
        import types

        sys.modules["sglang.srt.entrypoints.anthropic.utils"] = types.ModuleType("sglang_anthropic_utils_stub")
    import miles.rollout.session.anthropic_adapter as adapter

    assert adapter.AnthropicMessagesRequest is not None, "sglang anthropic protocol is not importable"


def miles_validate(raw: bytes) -> str | None:
    """None if Miles' Anthropic adapter accepts the request, else its error text."""
    _import_adapter_without_gpu()
    from miles.rollout.session.anthropic_adapter import _parse_anthropic_request, _validate_anthropic_features

    try:
        _validate_anthropic_features(_parse_anthropic_request(raw))
    except ValueError as exc:
        return str(exc)[:2000]
    return None


class MockServer:
    def __init__(self, role: str, log_path: str | None, thinking: bool, validate: bool | None = None):
        self.role = role
        self.thinking = thinking
        self.validate = (role == "session") if validate is None else validate
        self.log = open(log_path, "a", buffering=1) if log_path else None
        self.received: list[dict] = []
        self.fail_next: list[int] = []

    def _record(self, request: Request, raw: bytes, extra: dict) -> None:
        record = {"t": time.time(), "method": request.method, "path": request.url.path,
                  "query": str(request.url.query),
                  "headers": {k: v for k, v in request.headers.items() if k in LOGGED_HEADERS}, **extra}
        try:
            record["body"] = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            record["body_raw"] = raw.decode(errors="replace")
        self.received.append(record)
        if self.log:
            self.log.write(json.dumps(record) + "\n")

    async def messages(self, request: Request) -> Response:
        raw = await request.body()
        verdict = miles_validate(raw) if self.validate or self.role == "anthropic" else None
        self._record(request, raw, {"miles_verdict": verdict or "ok"})
        if self.fail_next:
            status = self.fail_next.pop(0)
            return JSONResponse({"type": "error", "error": {"type": "api_error", "message": "injected"}},
                                status_code=status)
        if self.role == "session" and verdict:
            return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": verdict}},
                                status_code=400)
        body = json.loads(raw)
        message = build_message(body, self.thinking)
        if body.get("stream"):
            return Response(sse_body(message), media_type="text/event-stream")
        return JSONResponse(message)

    async def count_tokens(self, request: Request) -> Response:
        raw = await request.body()
        self._record(request, raw, {})
        return JSONResponse({"input_tokens": _approx_tokens(json.loads(raw or b"{}"))})

    async def other(self, request: Request) -> Response:
        self._record(request, await request.body(), {"unhandled": True})
        return JSONResponse({"type": "error", "error": {"type": "not_found_error", "message": "mock"}},
                            status_code=404)

    def app(self) -> Starlette:
        return Starlette(routes=[
            Route("/v1/messages/count_tokens", self.count_tokens, methods=["POST"]),
            Route("/v1/messages", self.messages, methods=["POST"]),
            Route("/sessions/{sid}/v1/messages", self.messages, methods=["POST"]),
            Route("/{rest:path}", self.other, methods=["GET", "POST", "HEAD", "PUT", "DELETE"]),
        ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["anthropic", "session"], default="anthropic")
    parser.add_argument("--port", type=int, default=18500)
    parser.add_argument("--log", default=None)
    parser.add_argument("--thinking", action="store_true", help="Reply with thinking blocks.")
    parser.add_argument("--default-scenario", default="text_only", help="Script when the prompt names none.")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    import uvicorn

    global DEFAULT_SCENARIO
    DEFAULT_SCENARIO = args.default_scenario

    sys.stderr.write(f"mock {args.role} on {args.port}\n")
    uvicorn.run(MockServer(args.role, args.log, args.thinking).app(), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
