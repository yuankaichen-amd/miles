"""cc-gateway: Claude Code (Anthropic Messages) -> Miles session server v2 (plan §2.1, §2.4).

Every Claude Code container gets ANTHROPIC_BASE_URL=<gateway>/e/<episode> and a
per-episode bearer token. The gateway

* authenticates the episode token (only its sha256 is stored),
* normalizes each request identically (drops thinking, output_config, betas,
  context_management, metadata and every cache_control; turns
  ``is_error: true`` tool results into plain results starting with "Error:";
  rejects images),
* routes background-model calls (any model other than the served one) to an
  unrecorded stub,
* enforces the per-episode budgets: model calls, the per-call output clamp
  ``min(max_out, W - input - 256)``, the stop rule (input above W - margin ends
  the episode with a synthetic, unrecorded end_turn) and the wall clock,
* forwards to ``<session_url>/v1/messages`` and logs every request and reply.

Admin routes (``/episodes``) need the admin token from ``--admin-token-file``;
containers only ever see ``/e/<episode>/...`` through the forwarder.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = logging.getLogger("cc_gateway")

DROPPED_TOP_LEVEL = ("thinking", "output_config", "betas", "context_management", "metadata", "effort")
STOP_REASONS = ("context_limit", "max_calls", "wall_clock")
# Top-level user media ends the request; the same media inside tool output becomes a text placeholder.
REJECTED_BLOCKS = ("image", "document")
TEXT_ONLY_BLOCKS = ("tool_reference", "search_result")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class TokenCounter:
    """Prompt-size estimate for the stop rule and the clamp: exact with a tokenizer, else chars/3."""

    def __init__(self, tokenizer_path: str | None):
        self.tokenizer = None
        if tokenizer_path:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return (len(text) + 2) // 3


def _strip_cache_control(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [_strip_cache_control(v) for v in obj]
    return obj


class RequestRejected(ValueError):
    pass


def normalize_request(body: dict) -> tuple[dict, dict]:
    """The canonical request sent upstream, and counters of what was changed."""
    stats = {"dropped_fields": [], "is_error_results": 0, "dropped_blocks": 0, "replaced_blocks": 0}
    out = {k: v for k, v in body.items() if k not in DROPPED_TOP_LEVEL}
    stats["dropped_fields"] = sorted(k for k in body if k in DROPPED_TOP_LEVEL)
    out = _strip_cache_control(out)
    if isinstance(out.get("system"), list):
        # The billing header carries a per-build hash; it is client metadata, not prompt.
        out["system"] = [b for b in out["system"] if b.get("type") == "text"
                         and not b.get("text", "").startswith("x-anthropic-billing-header:")]
    messages = []
    for message in out.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            blocks = []
            for block in content:
                kind = block.get("type")
                if kind in REJECTED_BLOCKS:
                    raise RequestRejected(f"{kind} content is not supported")
                if kind in TEXT_ONLY_BLOCKS:
                    block = {"type": "text", "text": _placeholder(block)}
                    stats["replaced_blocks"] += 1
                if kind == "tool_result" and isinstance(block.get("content"), list):
                    block = _text_only_tool_result(block, stats)
                if kind == "redacted_thinking":
                    stats["dropped_blocks"] += 1
                    continue
                if kind == "thinking":
                    block = {k: v for k, v in block.items() if k != "signature"}
                    if message.get("role") != "assistant":
                        stats["dropped_blocks"] += 1
                        continue
                if kind == "tool_result" and block.get("is_error"):
                    block = _plain_error_result(block)
                    stats["is_error_results"] += 1
                elif kind == "tool_result" and "is_error" in block:
                    block = {k: v for k, v in block.items() if k != "is_error"}
                blocks.append(block)
            message = {**message, "content": blocks}
        messages.append(message)
    out["messages"] = messages
    if "tools" in out:
        out["tools"] = [t for t in out["tools"] if "input_schema" in t]
    return out, stats


def _placeholder(block: dict) -> str:
    if block.get("type") == "tool_reference":
        return f"Tool loaded: {block.get('tool_name', '')}"
    if block.get("type") == "search_result":
        texts = [b.get("text", "") for b in block.get("content") or [] if isinstance(b, dict)]
        return "\n".join([block.get("title", ""), block.get("source", ""), *texts]).strip()
    return f"[{block.get('type')} content omitted: not supported by this endpoint]"


def _text_only_tool_result(block: dict, stats: dict) -> dict:
    """Tool output (e.g. Read on a PNG) may carry media; the model only sees text placeholders."""
    parts = []
    for part in block["content"]:
        if isinstance(part, dict) and part.get("type") != "text":
            part = {"type": "text", "text": _placeholder(part)}
            stats["replaced_blocks"] += 1
        parts.append(part)
    return {**block, "content": parts}


def _plain_error_result(block: dict) -> dict:
    block = {k: v for k, v in block.items() if k != "is_error"}
    content = block.get("content")
    if content is None or isinstance(content, str):
        text = content or ""
        block["content"] = text if text.startswith("Error:") else f"Error: {text}"
    else:
        content = list(content)
        for index, part in enumerate(content):
            if part.get("type") == "text":
                text = part.get("text", "")
                content[index] = {**part, "text": text if text.startswith("Error:") else f"Error: {text}"}
                break
        else:
            content.insert(0, {"type": "text", "text": "Error:"})
        block["content"] = content
    return block


def request_text(body: dict) -> str:
    """Everything that becomes prompt tokens, flattened (used only for size estimates)."""
    parts = []
    system = body.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts += [b.get("text", "") for b in system]
    for tool in body.get("tools") or []:
        parts.append(json.dumps(tool, separators=(",", ":")))
    for message in body.get("messages", []):
        parts.append(_message_text(message))
    return "\n".join(parts)


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "thinking":
            parts.append(block.get("thinking", ""))
        elif kind == "tool_use":
            parts.append(json.dumps(block.get("input", {}), separators=(",", ":")) + block.get("name", ""))
        elif kind == "tool_result":
            inner = block.get("content")
            parts.append(inner if isinstance(inner, str) else
                         "\n".join(b.get("text", "") for b in inner or [] if isinstance(b, dict)))
    return "\n".join(parts)


@dataclass
class Episode:
    episode: str
    token_sha256: str
    session_url: str
    model: str
    context: int = 131072
    max_out: int = 16384
    stop_margin: int = 4096
    max_calls: int = 40
    wall_s: float = 1200.0
    created: float = field(default_factory=time.time)
    calls: int = 0
    stub_calls: int = 0
    rejected: int = 0
    upstream_errors: dict = field(default_factory=dict)
    stop_reason: str | None = None
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_message_count: int = 0
    last_messages_sha: str | None = None
    last_prompt_tokens_est: int = 0
    max_input_tokens: int = 0
    output_tokens: int = 0
    tool_uses: int = 0
    stop_reasons: dict = field(default_factory=dict)
    first_messages_sha: str | None = None
    roots: int = 0
    log: Any = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def summary(self) -> dict:
        return {
            "episode": self.episode, "calls": self.calls, "stub_calls": self.stub_calls, "rejected": self.rejected,
            "upstream_errors": self.upstream_errors, "stop_reason": self.stop_reason,
            "max_input_tokens": self.max_input_tokens, "output_tokens": self.output_tokens,
            "tool_uses": self.tool_uses, "stop_reasons": self.stop_reasons, "roots_est": self.roots,
            "wall_s": round(time.time() - self.created, 1),
        }


class Gateway:
    def __init__(self, args):
        self.args = args
        self.admin_token = Path(args.admin_token_file).read_text().strip() if args.admin_token_file else None
        self.episodes: dict[str, Episode] = {}
        self.counter = TokenCounter(args.tokenizer)
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=args.upstream_read_timeout, write=60.0, pool=60.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        )

    # ---------- admin ----------
    def _admin_ok(self, request: Request) -> bool:
        if self.admin_token is None:
            return True
        supplied = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        return hmac.compare_digest(supplied, self.admin_token)

    async def register(self, request: Request) -> Response:
        if not self._admin_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        token = body.pop("token", None)
        token_sha = body.pop("token_sha256", None) or (sha256(token) if token else None)
        if not token_sha or not body.get("episode") or not body.get("session_url"):
            return JSONResponse({"error": "episode, session_url and token(_sha256) are required"}, status_code=400)
        known = {f for f in Episode.__dataclass_fields__ if f not in ("token_sha256",)}
        params = {k: v for k, v in body.items() if k in known}
        params.setdefault("model", self.args.served_model)
        episode = Episode(token_sha256=token_sha, **params)
        episode.log = (self.log_dir / f"{_safe(episode.episode)}.jsonl").open("a", buffering=1)
        self.episodes[episode.episode] = episode
        self._log(episode, {"event": "register", "params": {k: v for k, v in params.items() if k != "session_url"}})
        return JSONResponse({"ok": True, "episode": episode.episode})

    async def finish(self, request: Request) -> Response:
        if not self._admin_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        episode = self.episodes.pop(request.path_params["episode"], None)
        if episode is None:
            return JSONResponse({"error": "unknown episode"}, status_code=404)
        summary = episode.summary()
        self._log(episode, {"event": "finish", "summary": summary})
        episode.log.close()
        return JSONResponse(summary)

    async def status(self, request: Request) -> Response:
        if not self._admin_ok(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        episode = self.episodes.get(request.path_params["episode"])
        if episode is None:
            return JSONResponse({"error": "unknown episode"}, status_code=404)
        return JSONResponse(episode.summary())

    async def healthz(self, request: Request) -> Response:
        return JSONResponse({"ok": True, "episodes": len(self.episodes)})

    # ---------- episode traffic ----------
    def _auth(self, request: Request) -> Episode | None:
        episode = self.episodes.get(request.path_params["episode"])
        if episode is None:
            return None
        supplied = request.headers.get("x-api-key") or \
            request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not supplied or not hmac.compare_digest(sha256(supplied), episode.token_sha256):
            return None
        return episode

    async def count_tokens(self, request: Request) -> Response:
        episode = self._auth(request)
        if episode is None:
            return _anthropic_error(401, "authentication_error", "invalid episode token")
        body = await request.json()
        normalized, _ = normalize_request(body)
        return JSONResponse({"input_tokens": self.counter.count(request_text(normalized))})

    async def messages(self, request: Request) -> Response:
        episode = self._auth(request)
        if episode is None:
            return _anthropic_error(401, "authentication_error", "invalid episode token")
        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return _anthropic_error(400, "invalid_request_error", "invalid JSON")
        stream = bool(body.get("stream"))
        if body.get("model") != episode.model:
            episode.stub_calls += 1
            self._log(episode, {"event": "stub", "model": body.get("model"), "request_sha": sha256(raw.decode())})
            return _synthetic(body.get("model") or "bg-stub", "", stream, recorded=False)
        async with episode.lock:
            return await self._model_call(episode, body, raw, stream)

    async def _model_call(self, episode: Episode, body: dict, raw: bytes, stream: bool) -> Response:
        if episode.stop_reason:
            return self._stop(episode, episode.stop_reason, body, stream)
        if time.time() - episode.created > episode.wall_s:
            return self._stop(episode, "wall_clock", body, stream)
        if episode.calls >= episode.max_calls:
            return self._stop(episode, "max_calls", body, stream)
        try:
            normalized, norm_stats = normalize_request(body)
        except RequestRejected as exc:
            episode.rejected += 1
            self._log(episode, {"event": "rejected", "reason": str(exc)})
            return _anthropic_error(400, "invalid_request_error", str(exc))
        input_est = self._input_estimate(episode, normalized)
        window = episode.context
        if input_est > window - episode.stop_margin:
            return self._stop(episode, "context_limit", body, stream, input_est=input_est)
        requested = int(normalized.get("max_tokens") or episode.max_out)
        normalized["max_tokens"] = max(1, min(requested, episode.max_out, window - input_est - 256))

        messages_sha = sha256(json.dumps(normalized.get("messages", [])[:1], sort_keys=True))
        if episode.first_messages_sha != messages_sha:
            episode.roots += 1
            episode.first_messages_sha = messages_sha
        episode.calls += 1
        call_index = episode.calls
        log_request = normalized if (self.args.log_full or call_index == 1) else {
            **{k: v for k, v in normalized.items() if k != "messages"},
            "messages_delta": normalized.get("messages", [])[episode.last_message_count:],
            "messages_total": len(normalized.get("messages", [])),
            "messages_sha256": sha256(json.dumps(normalized.get("messages", []), sort_keys=True)),
        }
        if call_index > 1 and not self.args.log_full:
            log_request.pop("system", None)
            log_request.pop("tools", None)
        self._log(episode, {"event": "request", "call": call_index, "normalize": norm_stats,
                            "input_est": input_est, "max_tokens": normalized["max_tokens"], "request": log_request})
        started = time.time()
        try:
            upstream = await self.client.post(
                f"{episode.session_url.rstrip('/')}/v1/messages",
                content=json.dumps(normalized).encode(),
                headers={"content-type": "application/json"},
            )
        except httpx.HTTPError as exc:
            episode.upstream_errors["transport"] = episode.upstream_errors.get("transport", 0) + 1
            self._log(episode, {"event": "upstream_error", "call": call_index, "error": repr(exc)})
            episode.calls -= 1
            return _anthropic_error(529, "overloaded_error", "upstream unavailable")
        elapsed = time.time() - started
        content = upstream.content
        if upstream.status_code != 200:
            key = str(upstream.status_code)
            episode.upstream_errors[key] = episode.upstream_errors.get(key, 0) + 1
            if upstream.status_code >= 500 or upstream.status_code == 429:
                episode.calls -= 1
            self._log(episode, {"event": "upstream_status", "call": call_index, "status": upstream.status_code,
                                "body": content[:4000].decode(errors="replace")})
            return Response(content=content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type", "application/json"))
        summary = _response_summary(content, upstream.headers.get("content-type", ""))
        episode.last_input_tokens = summary.get("input_tokens") or input_est
        episode.last_output_tokens = summary.get("output_tokens") or 0
        episode.last_message_count = len(normalized.get("messages", []))
        episode.last_messages_sha = _messages_sha(normalized.get("messages", []))
        episode.last_prompt_tokens_est = input_est
        episode.max_input_tokens = max(episode.max_input_tokens, episode.last_input_tokens)
        episode.output_tokens += episode.last_output_tokens
        episode.tool_uses += summary.get("tool_uses", 0)
        reason = summary.get("stop_reason") or "unknown"
        episode.stop_reasons[reason] = episode.stop_reasons.get(reason, 0) + 1
        self._log(episode, {"event": "response", "call": call_index, "elapsed_s": round(elapsed, 3),
                            "summary": summary, "body": content.decode(errors="replace")})
        media_type = "text/event-stream" if stream else "application/json"
        if not stream and upstream.headers.get("content-type", "").startswith("text/event-stream"):
            media_type = "text/event-stream"
        return Response(content=content, status_code=200, media_type=media_type,
                        headers={"cache-control": "no-cache"} if stream else None)

    def _input_estimate(self, episode: Episode, normalized: dict) -> int:
        messages = normalized.get("messages", [])
        count = episode.last_message_count
        extends_previous = (episode.calls and count and len(messages) > count
                            and _messages_sha(messages[:count]) == episode.last_messages_sha)
        if extends_previous:
            # Previous prompt + its completion are exact (usage); only the appended turns are estimated.
            delta = messages[count:]
            if delta[0].get("role") == "assistant":
                delta = delta[1:]
            delta_text = "\n".join(_message_text(m) for m in delta)
            known = episode.last_input_tokens + episode.last_output_tokens
            return known + self.counter.count(delta_text) + 16 * len(delta)
        return self.counter.count(request_text(normalized))

    def _stop(self, episode: Episode, reason: str, body: dict, stream: bool, input_est: int | None = None):
        if episode.stop_reason is None:
            episode.stop_reason = reason
        self._log(episode, {"event": "stop", "reason": reason, "input_est": input_est})
        return _synthetic(body.get("model") or episode.model, "", stream, recorded=False)

    def _log(self, episode: Episode, record: dict) -> None:
        record = {"t": round(time.time(), 3), **record}
        if episode.log is not None:
            episode.log.write(json.dumps(record, ensure_ascii=False) + "\n")


def _messages_sha(messages: list) -> str:
    return sha256(json.dumps(messages, sort_keys=True, separators=(",", ":")))


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:120]


def _anthropic_error(status: int, kind: str, message: str) -> Response:
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


def _synthetic(model: str, text: str, stream: bool, recorded: bool) -> Response:
    """An end_turn reply that never reaches the session server (so it is never trained on)."""
    message_id = f"msg_cc_gateway_{uuid.uuid4().hex[:20]}"
    usage = {"input_tokens": 0, "output_tokens": 0}
    message = {"id": message_id, "type": "message", "role": "assistant", "model": model,
               "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
               "stop_sequence": None, "usage": usage}
    headers = {"x-cc-gateway-synthetic": "1"}
    if not stream:
        return JSONResponse(message, headers=headers)
    events = [
        ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 0}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    payload = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return Response(content=payload.encode(), media_type="text/event-stream",
                    headers={**headers, "cache-control": "no-cache"})


def _response_summary(content: bytes, content_type: str) -> dict:
    """Usage, stop reason and tool-use count from a JSON or (fake) SSE Messages reply."""
    summary = {"tool_uses": 0}
    try:
        if content_type.startswith("text/event-stream") or content.startswith(b"event:"):
            for chunk in content.decode().split("\n\n"):
                data = next((line[5:].strip() for line in chunk.splitlines() if line.startswith("data:")), None)
                if not data:
                    continue
                event = json.loads(data)
                kind = event.get("type")
                if kind == "message_start":
                    usage = event["message"].get("usage") or {}
                    summary["input_tokens"] = usage.get("input_tokens")
                elif kind == "content_block_start" and event["content_block"].get("type") == "tool_use":
                    summary["tool_uses"] += 1
                elif kind == "message_delta":
                    summary["stop_reason"] = event.get("delta", {}).get("stop_reason")
                    usage = event.get("usage") or {}
                    summary["output_tokens"] = usage.get("output_tokens")
                    if usage.get("input_tokens"):
                        summary["input_tokens"] = usage["input_tokens"]
        else:
            message = json.loads(content)
            usage = message.get("usage") or {}
            summary.update(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                           stop_reason=message.get("stop_reason"),
                           tool_uses=sum(1 for b in message.get("content", []) if b.get("type") == "tool_use"))
    except (ValueError, KeyError, AttributeError) as exc:
        summary["parse_error"] = repr(exc)
    return summary


def build_app(args) -> Starlette:
    gateway = Gateway(args)
    app = Starlette(routes=[
        Route("/healthz", gateway.healthz, methods=["GET"]),
        Route("/episodes", gateway.register, methods=["POST"]),
        Route("/episodes/{episode}", gateway.status, methods=["GET"]),
        Route("/episodes/{episode}", gateway.finish, methods=["DELETE"]),
        Route("/e/{episode}/v1/messages/count_tokens", gateway.count_tokens, methods=["POST"]),
        Route("/e/{episode}/v1/messages", gateway.messages, methods=["POST"]),
    ])
    app.state.gateway = gateway
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18400)
    parser.add_argument("--served-model", default="glm-5.3-flash")
    parser.add_argument("--admin-token-file", default=os.environ.get("CC_GATEWAY_ADMIN_TOKEN_FILE"))
    parser.add_argument("--tokenizer", default=None, help="HF path for exact prompt counts (else chars/3).")
    parser.add_argument("--log-dir", default="/tmp/cc_gateway_logs")
    parser.add_argument("--log-full", action="store_true", help="Log full requests, not deltas.")
    parser.add_argument("--upstream-read-timeout", type=float, default=1800.0)
    return parser.parse_args(argv)


def main():
    import uvicorn

    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
