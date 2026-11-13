"""Minimal HTTP/1.1 over asyncio streams for the node-local cc-runner (stdlib only).

The runner runs on the host Python of a Miles node, next to the docker CLI, so it can't depend
on Starlette or aiohttp. One request per connection: every response carries
``Connection: close``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlsplit

MAX_HEAD = 64 * 1024
MAX_BODY = 64 * 1024 * 1024
REASONS = {200: "OK", 202: "Accepted", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden",
           404: "Not Found", 405: "Method Not Allowed", 409: "Conflict", 413: "Payload Too Large",
           429: "Too Many Requests", 500: "Internal Server Error", 502: "Bad Gateway", 503: "Service Unavailable"}


class HTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Request:
    method: str
    target: str
    version: str
    headers: list[tuple[str, str]]
    body: bytes = b""
    raw_head: bytes = field(default=b"", repr=False)

    def header(self, name: str, default: str = "") -> str:
        name = name.lower()
        return next((v for k, v in self.headers if k.lower() == name), default)

    @property
    def path(self) -> str:
        return urlsplit(self.target).path

    @property
    def query(self) -> dict[str, str]:
        return {k: v[-1] for k, v in parse_qs(urlsplit(self.target).query).items()}

    def json(self) -> dict:
        try:
            value = json.loads(self.body or b"{}")
        except ValueError as exc:
            raise HTTPError(400, "invalid JSON") from exc
        if not isinstance(value, dict):
            raise HTTPError(400, "JSON object expected")
        return value

    def bearer(self) -> str:
        return self.header("authorization").removeprefix("Bearer ").strip() or self.header("x-api-key")


async def read_head(reader: asyncio.StreamReader) -> tuple[bytes, str, str, str, list[tuple[str, str]]]:
    try:
        raw = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError as exc:
        raise HTTPError(413, "request head too large") from exc
    except asyncio.IncompleteReadError as exc:
        raise ConnectionResetError("client closed before a request") from exc
    if len(raw) > MAX_HEAD:
        raise HTTPError(413, "request head too large")
    lines = raw.decode("latin-1").split("\r\n")
    try:
        method, target, version = lines[0].split(" ", 2)
    except ValueError as exc:
        raise HTTPError(400, "bad request line") from exc
    headers = []
    for line in lines[1:]:
        if line:
            name, _, value = line.partition(":")
            headers.append((name.strip(), value.strip()))
    return raw, method, target, version, headers


async def read_body(reader: asyncio.StreamReader, headers: list[tuple[str, str]]) -> bytes:
    lookup = {k.lower(): v for k, v in headers}
    if "chunked" in lookup.get("transfer-encoding", "").lower():
        chunks, total = [], 0
        while True:
            size_line = await reader.readuntil(b"\r\n")
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while (await reader.readuntil(b"\r\n")) != b"\r\n":
                    pass
                break
            total += size
            if total > MAX_BODY:
                raise HTTPError(413, "request body too large")
            chunks.append(await reader.readexactly(size))
            await reader.readexactly(2)
        return b"".join(chunks)
    length = int(lookup.get("content-length", "0") or 0)
    if length > MAX_BODY:
        raise HTTPError(413, "request body too large")
    return await reader.readexactly(length) if length else b""


async def read_request(reader: asyncio.StreamReader) -> Request:
    raw, method, target, version, headers = await read_head(reader)
    body = await read_body(reader, headers)
    return Request(method, target, version, headers, body, raw)


def response_bytes(status: int, body: bytes, content_type: str = "application/json",
                   extra: dict[str, str] | None = None) -> bytes:
    head = [f"HTTP/1.1 {status} {REASONS.get(status, 'Status')}", f"Content-Type: {content_type}",
            f"Content-Length: {len(body)}", "Connection: close"]
    head += [f"{k}: {v}" for k, v in (extra or {}).items()]
    return ("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body


def json_response(status: int, value) -> bytes:
    return response_bytes(status, json.dumps(value).encode())


def serve(handler):
    """Wrap ``async handler(Request) -> bytes`` as an asyncio stream callback."""

    async def callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await read_request(reader)
            payload = await handler(request)
        except HTTPError as exc:
            payload = json_response(exc.status, {"error": str(exc)})
        except (ConnectionError, asyncio.IncompleteReadError):
            writer.close()
            return
        except Exception as exc:  # noqa: BLE001
            payload = json_response(500, {"error": f"{type(exc).__name__}"})
        try:
            writer.write(payload)
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()

    return callback
