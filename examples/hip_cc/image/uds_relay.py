#!/usr/bin/env python3
"""In-container TCP -> unix-socket relay (plan §4.2 forwarder, container side).

The rollout container runs with ``--network none``. Its only way out is the per-episode socket
directory the runner binds at /fwd: 127.0.0.1:8080 reaches the cc-gateway forwarder and
127.0.0.1:8081 the eval-proxy. Both host ends check every request; this relay only moves bytes.
"""

import asyncio
import sys

ROUTES = {8080: "/fwd/gateway.sock", 8081: "/fwd/eval.sock"}


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (OSError, asyncio.CancelledError):
        pass


def handler(path: str):
    async def handle(reader, writer):
        try:
            up_reader, up_writer = await asyncio.open_unix_connection(path)
        except OSError:
            writer.close()
            return
        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
        for w in (writer, up_writer):
            w.close()
    return handle


async def main() -> None:
    servers = [await asyncio.start_server(handler(path), "127.0.0.1", port) for port, path in ROUTES.items()]
    print("uds_relay ready", flush=True)
    await asyncio.gather(*(server.serve_forever() for server in servers))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
