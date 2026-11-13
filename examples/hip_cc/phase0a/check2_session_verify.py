"""Phase 0a check 2: Miles' Anthropic session-verify agent, with its model calls through the cc-gateway.

``anthropic_session_verify_agent.run_agent`` talks to ``<session>/v1/messages`` and reads the
session snapshot. Here a session is opened on node 0's session server, an episode is registered
with the node's cc-gateway, and every ``/v1/messages`` POST of the agent is sent to
``<gateway>/e/<episode>/v1/messages`` with the episode token instead; the agent's health and
snapshot reads still go to the session server. The agent's own assertions are the check.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import sys
import time
from pathlib import Path

import httpx

from miles.utils.test_utils import anthropic_session_verify_agent as verify
from miles.utils.test_utils.session_verify_agent import fixed_template_append_roles

SERVED = os.environ.get("CC_SERVED_MODEL", "glm-5.3-flash")
TITO_MODEL = os.environ.get("CHECK2_TITO_MODEL", "glm53")


def main() -> int:
    out = Path(os.environ["CC_OUT"])
    session_server = os.environ["CC_SESSION_URLS"].split(",")[0].rstrip("/")
    gateway_url = Path(os.environ["CC_NODES_FILE"]).read_text().split()[1].rstrip("/")
    admin = Path(os.environ["CC_GATEWAY_ADMIN_TOKEN_FILE"]).read_text().strip()
    episode, token = f"check2-{secrets.token_hex(4)}", secrets.token_urlsafe(32)
    session_id = httpx.post(f"{session_server}/sessions", json={}, timeout=30).json()["session_id"]
    base_url = f"{session_server}/sessions/{session_id}"
    registration = {"episode": episode, "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "session_url": base_url, "model": SERVED, "context": int(os.environ["CC_CONTEXT"]),
                    "max_out": 16384, "stop_margin": 4096, "max_calls": 40, "wall_s": 1200.0}
    admin_headers = {"Authorization": f"Bearer {admin}"}
    response = httpx.post(f"{gateway_url}/episodes", json=registration, headers=admin_headers, timeout=30)
    response.raise_for_status()
    routed = []

    class GatewayClient(httpx.AsyncClient):
        async def post(self, url, *args, **kwargs):
            url = str(url)
            if url == f"{base_url}/v1/messages":
                url = f"{gateway_url}/e/{episode}/v1/messages"
                kwargs["headers"] = {**(kwargs.get("headers") or {}), "Authorization": f"Bearer {token}"}
                routed.append(time.time())
            return await super().post(url, *args, **kwargs)

    verify.httpx.AsyncClient = GatewayClient
    expectation = "required" if "system" in fixed_template_append_roles(TITO_MODEL) else "forbidden"
    metadata = {"tito_model": TITO_MODEL, "anthropic_model": SERVED,
                "anthropic_intermediate_system_expectation": expectation}
    result, error = None, None
    started = time.time()
    try:
        result = asyncio.run(verify.run_agent(base_url, None, {"max_tokens": 1024, "temperature": 0.0}, metadata))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {str(exc)[:1500]}"
    gateway = httpx.delete(f"{gateway_url}/episodes/{episode}", headers=admin_headers, timeout=30).json()
    snapshot = httpx.get(base_url, timeout=60).json()
    tree = (snapshot.get("metadata") or {}).get("tree") or {}
    roots = sum(1 for n in tree.get("nodes", []) if n.get("parent") is None)
    httpx.delete(base_url, timeout=30)
    report = {"episode": episode, "session_id": session_id, "expectation": expectation, "seconds": round(time.time() - started, 1),
              "requests_through_gateway": len(routed), "gateway": gateway, "roots": roots,
              "tito_session_mismatch": (snapshot.get("metadata") or {}).get("tito_session_mismatch"),
              "result": result, "error": error}
    report["pass"] = error is None and len(routed) >= verify._REQUEST_COUNT and roots == 1
    (out / "check2.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print("CHECK 2:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
