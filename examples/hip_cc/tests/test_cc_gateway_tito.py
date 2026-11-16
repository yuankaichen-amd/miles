"""cc_gateway in front of the real session server v2 (GLM-5.3 TITO): reasoning cap, gateway retries of
cut-off replies and harness notes must leave one root, no TITO mismatch on any leaf, and every prompt's
incremental ``input_ids`` equal to the canonical render of its messages.

The SGLang backend is ``MockSGLangServer`` with its chat handler replaced by a scripted "perfect model": each
completion is exactly the GLM-5.3 template's render of the scripted assistant turn plus its stop token, and a
cut-off reply is bare reasoning stopped at ``length``. The client follows Claude Code: it echoes replies, answers
tool calls, and after a thinking-only ``max_tokens`` reply drops it and appends its nudge to the last user turn.

Needs the GLM-5.3-Flash tokenizer at ``GLM53_TOKENIZER_PATH`` (skipped otherwise) and sglang's Anthropic protocol.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import requests

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import cc_gateway  # noqa: E402

SERVED = "glm-5.3-flash"
ADMIN = "admin-test-token"
NUDGE = ("Output token limit hit. Resume directly \u2014 no apology, no recap of what you were doing. Pick up "
         "mid-thought if that is where the cut happened. Break remaining work into smaller pieces.")
TOOLS = [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {}}}
         for n in ("Bash", "Write", "Read")]

# One entry per backend request, in order: (kind, reasoning, tool name or text, tool input).
SCRIPT = [
    ("tool", "List the workspace first.", "Bash", {"command": "ls"}),
    ("cut", "The tile should be 64 by 64 because the output has 512 channels and", None, None),
    ("tool", "Write the naive kernel now.", "Write", {"file_path": "/workspace/solution/kernel.hip",
                                                      "content": "__global__ void k() {}\n"}),
    ("cut", "MFMA fragment layout: lane l holds rows", None, None),
    ("cut", "Continuing the layout: register r of lane l maps to column", None, None),
    ("cut", "So the C fragment for 16x16x4 f32 is", None, None),
    ("tool", "Compile it.", "Bash", {"command": "aka-eval compile"}),
    ("text", "It compiled.", "Done: kernel written and compiled.", None),
]


@pytest.fixture(scope="module")
def glm():
    path = os.environ.get("GLM53_TOKENIZER_PATH", "")
    try:
        from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizerType, get_tito_tokenizer
        from miles.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - offline CI without the model
        pytest.skip(f"GLM-5.3 tokenizer unavailable at {path!r}: {exc}")
    return SimpleNamespace(path=path, tito=get_tito_tokenizer(tokenizer, TITOTokenizerType.GLM53))


class PerfectGLM:
    """Replaces ``MockSGLangServer._compute_chat_completions_response``."""

    def __init__(self, tito, script):
        self.tito = tito
        self.script = list(script)
        self.prompts: list[dict] = []

    def __call__(self, payload: dict) -> dict:
        from miles.utils.chat_template_utils.tito_tokenizer import extract_template_args

        messages, prompt_ids = payload["messages"], list(payload["input_ids"])
        args = extract_template_args(payload)
        canonical = self.tito.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                  template_args=args)
        self.prompts.append({"canonical": canonical == prompt_ids, "messages": messages})
        kind, reasoning, payload_a, payload_b = self.script.pop(0)
        index = len(self.prompts)
        tokenizer = self.tito.tokenizer
        if kind == "cut":
            message = {"role": "assistant", "content": "", "reasoning_content": reasoning}
            completion = tokenizer.encode(reasoning, add_special_tokens=False)
            finish, finish_info = "length", {"type": "length", "length": len(completion)}
        else:
            message = {"role": "assistant", "content": payload_a if kind == "text" else "",
                       "reasoning_content": reasoning}
            if kind == "tool":
                message["tool_calls"] = [{"id": f"call_{index}", "type": "function", "index": 0,
                                          "function": {"name": payload_a, "arguments": json.dumps(payload_b)}}]
            full = self.tito.apply_chat_template(messages + [message], add_generation_prompt=False,
                                                 tokenize=True, template_args=args)
            assert full[:len(canonical)] == canonical, "template render does not extend the generation prompt"
            stop = tokenizer.convert_tokens_to_ids("<|observation|>" if kind == "tool" else "<|user|>")
            completion = full[len(canonical):] + [stop]
            finish = "tool_calls" if kind == "tool" else "stop"
            finish_info = {"type": "stop", "matched": stop}
        return {
            "id": f"chatcmpl-{index}", "object": "chat.completion", "created": int(time.time()), "model": "mock",
            "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish,
                         "meta_info": {"output_token_logprobs": [[-0.1, t] for t in completion],
                                       "completion_tokens": len(completion), "finish_reason": finish_info}}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(completion),
                      "total_tokens": len(prompt_ids) + len(completion)},
        }


def _serve(glm, backend_url):
    from tests.fast.fixtures.session_fixtures import make_session_server_config

    from miles.rollout.session.server import SessionServer
    from miles.utils.http_utils import find_available_port
    from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer

    config = make_session_server_config(
        backend_url=backend_url, timeout=60, hf_checkpoint=glm.path, tito_model="glm53",
        use_session_server="v2", instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    port = find_available_port(31000)
    server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
    server.start()
    return server, f"http://127.0.0.1:{port}"


async def _claude_code(client, episode, token, *, stream_calls=(2,)):
    """Drives the gateway like Claude Code; returns what the client saw (it never sees a harness note)."""
    body = {"model": SERVED, "max_tokens": 32000, "tools": TOOLS, "system": "You write HIP kernels.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Task: write the kernel."}]}]}
    seen = []
    for call in range(1, 20):
        body["stream"] = call in stream_calls
        r = await client.post(f"/e/{episode}/v1/messages", json=body, headers={"x-api-key": token})
        assert r.status_code == 200, r.text
        message = _sse_message(r.text) if body["stream"] else r.json()
        seen.append(message)
        content = message["content"]
        if message["stop_reason"] == "tool_use":
            results = [{"type": "tool_result", "tool_use_id": b["id"], "content": f"output of {b['name']}"}
                       for b in content if b["type"] == "tool_use"]
            body["messages"] += [{"role": "assistant", "content": content}, {"role": "user", "content": results}]
        elif message["stop_reason"] == "max_tokens" and all(b["type"] == "thinking" for b in content):
            body["messages"][-1]["content"].append({"type": "text", "text": NUDGE})
        else:
            return seen, body
    raise AssertionError("client did not finish")


def _sse_message(text: str) -> dict:
    message, blocks = {}, []
    for chunk in text.split("\n\n"):
        data = next((line[5:].strip() for line in chunk.splitlines() if line.startswith("data:")), None)
        if not data:
            continue
        event = json.loads(data)
        kind = event["type"]
        if kind == "message_start":
            message = event["message"]
        elif kind == "content_block_start":
            blocks.append(dict(event["content_block"]))
        elif kind == "content_block_delta":
            delta, block = event["delta"], blocks[event["index"]]
            if delta["type"] == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "signature_delta":
                block["signature"] = delta["signature"]
            elif delta["type"] == "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
            elif delta["type"] == "input_json_delta":
                block["_json"] = block.get("_json", "") + delta["partial_json"]
        elif kind == "message_delta":
            message["stop_reason"] = event["delta"]["stop_reason"]
    for block in blocks:
        if "_json" in block:
            block["input"] = json.loads(block.pop("_json") or "{}")
    message["content"] = blocks
    return message


def test_gateway_retries_and_notes_keep_tito_clean(glm, tmp_path):
    from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, with_mock_server

    backend_model = PerfectGLM(glm.tito, SCRIPT)
    with patch.object(MockSGLangServer, "_compute_chat_completions_response", new=backend_model), \
            with_mock_server(model_name=glm.path, host="127.0.0.1") as backend:
        server, url = _serve(glm, backend.url)
        try:
            session_id = requests.post(f"{url}/sessions", timeout=10).json()["session_id"]
            admin_file = tmp_path / "admin"
            admin_file.write_text(ADMIN)
            args = SimpleNamespace(admin_token_file=str(admin_file), tokenizer=None, log_dir=str(tmp_path / "gw"),
                                   log_full=False, upstream_read_timeout=120.0, served_model=SERVED)
            app = cc_gateway.build_app(args)

            async def go():
                client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw",
                                           timeout=120)
                r = await client.post("/episodes", headers={"authorization": f"Bearer {ADMIN}"}, json={
                    "episode": "tito1", "token": "tok", "session_url": f"{url}/sessions/{session_id}",
                    "model": SERVED, "max_out": 4096, "reasoning_cap": 2048, "continue_max": 2})
                assert r.status_code == 200, r.text
                seen, body = await _claude_code(client, "tito1", "tok")
                summary = (await client.delete("/episodes/tito1",
                                               headers={"authorization": f"Bearer {ADMIN}"})).json()
                return seen, body, summary

            seen, body, summary = asyncio.run(go())
            session = requests.get(f"{url}/sessions/{session_id}", timeout=30).json()
            samples = requests.post(f"{url}/sessions/{session_id}/samples", json={}, timeout=60)
        finally:
            server.stop()

    assert not backend_model.script, "every scripted backend reply was used"
    assert len(backend_model.prompts) == len(SCRIPT) == summary["calls"]
    assert all(p["canonical"] for p in backend_model.prompts), [p["canonical"] for p in backend_model.prompts]
    assert summary["continuations"] == 3 and summary["rejected"] == 0 and not summary["upstream_errors"]

    stops = [m["stop_reason"] for m in seen]
    assert stops == ["tool_use", "tool_use", "max_tokens", "tool_use", "end_turn"], stops
    assert "[harness]" not in json.dumps(seen) and "[harness]" not in json.dumps(body)

    carried = [json.dumps(p["messages"]) for p in backend_model.prompts]
    assert "previous_reasoning" in carried[2] and SCRIPT[1][1] in carried[2]
    assert all(SCRIPT[i][1] in carried[6] for i in (3, 4, 5)) and "Output token limit hit" in carried[6]
    assert "model calls left" in carried[7]

    metadata = session["metadata"]
    assert metadata["tito_session_mismatch"] == []
    nodes = metadata["tree"]["nodes"]
    assert len(nodes) == len(SCRIPT)
    assert sum(1 for n in nodes if n["parent"] is None) == 1
    assert sum(1 for n in nodes if n["truncated"]) == 4
    assert all(n["closing_span"] for n in nodes if n["truncated"])
    assert samples.status_code == 200, samples.text[:500]
