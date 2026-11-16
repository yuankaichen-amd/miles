"""cc_gateway tests: normalization and replay of 20 captured ``claude -p`` sessions.

The captured logs come from ``capture_cc_sessions.py --phase raw`` (``$P3_RAW_DIR``). The
upstream is the mock session server, which validates every request with Miles' own
Anthropic adapter, so "accepted upstream" here means "accepted by the session server".

    PYTHONPATH=<miles> python3 -m pytest -q examples/hip_cc/tests/test_cc_gateway.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import cc_gateway  # noqa: E402
from mock_anthropic import SERVED_MODEL, MockServer, miles_validate  # noqa: E402

RAW_DIR = Path(os.environ.get("P3_RAW_DIR", "/shared_inference/yuankach/glm53/hip-cc/p3/raw"))
ADMIN = "admin-test-token"
UPSTREAM = "http://upstream/sessions/s1"


def captured_logs() -> list[Path]:
    return sorted(RAW_DIR.glob("run*.requests.jsonl"))


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def served_bodies(records: list[dict]) -> list[dict]:
    return [r["body"] for r in records
            if r["path"] == "/v1/messages" and (r.get("body") or {}).get("model") == SERVED_MODEL]


class Harness:
    """Gateway and mock session server in one process, joined by ASGI transports."""

    def __init__(self, tmp_path: Path, **gateway_args):
        admin_file = tmp_path / "admin"
        admin_file.write_text(ADMIN)
        args = SimpleNamespace(admin_token_file=str(admin_file), tokenizer=None, log_dir=str(tmp_path / "logs"),
                               log_full=False, upstream_read_timeout=60.0, served_model=SERVED_MODEL)
        vars(args).update(gateway_args)
        self.mock = MockServer("session", None, thinking=True)
        self.app = cc_gateway.build_app(args)
        self.gateway = self.app.state.gateway
        self.gateway.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.mock.app()))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gw")
        self.log_dir = Path(args.log_dir)

    async def register(self, episode="ep1", token="tok-1", **params):
        body = {"episode": episode, "token": token, "session_url": UPSTREAM, "model": SERVED_MODEL, **params}
        r = await self.client.post("/episodes", json=body, headers={"authorization": f"Bearer {ADMIN}"})
        assert r.status_code == 200, r.text
        return episode, token

    async def call(self, episode, token, body, path="/v1/messages"):
        return await self.client.post(f"/e/{episode}{path}?beta=true", json=body, headers={"x-api-key": token})

    async def finish(self, episode):
        r = await self.client.delete(f"/episodes/{episode}", headers={"authorization": f"Bearer {ADMIN}"})
        assert r.status_code == 200
        return r.json()


def run(coro):
    return asyncio.run(coro)


def simple_body(text="hi", **extra):
    return {"model": SERVED_MODEL, "max_tokens": 32000, "stream": False,
            "tools": [{"name": "Bash", "description": "run", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}], **extra}


# ---------------- normalization on the captured traffic ----------------

def test_have_twenty_captured_sessions():
    logs = captured_logs()
    if not logs:
        pytest.skip(f"no captured logs in {RAW_DIR}")
    assert len(logs) == 20
    assert all(served_bodies(load(p)) for p in logs)


@pytest.mark.parametrize("path", captured_logs(), ids=lambda p: p.name.split(".")[0])
def test_normalized_captured_requests_pass_miles_validation(path):
    for body in served_bodies(load(path)):
        normalized, stats = cc_gateway.normalize_request(body)
        assert miles_validate(json.dumps(normalized).encode()) is None
        assert cc_gateway.normalize_request(normalized)[0] == normalized
        wire = json.dumps(normalized)
        assert "cache_control" not in wire and '"is_error"' not in wire
        assert not set(normalized) & set(cc_gateway.DROPPED_TOP_LEVEL)
        assert len(normalized["messages"]) == len(body["messages"])
        uses = [b["id"] for m in normalized["messages"] if isinstance(m["content"], list)
                for b in m["content"] if b.get("type") == "tool_use"]
        results = [b["tool_use_id"] for m in normalized["messages"] if isinstance(m["content"], list)
                   for b in m["content"] if b.get("type") == "tool_result"]
        assert uses == results
        assert all(b.get("type") == "text" for b in normalized.get("system") or [] if isinstance(b, dict))


def test_captured_traffic_needs_normalization():
    """Raw Claude Code requests are what the session server rejects; this is why the gateway exists."""
    logs = captured_logs()
    if not logs:
        pytest.skip("no captured logs")
    bodies = [b for p in logs for b in served_bodies(load(p))]
    rejected = sum(1 for b in bodies if miles_validate(json.dumps(b).encode()))
    print(f"raw served requests {len(bodies)}, rejected by Miles before normalization {rejected}")
    assert rejected > 0


# ---------------- replay ----------------

@pytest.mark.parametrize("path", captured_logs(), ids=lambda p: p.name.split(".")[0])
def test_replay_captured_session(path, tmp_path):
    records = load(path)

    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_out=8192, context=131072, reasoning_cap=0, continue_max=0,
                                          reminders=False)
        served = 0
        for record in records:
            body = record.get("body")
            if record["path"] == "/v1/messages/count_tokens":
                r = await h.call(episode, token, body, "/v1/messages/count_tokens")
                assert r.status_code == 200 and r.json()["input_tokens"] > 0
                continue
            if record["path"] != "/v1/messages":
                continue
            before = len(h.mock.received)
            r = await h.call(episode, token, body)
            assert r.status_code == 200, r.text
            if body.get("model") == SERVED_MODEL:
                served += 1
                assert len(h.mock.received) == before + 1
                sent = h.mock.received[-1]
                assert sent["miles_verdict"] == "ok", sent["miles_verdict"]
                expected, _ = cc_gateway.normalize_request(body)
                assert {k: v for k, v in sent["body"].items() if k != "max_tokens"} == \
                    {k: v for k, v in expected.items() if k != "max_tokens"}
                assert sent["body"]["max_tokens"] <= min(8192, body.get("max_tokens", 8192))
                assert "x-cc-gateway-synthetic" not in r.headers
            else:
                assert len(h.mock.received) == before
                assert r.headers.get("x-cc-gateway-synthetic") == "1"
        summary = await h.finish(episode)
        assert summary["calls"] == served and summary["rejected"] == 0 and not summary["upstream_errors"]
        assert summary["stop_reason"] is None
        events = [json.loads(line)["event"] for line in (h.log_dir / f"{episode}.jsonl").read_text().splitlines()]
        assert events[0] == "register" and events[-1] == "finish"
        assert events.count("request") == served == events.count("response")

    run(go())


@pytest.mark.parametrize("path", captured_logs(), ids=lambda p: p.name.split(".")[0])
def test_replay_with_harness_notes_keeps_history_consistent(path, tmp_path):
    """With reminders on, every request still passes Miles validation, and whenever the client's history
    extends its previous request, what the session server sees extends what it saw before."""
    records = load(path)

    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_out=8192, context=131072)
        previous_client, previous_sent = None, None
        for record in records:
            body = record.get("body")
            if record["path"] != "/v1/messages" or body.get("model") != SERVED_MODEL:
                continue
            r = await h.call(episode, token, body)
            assert r.status_code == 200, r.text
            sent = h.mock.received[-1]
            assert sent["miles_verdict"] == "ok", sent["miles_verdict"]
            client = cc_gateway.normalize_request(body)[0]["messages"]
            sent_messages = sent["body"]["messages"]
            assert len(sent_messages) == len(client)
            if len(client) > 1:
                assert "[harness]" in json.dumps(sent_messages[-1])
            if previous_client is not None and client[:len(previous_client)] == previous_client:
                assert sent_messages[:len(previous_sent)] == previous_sent
            previous_client, previous_sent = client, sent_messages
        summary = await h.finish(episode)
        assert summary["rejected"] == 0 and not summary["upstream_errors"] and summary["continuations"] == 0

    run(go())


# ---------------- auth ----------------

def test_auth(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register()
        assert (await h.call(episode, "wrong", simple_body())).status_code == 401
        assert (await h.call("nope", token, simple_body())).status_code == 401
        assert (await h.client.post(f"/e/{episode}/v1/messages", json=simple_body(),
                                    headers={"authorization": f"Bearer {token}"})).status_code == 200
        assert (await h.client.post("/episodes", json={"episode": "x", "token": "t", "session_url": UPSTREAM},
                                    headers={"authorization": "Bearer wrong"})).status_code == 401
        assert (await h.client.get(f"/episodes/{episode}")).status_code == 401
        assert (await h.client.delete(f"/episodes/{episode}")).status_code == 401
        assert h.gateway.episodes[episode].token_sha256 == cc_gateway.sha256(token)
        log = (h.log_dir / f"{episode}.jsonl").read_text()
        assert token not in log and ADMIN not in log

    run(go())


# ---------------- budgets ----------------

def test_output_clamp(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_out=100)
        await h.call(episode, token, simple_body())
        assert h.mock.received[-1]["body"]["max_tokens"] == 100
        episode2, token2 = await h.register("ep2", "tok-2", max_out=16384, context=6000, stop_margin=100)
        body = simple_body("x" * 9000)
        await h.call(episode2, token2, body)
        input_est = h.gateway.counter.count(cc_gateway.request_text(cc_gateway.normalize_request(body)[0]))
        assert h.mock.received[-1]["body"]["max_tokens"] == 6000 - input_est - 256

    run(go())


def test_stop_rule_context_limit(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(context=4000, stop_margin=1000)
        r = await h.call(episode, token, simple_body("y" * 3000))
        assert r.status_code == 200 and len(h.mock.received) == 1
        r = await h.call(episode, token, simple_body("y" * 12000))
        assert r.headers.get("x-cc-gateway-synthetic") == "1" and r.json()["stop_reason"] == "end_turn"
        assert len(h.mock.received) == 1
        r = await h.call(episode, token, simple_body("short"))
        assert r.headers.get("x-cc-gateway-synthetic") == "1" and len(h.mock.received) == 1
        summary = await h.finish(episode)
        assert summary["stop_reason"] == "context_limit" and summary["calls"] == 1

    run(go())


def test_stop_rule_streaming_reply_is_valid_sse(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(context=1000, stop_margin=500)
        r = await h.call(episode, token, simple_body("z" * 6000, stream=True))
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
        assert [e["type"] for e in events] == ["message_start", "content_block_start", "content_block_delta",
                                               "content_block_stop", "message_delta", "message_stop"]
        assert events[4]["delta"]["stop_reason"] == "end_turn"

    run(go())


def test_max_calls_and_wall_clock(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_calls=2)
        for _ in range(3):
            await h.call(episode, token, simple_body())
        assert len(h.mock.received) == 2
        assert (await h.finish(episode))["stop_reason"] == "max_calls"
        episode, token = await h.register("ep2", "tok-2", wall_s=0)
        r = await h.call(episode, token, simple_body())
        assert r.headers.get("x-cc-gateway-synthetic") == "1" and len(h.mock.received) == 2
        assert (await h.finish(episode))["stop_reason"] == "wall_clock"

    run(go())


def test_input_estimate_uses_previous_usage(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register()
        body = simple_body()
        r = await h.call(episode, token, body)
        reply = r.json()
        ep = h.gateway.episodes[episode]
        assert ep.last_input_tokens == reply["usage"]["input_tokens"]
        body2 = {**body, "messages": body["messages"] + [
            {"role": "assistant", "content": reply["content"]},
            {"role": "user", "content": [{"type": "text", "text": "abcdef"}]}]}
        assert h.gateway._input_estimate(ep, cc_gateway.normalize_request(body2)[0]) > \
            reply["usage"]["input_tokens"] + reply["usage"]["output_tokens"]

    run(go())


# ---------------- upstream failures ----------------

def test_upstream_failures(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register()
        h.mock.fail_next = [500, 429, 400]
        assert (await h.call(episode, token, simple_body())).status_code == 500
        assert (await h.call(episode, token, simple_body())).status_code == 429
        assert (await h.call(episode, token, simple_body())).status_code == 400
        ep = h.gateway.episodes[episode]
        assert ep.calls == 1 and ep.upstream_errors == {"500": 1, "429": 1, "400": 1}

        async def boom(*a, **k):
            raise httpx.ConnectError("down")
        h.gateway.client.post = boom
        r = await h.call(episode, token, simple_body())
        assert r.status_code == 529 and r.json()["error"]["type"] == "overloaded_error"
        assert ep.calls == 1 and ep.upstream_errors["transport"] == 1

    run(go())


# ---------------- normalization cases Claude Code can produce ----------------

def test_normalize_cases():
    body = {
        "model": SERVED_MODEL, "max_tokens": 10, "thinking": {"type": "enabled", "budget_tokens": 1024},
        "metadata": {"user_id": "u"}, "context_management": {"edits": []}, "output_config": {"effort": "high"},
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Bash", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}},
                  {"type": "web_search_20250305", "name": "web_search"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "q", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "t", "signature": "s"},
                {"type": "redacted_thinking", "data": "xx"},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": True, "content": "boom"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": [
                {"type": "text", "text": "png"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}]},
        ],
    }
    out, stats = cc_gateway.normalize_request(body)
    assert miles_validate(json.dumps(out).encode()) is None
    assert [t["name"] for t in out["tools"]] == ["Bash"]
    assert out["messages"][1]["content"][0] == {"type": "thinking", "thinking": "t"}
    assert len(out["messages"][1]["content"]) == 2
    assert out["messages"][2]["content"][0]["content"] == "Error: boom"
    assert out["messages"][4]["content"][0]["content"][1]["type"] == "text"
    assert stats["is_error_results"] == 1 and stats["dropped_blocks"] == 1 and stats["replaced_blocks"] == 1
    assert sorted(stats["dropped_fields"]) == ["context_management", "metadata", "output_config", "thinking"]

    with pytest.raises(cc_gateway.RequestRejected):
        cc_gateway.normalize_request({"model": SERVED_MODEL, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]}]})


def test_image_request_is_rejected_not_forwarded(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register()
        body = simple_body()
        body["messages"][0]["content"].append(
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}})
        r = await h.call(episode, token, body)
        assert r.status_code == 400 and not h.mock.received
        assert (await h.finish(episode))["rejected"] == 1

    run(go())


def test_logs_are_deltas_after_first_call(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register()
        body = simple_body(system="be brief")
        reply = (await h.call(episode, token, body)).json()
        body2 = {**body, "messages": body["messages"] + [
            {"role": "assistant", "content": reply["content"]}, {"role": "user", "content": "next"}]}
        await h.call(episode, token, body2)
        await h.finish(episode)
        reqs = [json.loads(l) for l in (h.log_dir / f"{episode}.jsonl").read_text().splitlines()]
        reqs = [r for r in reqs if r["event"] == "request"]
        assert "system" in reqs[0]["request"] and "messages" in reqs[0]["request"]
        assert "system" not in reqs[1]["request"] and len(reqs[1]["request"]["messages_delta"]) == 2
        assert reqs[1]["request"]["messages_total"] == 3

    run(go())


def test_requests_off_the_allowlist_are_logged(tmp_path):
    h = Harness(tmp_path)

    async def main():
        episode, token = await h.register()
        statuses = [
            (await h.client.get("/sessions/s1/samples")).status_code,
            (await h.client.post("/flush_cache")).status_code,
            (await h.client.post("/episodes", json={}, headers={"authorization": "Bearer wrong"})).status_code,
            (await h.client.get(f"/episodes/{episode}")).status_code,
            (await h.client.post(f"/e/{episode}/v1/messages", json={}, headers={"x-api-key": "wrong"})).status_code,
            (await h.client.get(f"/e/{episode}/v1/messages", headers={"x-api-key": token})).status_code,
        ]
        await h.finish(episode)
        return statuses, token

    statuses, token = run(main())
    assert statuses == [404, 404, 401, 401, 401, 405], statuses
    text = (h.log_dir / "denials.jsonl").read_text()
    records = [json.loads(line) for line in text.splitlines()]
    assert [(r["method"], r["path"], r["status"]) for r in records] == [
        ("GET", "/sessions/s1/samples", 404), ("POST", "/flush_cache", 404), ("POST", "/episodes", 401),
        ("GET", "/episodes/ep1", 401), ("POST", "/e/ep1/v1/messages", 401), ("GET", "/e/ep1/v1/messages", 405)]
    assert token not in text and ADMIN not in text and "wrong" not in text


# ---------------- reasoning cap, continuation of cut-off replies, harness notes ----------------

NUDGE = ("Output token limit hit. Resume directly \u2014 no apology, no recap of what you were doing. Pick up "
         "mid-thought if that is where the cut happened. Break remaining work into smaller pieces.")


def thinking(text):
    return {"type": "thinking", "thinking": text, "signature": "sig"}


def tool(name, args):
    return {"type": "tool_use", "id": f"toolu_{abs(hash((name, json.dumps(args)))) % 10**12:012d}",
            "name": name, "input": args}


def result_for(block, text="ok"):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": text}]}


def events_of(h, episode):
    return [json.loads(line) for line in (h.log_dir / f"{episode}.jsonl").read_text().splitlines()]


def test_cut_reasoning_is_carried_into_a_gateway_retry(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_out=1000, reasoning_cap=100, reminders=False)
        write = tool("Write", {"file_path": "/workspace/solution/kernel.hip", "content": "k"})
        h.mock.replies = [([thinking("R1 tiling plan")], "max_tokens"), ([thinking("short"), write], "tool_use")]
        body = simple_body("task")
        reply = (await h.call(episode, token, body)).json()
        assert reply["stop_reason"] == "tool_use" and reply["content"][1]["id"] == write["id"]
        first, second = (r["body"] for r in h.mock.received)
        assert first["max_tokens"] == second["max_tokens"] == 100
        assert first["messages"] == body["messages"]
        note = second["messages"][-1]["content"][-1]["text"]
        assert note.startswith("task\n\n[harness] Your previous reply hit the 100-token reasoning cap")
        assert note.endswith("<previous_reasoning>\nR1 tiling plan\n</previous_reasoning>")
        assert all(r["miles_verdict"] == "ok" for r in h.mock.received)

        body2 = {**body, "messages": body["messages"] + [{"role": "assistant", "content": reply["content"]},
                                                         result_for(write)]}
        await h.call(episode, token, body2)
        third = h.mock.received[-1]["body"]["messages"]
        assert third[0] == second["messages"][0]
        assert third[2] == body2["messages"][2]
        summary = await h.finish(episode)
        assert summary["calls"] == 3 and summary["continuations"] == 1
        (cont,) = [e for e in events_of(h, episode) if e["event"] == "continue"]
        assert cont["round"] == 1 and cont["cut_in_reasoning"] and cont["limit"] == 100

    run(go())


def test_cut_while_acting_under_the_cap_retries_with_full_output(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_out=1000, reasoning_cap=100, reminders=False)
        compile_ = tool("Bash", {"command": "aka-eval compile"})
        h.mock.replies = [([thinking("plan"), compile_], "max_tokens"), ([compile_], "tool_use")]
        reply = (await h.call(episode, token, simple_body("task"))).json()
        assert reply["stop_reason"] == "tool_use"
        first, second = (r["body"] for r in h.mock.received)
        assert (first["max_tokens"], second["max_tokens"]) == (100, 1000)
        note = second["messages"][-1]["content"][-1]["text"]
        assert "100-token output cap" in note and "<previous_reasoning>\nplan\n" in note

    run(go())


def test_continuations_are_bounded_and_the_client_retry_keeps_the_carry(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_out=500, reasoning_cap=0, continue_max=2, reminders=False)
        bash = tool("Bash", {"command": "ls"})
        h.mock.replies = [([thinking("A")], "max_tokens"), ([thinking("B")], "max_tokens"),
                          ([thinking("C")], "max_tokens"), ([bash], "tool_use")]
        body = simple_body("task")
        reply = (await h.call(episode, token, body)).json()
        assert reply["stop_reason"] == "max_tokens" and reply["content"][0]["thinking"] == "C"
        assert [r["body"]["max_tokens"] for r in h.mock.received] == [500, 500, 500]
        # Claude Code drops the thinking-only reply and appends its nudge to the last user turn.
        retry = {**body, "messages": [{"role": "user", "content": body["messages"][0]["content"] + [
            {"type": "text", "text": NUDGE}]}]}
        reply = (await h.call(episode, token, retry)).json()
        assert reply["stop_reason"] == "tool_use"
        last = h.mock.received[-1]["body"]["messages"][-1]["content"]
        assert last[0] == {"type": "text", "text": "task"} and last[1]["text"].startswith(NUDGE)
        assert "<previous_reasoning>\nA\n\nB\n\nC\n</previous_reasoning>" in last[1]["text"]
        summary = await h.finish(episode)
        assert summary["calls"] == 4 and summary["continuations"] == 2

    run(go())


def test_no_continuation_when_disabled_or_out_of_calls(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(reasoning_cap=0, continue_max=0)
        h.mock.replies = [([thinking("A")], "max_tokens")]
        assert (await h.call(episode, token, simple_body())).json()["stop_reason"] == "max_tokens"
        assert len(h.mock.received) == 1
        episode, token = await h.register("ep2", "tok-2", reasoning_cap=0, max_calls=1)
        h.mock.replies = [([thinking("A")], "max_tokens")]
        assert (await h.call(episode, token, simple_body())).json()["stop_reason"] == "max_tokens"
        assert len(h.mock.received) == 2
        assert (await h.finish(episode))["continuations"] == 0

    run(go())


def test_streaming_cut_reasoning_is_continued(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(max_out=1000, reasoning_cap=100, reminders=False)
        bash = tool("Bash", {"command": "ls"})
        h.mock.replies = [([thinking("streamed plan")], "max_tokens"), ([bash], "tool_use")]
        r = await h.call(episode, token, simple_body("task", stream=True))
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[5:]) for line in r.text.splitlines() if line.startswith("data:")]
        assert [e["delta"]["stop_reason"] for e in events if e["type"] == "message_delta"] == ["tool_use"]
        assert "<previous_reasoning>\nstreamed plan\n" in json.dumps(h.mock.received[-1]["body"]["messages"])

    run(go())


def test_reminders_report_time_calls_and_progress(tmp_path):
    async def go():
        h = Harness(tmp_path)
        episode, token = await h.register(reasoning_cap=0, max_calls=40, wall_s=1200)
        body = simple_body("task")
        await h.call(episode, token, body)
        assert h.mock.received[-1]["body"]["messages"] == body["messages"]

        def note_of_last():
            content = h.mock.received[-1]["body"]["messages"][-1]["content"]
            return content[-1]["content"]

        messages = list(body["messages"])
        steps = [
            (tool("Read", {"file_path": "/workspace/README.md"}), "solution/kernel.hip has not been written yet."),
            (tool("Read", {"file_path": "/kb/INDEX.md"}), "Write a simple correct kernel"),
            (tool("Bash", {"command": "cat > solution/kernel.hip <<'EOF'\nx\nEOF"}), "has not been compiled yet"),
            (tool("Bash", {"command": "aka-eval compile"}), "No correctness check has been run yet"),
            (tool("Bash", {"command": "aka-eval correctness"}), None),
        ]
        sent_before = None
        for index, (block, expected) in enumerate(steps):
            if index == 1:
                h.gateway.episodes[episode].created -= 600
            messages += [{"role": "assistant", "content": [block]}, result_for(block, "out")]
            await h.call(episode, token, {**body, "messages": messages})
            sent = h.mock.received[-1]["body"]["messages"]
            note = note_of_last()
            assert note.startswith("out\n\n[harness] ") and f"of 40 model calls left." in note
            assert f"{40 - index - 1} of 40 model calls left." in note
            if expected:
                assert expected in note
            else:
                assert note.endswith("model calls left.")
            if sent_before is not None:
                assert sent[:len(sent_before)] == sent_before
            sent_before = sent
        assert "10m" in note_of_last() and "of 20m used" in note_of_last()

    run(go())


def test_append_note_forms():
    note = "[harness] n"
    assert cc_gateway._append_note({"role": "user", "content": "a"}, note)["content"] == "a\n\n[harness] n"
    text = cc_gateway._append_note({"role": "user", "content": [{"type": "text", "text": "a"}]}, note)
    assert text["content"] == [{"type": "text", "text": "a\n\n[harness] n"}]
    res = {"type": "tool_result", "tool_use_id": "t"}
    assert cc_gateway._append_note({"role": "user", "content": [{**res, "content": "o"}]}, note)["content"][0][
        "content"] == "o\n\n[harness] n"
    assert cc_gateway._append_note({"role": "user", "content": [res]}, note)["content"][0]["content"] == note
    listed = cc_gateway._append_note({"role": "user", "content": [{**res, "content": [{"type": "text", "text": "o"}]}]},
                                     note)
    assert listed["content"][0]["content"] == [{"type": "text", "text": "o"}, {"type": "text", "text": note}]
    original = {"role": "user", "content": [{"type": "text", "text": "a"}]}
    cc_gateway._append_note(original, note)
    assert original == {"role": "user", "content": [{"type": "text", "text": "a"}]}


def test_progress_detection():
    def assistant(*blocks):
        return {"role": "assistant", "content": list(blocks)}

    assert cc_gateway._progress([assistant(tool("Read", {"file_path": "solution/kernel.hip"}))]) == \
        {"kernel": False, "compile": False, "correctness": False}
    assert cc_gateway._progress([assistant(tool("Edit", {"file_path": "/workspace/solution/kernel.hip"}))])["kernel"]
    assert cc_gateway._progress([assistant(tool("Bash", {"command": "tee solution/kernel.hip < x"}))])["kernel"]
    assert not cc_gateway._progress([assistant(tool("Bash", {"command": "cat solution/kernel.hip"}))])["kernel"]
