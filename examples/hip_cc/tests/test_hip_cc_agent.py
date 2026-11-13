"""hip_cc_agent unit tests: final-grade queue, circuit breaker, episode flow and drops, and the
per-episode advantage in Miles' reward normalization. No network.

    PYTHONPATH=<miles> python3 -m pytest -q examples/hip_cc/tests/test_hip_cc_agent.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import hip_cc_agent  # noqa: E402
from hip_grader_client import Result, Stats  # noqa: E402

GRADE_OK = {"compiled": True, "correct": True, "audit": True, "method_consistent": True, "speedup": 1.8,
            "score": 0.4}


class ScriptedGrader:
    def __init__(self, replies: dict[str, list[Result]]):
        self.replies = replies
        self.calls: list[tuple[str, str, dict | None]] = []
        self.stats = Stats()

    def call(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        queue = self.replies[path]
        return queue.pop(0) if len(queue) > 1 else queue[0]


class Sleeps:
    def __init__(self, clock: list[float]):
        self.clock = clock
        self.slept: list[float] = []

    async def __call__(self, seconds):
        self.slept.append(seconds)
        self.clock[0] += seconds


def run(coro):
    return asyncio.run(coro)


def queue(grader, clock):
    sleeps = Sleeps(clock)
    return hip_cc_agent.GradeQueue(grader, 4, sleep=sleeps, clock=lambda: clock[0]), sleeps


def test_grade_retries_with_backoff_then_succeeds():
    grader = ScriptedGrader({"/grade": [Result("infra", 502), Result("infra", 0, error="transport"),
                                        Result("ok", 200, GRADE_OK)]})
    grades, sleeps = queue(grader, [0.0])
    body, attempts = run(grades.grade({"episode": "e"}))
    assert body["score"] == 0.4
    assert sleeps.slept == [30, 60]
    assert [a["kind"] for a in attempts] == ["infra", "infra", "ok"]


def test_grade_gives_up_within_15_minutes_and_never_scores_zero():
    grader = ScriptedGrader({"/grade": [Result("infra", 503)]})
    grades, sleeps = queue(grader, [0.0])
    body, attempts = run(grades.grade({"episode": "e"}))
    assert body is None
    assert sum(sleeps.slept) <= 15 * 60
    assert sleeps.slept == [30, 60, 120, 240]
    assert len(attempts) == 5


def test_grade_rejected_is_not_retried():
    grader = ScriptedGrader({"/grade": [Result("rejected", 400, {"error": "unknown task"})]})
    grades, sleeps = queue(grader, [0.0])
    body, attempts = run(grades.grade({"episode": "e"}))
    assert body is None and not sleeps.slept and len(attempts) == 1


def test_grade_ok_without_score_counts_as_unavailable():
    grader = ScriptedGrader({"/grade": [Result("ok", 200, {"compiled": True}), Result("ok", 200, GRADE_OK)]})
    grades, _ = queue(grader, [0.0])
    body, _ = run(grades.grade({"episode": "e"}))
    assert body["score"] == 0.4


def test_breaker_opens_below_floor_and_waits_until_recovered():
    grader = ScriptedGrader({"/health": [Result("ok", 200, {"healthy": 7}), Result("infra", 502),
                                         Result("ok", 200, {"healthy": 8})]})
    clock = [0.0]
    breaker = hip_cc_agent.Breaker(grader, floor=8, probe_s=30, clock=lambda: clock[0])
    sleeps = Sleeps(clock)
    waited = run(breaker.wait_closed(sleep=sleeps))
    assert waited == 60 and not breaker.open and breaker.trips == 1 and breaker.healthy == 8


def test_breaker_stays_closed_with_one_of_two_nodes():
    grader = ScriptedGrader({"/health": [Result("ok", 200, {"healthy": 8, "nodes": 1})]})
    breaker = hip_cc_agent.Breaker(grader, floor=8, probe_s=30)
    assert run(breaker.wait_closed()) == 0 and breaker.trips == 0


def test_breaker_caches_health_between_probes():
    grader = ScriptedGrader({"/health": [Result("ok", 200, {"healthy": 16})]})
    clock = [0.0]
    breaker = hip_cc_agent.Breaker(grader, floor=8, probe_s=30, clock=lambda: clock[0])
    run(breaker.refresh())
    clock[0] = 10
    run(breaker.refresh())
    assert len(grader.calls) == 1


# ---------------------------------------------------------------- episode flow


class FakeCluster:
    """One runner and one gateway behind an httpx MockTransport."""

    def __init__(self, result: dict | None, runner_status: int = 202):
        self.result = result
        self.runner_status = runner_status
        self.events: list[str] = []
        self.registered: dict = {}
        self.spec: dict = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, path, method = request.url.host, request.url.path, request.method
        if host == "runner" and path == "/health":
            return httpx.Response(200, json={"capacity": 2})
        if host == "runner" and path == "/episodes" and method == "POST":
            self.events.append("runner_start")
            self.spec = json.loads(request.content)
            return httpx.Response(self.runner_status, json={})
        if host == "runner" and path.startswith("/episodes/") and method == "GET":
            return httpx.Response(200, json={"status": "done", "result": self.result})
        if host == "runner" and method == "DELETE":
            self.events.append("runner_cancel")
            return httpx.Response(200, json={})
        if host == "gw" and path == "/episodes":
            self.events.append("gateway_register")
            self.registered = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})
        if host == "gw" and method == "DELETE":
            self.events.append("gateway_finish")
            return httpx.Response(200, json={"calls": 7, "stop_reason": None})
        return httpx.Response(404)


def make_agent(tmp_path, cluster: FakeCluster, grade_replies: list[Result]):
    nodes = tmp_path / "nodes.txt"
    nodes.write_text("http://runner:18500 http://gw:18400\n")
    config = hip_cc_agent.Config(nodes_file=str(nodes), episode_log=str(tmp_path / "episodes.jsonl"))
    grader = ScriptedGrader({"/grade": grade_replies, "/health": [Result("ok", 200, {"healthy": 8})]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(cluster.handler))
    agent = hip_cc_agent.Agent(config, grader=grader, client=client)
    agent.grades = hip_cc_agent.GradeQueue(grader, 4, sleep=Sleeps([0.0]), clock=lambda: 0.0)
    return agent, grader


LABEL = {"task_id": "t2h/cudaagent/fused_torch_1_relu6_72ecee24", "task_version": "6e0402dbba69efa9",
         "workspace_dir": "/data/ws/relu6", "split": "train"}
RUNNER_OK = {"ok": True, "final_code": "// kernel", "log_dir": "/logs/x",
             "metrics": {"reason": "exit", "checks": {"used": {"correctness": 2}}, "gateway_denials": 0}}


def test_episode_registers_before_start_and_returns_the_grade(tmp_path):
    cluster = FakeCluster(RUNNER_OK)
    agent, grader = make_agent(tmp_path, cluster, [Result("ok", 200, GRADE_OK)])
    out = run(agent.run("http://ss:9000/sessions/abc", LABEL))
    assert out["reward"] == 0.4 and out["hip_cc_correct"] and out["hip_cc_calls"] == 7
    assert cluster.events == ["gateway_register", "runner_start", "gateway_finish"]
    token = cluster.spec["gateway_token"]
    assert cluster.registered["token_sha256"] == hashlib.sha256(token.encode()).hexdigest()
    assert "token" not in cluster.registered and cluster.registered["session_url"] == "http://ss:9000/sessions/abc"
    assert cluster.spec["gateway_url"] == "http://gw:18400"
    grade_payload = next(p for _, path, p in grader.calls if path == "/grade")
    assert grade_payload["code"] == "// kernel" and grade_payload["task_id"] == LABEL["task_id"]
    logged = (tmp_path / "episodes.jsonl").read_text()
    assert token not in logged and "// kernel" not in logged


def test_runner_error_drops_the_episode(tmp_path):
    cluster = FakeCluster({**RUNNER_OK, "ok": False, "metrics": {"reason": "error"}})
    agent, grader = make_agent(tmp_path, cluster, [Result("ok", 200, GRADE_OK)])
    out = run(agent.run("http://ss/sessions/a", LABEL))
    assert out["reward"] is None and out["hip_cc_dropped"] == "runner:error"
    assert not any(path == "/grade" for _, path, _ in grader.calls)


def test_context_stop_is_still_graded(tmp_path):
    cluster = FakeCluster({**RUNNER_OK, "metrics": {"reason": "exit"}})
    agent, _ = make_agent(tmp_path, cluster, [Result("ok", 200, {**GRADE_OK, "score": 0.0, "correct": False})])
    out = run(agent.run("http://ss/sessions/a", LABEL))
    assert out["reward"] == 0.0 and "hip_cc_dropped" not in out


def test_unavailable_final_grade_drops_not_zero(tmp_path):
    cluster = FakeCluster(RUNNER_OK)
    agent, _ = make_agent(tmp_path, cluster, [Result("infra", 502)])
    out = run(agent.run("http://ss/sessions/a", LABEL))
    assert out["reward"] is None and out["hip_cc_dropped"] == "final_grade_unavailable"
    assert agent.dropped == 1


def test_runner_refusal_unregisters_from_gateway(tmp_path):
    cluster = FakeCluster(RUNNER_OK, runner_status=400)
    agent, _ = make_agent(tmp_path, cluster, [Result("ok", 200, GRADE_OK)])
    out = run(agent.run("http://ss/sessions/a", LABEL))
    assert out["reward"] is None
    assert cluster.events[:3] == ["gateway_register", "runner_start", "gateway_finish"]
    assert agent.runners.nodes[0].in_flight == 0


def test_label_comes_from_metadata():
    label = {**LABEL, "max_seq_len": 1}
    assert hip_cc_agent._as_dict(json.dumps(LABEL)) == LABEL
    assert hip_cc_agent._as_dict("t2h task") == {}
    assert label["task_id"] == LABEL["task_id"]


# ---------------------------------------------------------------- Miles integration


def _miles():
    try:
        from miles.ray.rollout import train_data_conversion  # noqa: PLC0415
        from miles.utils.types import Sample  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"miles not importable here: {type(exc).__name__}")
    return train_data_conversion, Sample


def test_one_advantage_per_episode():
    """Episode A has 3 kept leaves, B has 1: A's samples share one advantage, and B's equals -A's."""
    tdc, Sample = _miles()
    args = SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True,
                           n_samples_per_prompt=2, rollout_batch_size=1, reward_key=None)
    samples = []
    for rollout_id, reward, leaves in ((10, 1.0, 3), (11, 0.0, 1)):
        for _ in range(leaves):
            sample = Sample(index=rollout_id, group_index=0, reward=reward)
            sample.rollout_id = rollout_id
            samples.append(sample)
    raw, normalized = tdc._post_process_rewards(args, samples, None)
    assert raw == [1.0, 1.0, 1.0, 0.0]
    assert normalized[0] == normalized[1] == normalized[2]
    assert normalized[0] == pytest.approx(-normalized[3])
    assert normalized[0] == pytest.approx(0.7071, abs=1e-3)


def test_generate_marks_dropped_episodes_aborted(monkeypatch):
    _, Sample = _miles()
    kept = [Sample(index=1, reward=0.4, metadata={"hip_cc_episode": "a"})]
    dropped = [Sample(index=2, reward=None, metadata={"hip_cc_episode": "b", "hip_cc_dropped": "x"})]
    outputs = iter([SimpleNamespace(samples=kept), SimpleNamespace(samples=dropped)])

    async def fake_generate(fn_input):
        return next(outputs)

    # The real module imports sglang; only its generate is replaced here.
    fake = SimpleNamespace(generate=fake_generate)
    monkeypatch.setitem(sys.modules, "miles.rollout.generate_hub.agentic_tool_call", fake)
    for expected in (Sample.Status.PENDING, Sample.Status.ABORTED):
        sample = Sample(index=0, label=json.dumps(LABEL), metadata={})
        out = run(hip_cc_agent.generate(SimpleNamespace(sample=sample)))
        assert sample.metadata["task_id"] == LABEL["task_id"]
        assert out.samples[0].status == expected


def test_label_sets_the_kb_and_counters_cell(tmp_path):
    agent, _ = make_agent(tmp_path, FakeCluster(RUNNER_OK), [])
    spec = agent.episode_spec("e", "tok", LABEL)
    assert spec["kb"] is True and spec["counters_enabled"] is True
    spec = agent.episode_spec("e", "tok", {**LABEL, "kb": False, "counters": False})
    assert spec["kb"] is False and spec["counters_enabled"] is False
