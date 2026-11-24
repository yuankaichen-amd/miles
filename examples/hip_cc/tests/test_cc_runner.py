"""cc_runner unit tests: eval-proxy budgets, dedupe, retries and unavailable checks, the forwarder
allowlist, the socket-bind guard, resource remarks and stream metrics. No docker, no network.

    PYTHONPATH=<miles> python3 -m pytest -q examples/hip_cc/tests/test_cc_runner.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import sys
import tarfile
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import cc_http  # noqa: E402
import cc_runner  # noqa: E402
import cc_workspace  # noqa: E402
from hip_grader_client import Result, Stats  # noqa: E402

aka_eval = SourceFileLoader("aka_eval", str(HERE.parent / "image" / "aka_eval.py")).load_module()

SECRET = "Zq9-gateway-secret-xK2"
GOOD = "__global__ void k() {}  // good"
WRONG = "__global__ void k() {}  // wrong"


class FakeClient:
    """Scripted grader replies keyed by path; records every payload."""

    def __init__(self, script: dict[str, list[Result]] | None = None, timeout: float = 900.0):
        self.script = script or {}
        self.calls: list[tuple[str, dict]] = []
        self.timeouts: list[float | None] = []
        self.timeout = timeout
        self.stats = Stats()

    def call(self, method, path, payload=None):
        self.calls.append((path, payload))
        queue = self.script.get(path)
        if queue:
            result = queue.pop(0) if len(queue) > 1 else queue[0]
        else:
            correct = "good" in (payload or {}).get("code", "")
            result = Result("ok", 200, {"compiled": True, "correct": correct, "elapsed_s": 1.0})
        self.stats.record(result)
        return result

    def call_with_retry(self, method, path, payload=None, *, retries=1, delay_s=5.0, sleep=None, timeout=None,
                        terminal_timeout=False):
        self.timeouts.append(timeout)
        result = self.call(method, path, payload)
        for _ in range(retries):
            if result.kind != "infra" or (terminal_timeout and result.timed_out):
                break
            result = self.call(method, path, payload)
        return result


class NullLog:
    def __init__(self):
        self.events = []

    def __call__(self, event, **fields):
        self.events.append({"event": event, **fields})


def make_proxy(tmp_path, client, proxy_kwargs=None, **spec_overrides):
    ws = tmp_path / "ws"
    (ws / "solution").mkdir(parents=True, exist_ok=True)
    (ws / "solution" / "kernel.hip").write_text("// stub\n")
    body = {"episode": "ep1", "task_id": "t2h/x/y", "task_version": "v", "workspace_dir": str(ws),
            "gateway_url": "http://127.0.0.1:1", "gateway_token": SECRET, "model": "m", **spec_overrides}
    spec = cc_runner.EpisodeSpec.from_json(body)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    return cc_runner.EvalProxy(spec, client, NullLog(), pool, retry_delay_s=0, **(proxy_kwargs or {}))


def run(coro):
    return asyncio.run(coro)


TIMEOUT = Result("infra", 0, None, "transport: TimeoutError", timed_out=True)


def test_correctness_is_charged_once_per_code_and_cached(tmp_path):
    client = FakeClient()
    proxy = make_proxy(tmp_path, client)
    ok, text = run(proxy.check("correctness", GOOD))
    assert ok and "correctness check 1/10" in text
    ok, text = run(proxy.check("correctness", GOOD))
    assert ok and "not charged" in text
    assert proxy.used["correctness"] == 1 and len(client.calls) == 1
    run(proxy.check("correctness", WRONG))
    assert proxy.used["correctness"] == 2
    assert [p["seq"] for _, p in client.calls] == [1, 2]


def test_budget_exhaustion_is_reported_and_not_sent(tmp_path):
    client = FakeClient()
    proxy = make_proxy(tmp_path, client, budgets={"correctness": 2})
    for i in range(2):
        run(proxy.check("correctness", f"{GOOD} {i}"))
    ok, text = run(proxy.check("correctness", f"{GOOD} 3"))
    assert not ok and "budget exhausted" in text
    assert len(client.calls) == 2


def test_infra_retries_once_then_reports_unavailable_uncharged(tmp_path):
    client = FakeClient({"/tool/performance": [Result("infra", 502, {}, "HTTP 502")]})
    proxy = make_proxy(tmp_path, client)
    ok, text = run(proxy.check("performance", GOOD))
    assert not ok and "not counted against your budget" in text
    assert proxy.used["performance"] == 0 and proxy.unavailable["performance"] == 1
    assert len(client.calls) == 2


def test_retry_after_502_succeeds_and_charges_once(tmp_path):
    client = FakeClient({"/tool/correctness": [Result("infra", 502, {}, "HTTP 502"),
                                               Result("ok", 200, {"correct": True})]})
    proxy = make_proxy(tmp_path, client)
    ok, _ = run(proxy.check("correctness", GOOD))
    assert ok and proxy.used["correctness"] == 1
    seqs = [p["seq"] for _, p in client.calls]
    assert seqs == [1, 1], "a retry must reuse the seq so the grader dedupes it"


def test_timeout_is_terminal_uncharged_and_caps_the_action(tmp_path):
    client = FakeClient({"/tool/correctness": [TIMEOUT]})
    proxy = make_proxy(tmp_path, client)
    ok, text = run(proxy.check("correctness", GOOD))
    # Same treatment as any infra failure before the cap: unavailable, not charged, nothing cached.
    assert not ok and "check unavailable (infrastructure); not counted against your budget" in text
    assert proxy.used["correctness"] == 0 and proxy.unavailable["correctness"] == 1
    assert len(client.calls) == 1, "a timeout must not be retried"
    ok, text = run(proxy.check("correctness", WRONG))
    assert not ok and "timed out on an earlier correctness check" in text and "not counted" in text
    assert len(client.calls) == 1 and proxy.unavailable["correctness"] == 2 and proxy.used["correctness"] == 0
    ok, _ = run(proxy.check("performance", GOOD))
    assert ok and len(client.calls) == 2, "the cap is per action"
    assert [e.get("kind") for e in proxy.log.events if e["event"] == "check"] == ["infra", "timeout_cap", "ok"]
    metrics = proxy.metrics()
    assert metrics["timeouts"]["correctness"] == 1 and metrics["checks_inflight"] == 0


def test_timeout_cap_resets_after_an_ok_result(tmp_path):
    ok_reply = Result("ok", 200, {"correct": False})
    client = FakeClient({"/tool/correctness": [TIMEOUT, ok_reply, TIMEOUT, TIMEOUT, ok_reply]})
    proxy = make_proxy(tmp_path, client, proxy_kwargs={"max_consecutive_timeouts": 2})
    results = [run(proxy.check("correctness", f"{WRONG} {i}"))[0] for i in range(5)]
    # timeout (1), ok (reset to 0), timeout (1), timeout (2), then capped without a call
    assert results == [False, True, False, False, False]
    assert len(client.calls) == 4
    assert proxy.consecutive_timeouts["correctness"] == 2 and proxy.timeouts["correctness"] == 3


def test_grader_error_reply_does_not_reset_the_cap(tmp_path):
    error_reply = Result("ok", 200, {"status": "error", "detail": "profiling timeout"})
    client = FakeClient({"/tool/correctness": [TIMEOUT, error_reply, TIMEOUT]})
    proxy = make_proxy(tmp_path, client, proxy_kwargs={"max_consecutive_timeouts": 2})
    for i in range(4):
        run(proxy.check("correctness", f"{WRONG} {i}"))
    assert len(client.calls) == 3 and proxy.consecutive_timeouts["correctness"] == 2


def test_late_failure_near_the_clamped_timeout_counts_as_a_timeout(tmp_path):
    import time as _time

    class Slow(FakeClient):
        def call(self, method, path, payload=None):
            _time.sleep(0.3)
            return super().call(method, path, payload)

    client = Slow({"/tool/correctness": [Result("infra", 502, {}, "HTTP 502")]})
    proxy = make_proxy(tmp_path, client)
    proxy.arm(0.3)
    run(proxy.check("correctness", GOOD))
    assert proxy.consecutive_timeouts["correctness"] == 1


def test_no_grader_call_after_the_episode_is_done_or_closed(tmp_path):
    client = FakeClient()
    proxy = make_proxy(tmp_path, client)
    proxy.done.set()
    ok, text = run(proxy.check("correctness", GOOD))
    assert not ok and "episode is over" in text and client.calls == []
    proxy = make_proxy(tmp_path, client)
    proxy.close()
    ok, text = run(proxy.check("performance", GOOD))
    assert not ok and "episode is over" in text and client.calls == []
    assert proxy.log.events == [], "nothing is written to a closed episode log"


def test_checks_queued_behind_a_stuck_check_are_dropped_when_the_episode_closes(tmp_path):
    import threading

    release = threading.Event()

    class Stuck(FakeClient):
        def call(self, method, path, payload=None):
            if not self.calls:
                release.wait(5)
            return super().call(method, path, payload)

    client = Stuck()
    proxy = make_proxy(tmp_path, client)

    async def scenario():
        first = asyncio.create_task(proxy.check("correctness", GOOD))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(proxy.check("correctness", WRONG))
        await asyncio.sleep(0.05)
        assert proxy.metrics()["checks_inflight"] == 1
        proxy.close()
        release.set()
        return await first, await queued

    (ok1, _), (ok2, text2) = run(scenario())
    assert ok1 and not ok2 and "episode is over" in text2
    assert len(client.calls) == 1 and proxy.inflight == 0
    assert proxy.log.events == [], "the late result of the in-flight check is not logged after close"


def test_grader_call_timeout_is_clamped_to_the_remaining_wall_time(tmp_path):
    client = FakeClient(timeout=900.0)
    proxy = make_proxy(tmp_path, client)
    run(proxy.check("correctness", f"{GOOD} 0"))
    assert client.timeouts[-1] == 900.0, "unarmed: the client default"
    proxy.arm(100.0)
    run(proxy.check("correctness", f"{GOOD} 1"))
    assert 99.0 < client.timeouts[-1] <= 100.0
    proxy.arm(5000.0)
    run(proxy.check("correctness", f"{GOOD} 2"))
    assert client.timeouts[-1] == 900.0, "never longer than the client default"
    proxy.arm(-1.0)
    ok, text = run(proxy.check("correctness", f"{GOOD} 3"))
    assert not ok and "time limit has passed" in text and len(client.calls) == 3


def test_counters_need_passing_correctness_and_501_is_not_charged(tmp_path):
    client = FakeClient({"/tool/counters": [Result("unsupported", 501, {}, "HTTP 501")]})
    proxy = make_proxy(tmp_path, client)
    ok, text = run(proxy.check("counters", GOOD))
    assert not ok and "passing correctness" in text and not client.calls
    run(proxy.check("correctness", GOOD))
    ok, text = run(proxy.check("counters", GOOD))
    assert not ok and "unavailable" in text and "not counted" in text
    assert proxy.used["counters"] == 0


def test_grader_side_error_is_unavailable_not_charged(tmp_path):
    error = {"status": "error", "stage": "counters", "exit": -9, "detail": "profiling pass exceeded 180s"}
    for body in (error, {"summary": json.dumps(error)}):
        client = FakeClient({"/tool/counters": [Result("ok", 200, body)]})
        proxy = make_proxy(tmp_path, client)
        run(proxy.check("correctness", GOOD))
        ok, text = run(proxy.check("counters", GOOD))
        assert not ok and "profiling pass exceeded 180s" in text and "not counted" in text
        assert proxy.used["counters"] == 0 and proxy.unavailable["counters"] == 1
    assert cc_runner.grader_error({"compiled": False, "status": "error", "compile_error": "x"}) is None


def test_counters_disabled_in_run(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient(), counters_enabled=False)
    run(proxy.check("correctness", GOOD))
    ok, text = run(proxy.check("counters", GOOD))
    assert not ok and "unavailable in this run" in text


def test_compile_budget(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient(), budgets={"compile": 1})
    assert proxy.compile_grant("a")["ok"]
    assert not proxy.compile_grant("b")["ok"]


def test_proxy_rejects_wrong_token_and_unknown_route(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient())
    bad = cc_http.Request("POST", "/check", "HTTP/1.1", [("authorization", "Bearer nope")], b"{}")
    assert b"401" in run(proxy.handle(bad)).split(b"\r\n", 1)[0]
    unknown = cc_http.Request("POST", "/tool/correctness", "HTTP/1.1", [("authorization", f"Bearer {proxy.token}")],
                              b"{}")
    with pytest.raises(cc_http.HTTPError):
        run(proxy.handle(unknown))


def test_grader_log_events_carry_no_code(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient())
    run(proxy.check("correctness", GOOD))
    assert GOOD not in json.dumps(proxy.log.events)


def test_spec_rejects_unrendered_or_forbidden_workspace(tmp_path):
    with pytest.raises(cc_http.HTTPError):
        cc_runner.EpisodeSpec.from_json({"episode": "e", "task_id": "t", "task_version": "v",
                                         "workspace_dir": str(tmp_path), "gateway_url": "u",
                                         "gateway_token": "x", "model": "m"})
    with pytest.raises(cc_workspace.IsolationError):
        cc_runner.EpisodeSpec.from_json({"episode": "e", "task_id": "t", "task_version": "v",
                                         "workspace_dir": "/data/hip_ref/x", "gateway_url": "u",
                                         "gateway_token": "x", "model": "m"})


def test_spec_repr_hides_token(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient())
    assert SECRET not in repr(proxy.spec) and "gateway_token" not in proxy.spec.public()


@pytest.mark.parametrize("path,allowed", [
    ("/e/ep1/v1/messages", True),
    ("/e/ep1/v1/messages/../../../episodes", False),
    ("/e/ep1/v1/messages%2F..%2Fx", False),
    ("/e/ep1/v1/messages/count_tokens", True),
    ("/e/ep2/v1/messages", False),
    ("/episodes", False),
    ("/sessions/x/samples", False),
    ("/e/ep1/../../episodes", False),
])
def test_forwarder_allowlist(tmp_path, path, allowed):
    proxy = make_proxy(tmp_path, FakeClient())
    forwarder = cc_runner.GatewayForwarder(proxy.spec, NullLog())
    assert forwarder.allowed("POST", path) is allowed
    assert forwarder.allowed("GET", "/e/ep1/v1/messages") is False


def test_proxy_denials_reach_node_denial_log(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient())
    proxy.denial_log = NullLog()
    auth = [("authorization", f"Bearer {proxy.token}")]
    run(proxy.handle(cc_http.Request("POST", "/check", "HTTP/1.1", [("authorization", "Bearer nope")], b"{}")))
    for request in (cc_http.Request("GET", "/check", "HTTP/1.1", auth, b""),
                    cc_http.Request("POST", "/sessions/x/samples", "HTTP/1.1", auth, b"{}")):
        with pytest.raises(cc_http.HTTPError):
            run(proxy.handle(request))
    reasons = [e["reason"] for e in proxy.denial_log.events]
    assert reasons == ["token", "method", "route"] and proxy.denials == 3
    assert all(e["episode"] == "ep1" for e in proxy.denial_log.events)
    assert proxy.token not in json.dumps(proxy.denial_log.events + proxy.log.events)


def test_forwarder_denies_and_logs_over_unix_socket(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient())
    denials = NullLog()
    forwarder = cc_runner.GatewayForwarder(proxy.spec, NullLog(), denial_log=denials)
    sock = str(tmp_path / "gw.sock")
    probes = [b"GET /sessions/s1/samples HTTP/1.1\r\nHost: x\r\n\r\n",
              b"POST /flush_cache HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
              b"POST /e/ep1/v1/messages/../../../episodes HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n",
              b"garbage\r\n\r\n"]

    async def main():
        server = await asyncio.start_unix_server(forwarder, path=sock)
        statuses = []
        async with server:
            for probe in probes:
                reader, writer = await asyncio.open_unix_connection(sock)
                writer.write(probe)
                await writer.drain()
                statuses.append((await reader.read()).split(b"\r\n", 1)[0])
                writer.close()
        return statuses

    statuses = run(main())
    assert all(b" 403 " in s for s in statuses[:3]), statuses
    assert b" 502 " in statuses[3] or b" 400 " in statuses[3], statuses
    assert [e["event"] for e in denials.events] == ["gateway_denied"] * 4
    assert [e["path"] for e in denials.events[:3]] == ["/sessions/s1/samples", "/flush_cache",
                                                       "/e/ep1/v1/messages/../../../episodes"]
    assert denials.events[3]["reason"] == "malformed request" and forwarder.denials == 4


def test_socket_bind_guard(tmp_path):
    root = str(tmp_path / "socks")
    cc_workspace.container_args("c", "img", 10001, 10001, fwd_dir=f"{root}/ep1", sock_root=root)
    for bad in (f"{root}/ep1/../..:/fwd:ro", f"{root}/ep1:/fwd", "/home/yuankach:/fwd:ro",
                "/shared_inference/x:/fwd:ro", f"{root}:/fwd:ro"):
        args = ["docker", "run", "--rm", "-v", bad, "img"]
        with pytest.raises(cc_workspace.IsolationError):
            cc_workspace.check_docker_args(args, sock_root=root)


# From a real gfx950 build of tests/kernels/relu6_good.hip (cpp_extension, verbose ninja output).
REMARKS = """\
src/kernel.hip:7:1: remark: Function Name: _Z10relu6_vec4PK15HIP_vector_typeIfLj4EEPS0_l [-Rpass-analysis=kernel-resource-usage]
    7 | __global__ void __launch_bounds__(256) relu6_vec4(const float4* __restrict__ x, float4* __restrict__ out, int64_t n4) {
      | ^
src/kernel.hip:7:1: remark:     TotalSGPRs: 18 [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     VGPRs: 12 [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
src/kernel.hip:7:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
      | ^
"""


def test_resource_remarks_parse_and_report():
    kernels = aka_eval.parse_resource_remarks(REMARKS + REMARKS)
    assert kernels == [{"name": "_Z10relu6_vec4PK15HIP_vector_typeIfLj4EEPS0_l", "sgpr": 18, "vgpr": 12, "agpr": 0,
                        "scratch": 0, "dynamic_stack": "False", "occupancy": 8, "sgpr_spill": 0, "vgpr_spill": 0,
                        "lds": 0}]
    report = aka_eval.resource_report(kernels)
    assert "VGPR 12" in report and "SGPR 18" in report and "LDS 0 B" in report and "occ_lds" not in report
    with_lds = aka_eval.resource_report([{**kernels[0], "lds": 4096}])
    assert "occ_lds 32 workgroups/CU" in with_lds  # 4096 B rounds up to 5120 B of the 160 KiB LDS
    assert aka_eval.resource_report([]).startswith("resources: no device kernels")


def test_compile_requests_the_clang17_abi(tmp_path, monkeypatch):
    """`aka-eval compile` passes -fclang-abi-compat=17 to hipcc (extra_cuda_cflags), so const_data_ptr<T>() links
    against libtorch's export."""
    import subprocess
    import types

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="error: stop here\n")

    audit = types.ModuleType("aka_audit")
    audit.PolicyError = type("PolicyError", (Exception,), {})
    audit.audit_hip_sources = lambda sources, name: None
    monkeypatch.setitem(sys.modules, "aka_audit", audit)
    monkeypatch.setattr(aka_eval, "BUILD_ROOT", tmp_path)
    monkeypatch.setattr(aka_eval.subprocess, "run", fake_run)
    ok, text = aka_eval.compile_candidate("// kernel")
    assert not ok and text.startswith("compile: FAIL")
    (argv,) = calls
    assert argv[1:3] == ["-c", aka_eval.BUILD_SCRIPT]
    cflags = argv[6:]
    assert "-fclang-abi-compat=17" in cflags and "-O3" in cflags and aka_eval.REMARK_FLAG in cflags
    assert "extra_cuda_cflags=sys.argv[4:]" in aka_eval.BUILD_SCRIPT


def test_stream_metrics():
    stream = "\n".join([
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"},
                                                                 {"type": "text", "text": "x"}]}}),
        json.dumps({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Write"}]}}),
        "not json",
        json.dumps({"type": "result", "subtype": "success", "num_turns": 2, "is_error": False}),
    ])
    metrics = cc_runner.stream_metrics(stream)
    assert metrics["assistant_messages"] == 2
    assert metrics["tool_calls"] == {"Bash": 1, "Write": 1}
    assert metrics["result"]["num_turns"] == 2 and metrics["stream_parse_errors"] == 1


def test_claude_env_and_run_script_carry_no_secret(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient())
    env = cc_runner.claude_env(proxy.spec)
    assert SECRET not in json.dumps(env)
    assert env["ANTHROPIC_BASE_URL"].endswith("/e/ep1")
    script = cc_runner.RUN_SH
    assert "--disallowedTools" in script and "gw_token" in script


def test_budgets_reach_prompt_and_system_prompt(tmp_path):
    proxy = make_proxy(tmp_path, FakeClient())
    spec = proxy.spec
    spec.wall_s, spec.max_turns = 1200, 40
    ws = Path(spec.workspace_dir)
    (ws / "README.md").write_text("# t2h/x/y\n\nintro\n")
    prompt = cc_runner.build_prompt(ws, wall_s=spec.wall_s, max_calls=spec.max_turns)
    assert "20 minutes of wall clock and 40 model calls" in prompt
    assert prompt.index("Budget (hard limits") < prompt.index("Task instructions from README.md")
    assert "run `aka-eval compile`" in prompt and "`aka-eval correctness` passes" in prompt
    assert "10 minutes of wall clock and 12 model calls" in cc_runner.build_prompt(ws, wall_s=600, max_calls=12)

    append = cc_runner.system_append(spec)
    assert "20 minutes of wall clock and 40 model calls" in append
    assert "don't need to wrap up early" in append and "Context management" in append
    script = cc_runner.RUN_SH.format(exports="", wall=1200, max_turns=40, tools=cc_runner.TOOLS,
                                     disallowed=cc_runner.DISALLOWED_TOOLS)
    assert '--append-system-prompt "$(cat /home/agent/.cc/system_append.md)"' in script
    with tarfile.open(fileobj=io.BytesIO(cc_runner.home_tar(spec, "eval-token", prompt))) as tar:
        assert tar.extractfile(".cc/system_append.md").read().decode() == append
        assert tar.extractfile(".cc/prompt.md").read().decode() == prompt


def test_kb_off_episode_hides_every_kb_pointer(tmp_path):
    from cc_workspace import KB_POINTER  # noqa: PLC0415

    proxy = make_proxy(tmp_path, FakeClient())
    ws = Path(proxy.spec.workspace_dir)
    (ws / "README.md").write_text(f"# t2h/x/y\n\nintro\n\n{KB_POINTER}## Interface\n\nrun\n")
    assert "/kb" in cc_runner.build_prompt(ws)
    assert "/kb" not in cc_runner.build_prompt(ws, with_kb=False)

    def readme(data):
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            return tar.extractfile("README.md").read().decode()

    assert "/kb" in readme(cc_runner.workspace_tar(ws))
    off = readme(cc_runner.workspace_tar(ws, with_kb=False))
    assert "/kb" not in off and "## Interface" in off
    body = {"episode": "e", "task_id": "t", "task_version": "v", "workspace_dir": str(ws), "gateway_url": "u",
            "gateway_token": "x", "model": "m", "kb": False, "prompt": "see /kb/INDEX.md"}
    with pytest.raises(cc_http.HTTPError):
        cc_runner.EpisodeSpec.from_json(body)
    assert cc_runner.EpisodeSpec.from_json({**body, "prompt": ""}).kb is False
