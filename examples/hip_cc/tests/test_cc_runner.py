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

    def __init__(self, script: dict[str, list[Result]] | None = None):
        self.script = script or {}
        self.calls: list[tuple[str, dict]] = []
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

    def call_with_retry(self, method, path, payload=None, *, retries=1, delay_s=5.0, sleep=None):
        result = self.call(method, path, payload)
        for _ in range(retries):
            if result.kind != "infra":
                break
            result = self.call(method, path, payload)
        return result


class NullLog:
    def __init__(self):
        self.events = []

    def __call__(self, event, **fields):
        self.events.append({"event": event, **fields})


def make_proxy(tmp_path, client, **spec_overrides):
    ws = tmp_path / "ws"
    (ws / "solution").mkdir(parents=True, exist_ok=True)
    (ws / "solution" / "kernel.hip").write_text("// stub\n")
    body = {"episode": "ep1", "task_id": "t2h/x/y", "task_version": "v", "workspace_dir": str(ws),
            "gateway_url": "http://127.0.0.1:1", "gateway_token": SECRET, "model": "m", **spec_overrides}
    spec = cc_runner.EpisodeSpec.from_json(body)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    return cc_runner.EvalProxy(spec, client, NullLog(), pool, retry_delay_s=0)


def run(coro):
    return asyncio.run(coro)


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
