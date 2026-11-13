"""P4 dry run: one Claude Code episode through cc-runner against a mock session server and a mock grader.

Runs on a compute node's host Python (the runner is stdlib-only). The mock session server and
cc-gateway need Starlette, so they run in the dev image with host networking. Nothing here touches
the Crusoe token: the mock grader gets a throwaway token file.

    python3 dry_run_episode.py --image hip-cc-rollout:<tag> --workspace <rendered relu6 ws> --out <dir>

Checks: the episode finishes with the known-good kernel; budgets, dedupe, the 502 retry, URL
failover and the counters 501 behave as plan §5.4/§7.3 say; the forwarder denies other paths; the
red-team probe inside the container reaches nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEV_IMAGE = "rocm/sgl-dev:miles-rocm10-mi30x-20261001"
SERVED = "glm-5.3-flash"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, token: str | None = None, body: dict | None = None, timeout: float = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_http(url: str, timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3).read()
            return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise RuntimeError(f"{url} did not come up")


def secret_file(path: Path) -> str:
    value = secrets.token_hex(24)
    path.touch(mode=0o600)
    path.write_text(value)
    return value


# The red-team probes that must each appear in the runner's node-wide denials.jsonl.
HOST_DENIALS_EXPECTED = {
    ("gateway_denied", "/episodes"), ("gateway_denied", "/e/other-episode/v1/messages"),
    ("gateway_denied", "/sessions/s1/samples"), ("gateway_denied", "/flush_cache"),
    ("gateway_denied", "/update_weights_from_disk"), ("eval_denied", "/grade"), ("eval_denied", "/tasks/upload"),
    ("eval_denied", "/health"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wall-s", type=int, default=900)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    secret_dir = Path(f"/tmp/cc-dry-{os.getpid()}")
    secret_dir.mkdir(mode=0o700)
    grader_token = secret_file(secret_dir / "grader_token")
    admin = secret_file(secret_dir / "admin_token")
    runner_token = secret_file(secret_dir / "runner_token")
    grader_port, session_port, gw_port, runner_port = (free_port() for _ in range(4))
    url_file = out / "grader_urls.txt"
    url_file.write_text(f"http://127.0.0.1:{free_port()}\nhttp://127.0.0.1:{grader_port}\n")
    env = {**os.environ, "HIP_SANDBOX_TOKEN_FILE": str(secret_dir / "grader_token"), "HIP_SF_URL_FILE": str(url_file)}
    procs = []
    dev = f"cc-dry-dev-{os.getpid()}"
    try:
        procs.append(subprocess.Popen([sys.executable, str(HERE / "mock_grader.py"), "--port", str(grader_port),
                                       "--token-file", str(secret_dir / "grader_token"), "--log",
                                       str(out / "grader.jsonl"), "--fail-first", "1"],
                                      stdout=open(out / "grader.txt", "w"), stderr=subprocess.STDOUT))
        uid = f"{os.getuid()}:{os.getgid()}"
        inner = (f"cd {ROOT} && (python3 tests/mock_anthropic.py --role session --port {session_port} "
                 f"--log {out}/session.jsonl --default-scenario hip_episode > {out}/session.txt 2>&1 &) && "
                 f"python3 cc_gateway.py --host 127.0.0.1 --port {gw_port} --admin-token-file "
                 f"{secret_dir}/admin_token --log-dir {out}/gw_logs --served-model {SERVED} > {out}/gateway.txt 2>&1")
        subprocess.run(["docker", "run", "-d", "--rm", "--name", dev, "--user", uid, "--network", "host",
                        "-e", "HOME=/tmp", "-e", "OMP_NUM_THREADS=1",
                        "-e", f"PYTHONPATH={ROOT.parents[1]}", "-v", "/home/yuankach:/home/yuankach:ro",
                        "-v", f"{out}:{out}", "-v", f"{secret_dir}:{secret_dir}:ro", DEV_IMAGE, "bash", "-c", inner],
                       check=True, stdout=subprocess.DEVNULL)
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "cc_runner.py"), "--host", "127.0.0.1", "--port",
                                       str(runner_port), "--image", args.image, "--capacity", "4", "--token-file",
                                       str(secret_dir / "runner_token"), "--sock-root", f"/tmp/cc-sock-dry-{os.getpid()}",
                                       "--log-root", str(out / "episodes")],
                                      env=env, stdout=open(out / "runner.txt", "w"), stderr=subprocess.STDOUT))
        wait_http(f"http://127.0.0.1:{grader_port}/healthz")
        wait_http(f"http://127.0.0.1:{gw_port}/healthz")
        episode, token = f"dry-{int(time.time())}", secrets.token_hex(24)
        http("POST", f"http://127.0.0.1:{gw_port}/episodes", admin, {
            "episode": episode, "token": token, "model": SERVED, "context": 131072, "max_out": 16384,
            "max_calls": 40, "wall_s": args.wall_s, "session_url": f"http://127.0.0.1:{session_port}/sessions/{episode}"})
        time.sleep(1)
        started = time.time()
        reply = http("POST", f"http://127.0.0.1:{runner_port}/episodes", runner_token, {
            "episode": episode, "task_id": "t2h/cudaagent/fused_torch_1_relu6_72ecee24",
            "task_version": "6e0402dbba69efa9", "workspace_dir": args.workspace,
            "gateway_url": f"http://127.0.0.1:{gw_port}", "gateway_token": token, "model": SERVED,
            "wall_s": args.wall_s})
        print("start:", reply, flush=True)
        result = {}
        while time.time() - started < args.wall_s + 300:
            result = http("GET", f"http://127.0.0.1:{runner_port}/episodes/{episode}?wait=60", runner_token,
                          timeout=90)
            if "result" in result:
                break
        gateway = http("DELETE", f"http://127.0.0.1:{gw_port}/episodes/{episode}", admin)
        health = http("GET", f"http://127.0.0.1:{runner_port}/health", runner_token)
    finally:
        for proc in procs:
            proc.terminate()
        subprocess.run(["docker", "rm", "-f", dev], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for path in secret_dir.iterdir():
            path.unlink()
        secret_dir.rmdir()
    res = result.get("result", {})
    metrics = res.get("metrics", {})
    log_dir = Path(res.get("log_dir", out))
    events = [json.loads(line) for line in (log_dir / "events.jsonl").read_text().splitlines()] \
        if (log_dir / "events.jsonl").exists() else []
    checks = [e for e in events if e["event"] == "check"]
    stream = (log_dir / "stream.jsonl").read_text() if (log_dir / "stream.jsonl").exists() else ""
    grader_log = [json.loads(line) for line in (out / "grader.jsonl").read_text().splitlines()]
    good = (HERE / "kernels" / "relu6_good.hip").read_text()

    def jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    runner_denials = jsonl(out / "episodes" / "denials.jsonl")
    gateway_denials = jsonl(out / "gw_logs" / "denials.jsonl")
    verdicts = {
        "finished_with_good_kernel": res.get("final_code") == good,
        "reason_exit": metrics.get("reason") == "exit",
        "compiles_counted_2": metrics.get("checks", {}).get("used", {}).get("compile") == 2,
        "correctness_charged_2": metrics.get("checks", {}).get("used", {}).get("correctness") == 2,
        "dedupe_not_charged": any(c.get("kind") == "cached" for c in checks),
        "retry_after_502_ok": any(g.get("injected") == 502 for g in grader_log)
        and checks and checks[0].get("kind") == "ok",
        "performance_charged_1": metrics.get("checks", {}).get("used", {}).get("performance") == 1,
        "counters_unavailable_not_charged": metrics.get("checks", {}).get("used", {}).get("counters") == 0
        and metrics.get("checks", {}).get("unavailable", {}).get("counters", 0) >= 1,
        "forwarder_denials": metrics.get("gateway_denials", 0) >= 2,
        "red_team_blocked": not re.search(r"REACHED \d", stream) and len(re.findall(r"blocked \d", stream)) >= 3,
        "no_host_paths": "cannot access '/shared_inference'" in stream and "cannot access '/home/yuankach'" in stream,
        "grader_saw_no_code_in_log": all("code" not in g for g in grader_log),
        "denials_in_host_log": {(d["event"], d.get("path")) for d in runner_denials} >= HOST_DENIALS_EXPECTED
        and any(d["event"] == "gateway_denied" and "/../" in d.get("path", "") for d in runner_denials),
        "gateway_calls": gateway.get("calls"),
    }
    report = {"episode": episode, "wall_s": round(time.time() - started, 1), "verdicts": verdicts,
              "metrics": metrics, "gateway": gateway, "runner_health": health, "log_dir": str(log_dir),
              "host_denials": {"runner": runner_denials, "gateway": gateway_denials}}
    (out / "dry-run-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = all(v for k, v in verdicts.items() if k != "gateway_calls")
    print("DRY RUN:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
