"""Run 20 real ``claude -p`` sessions against the scripted mock, twice.

* ``raw``: Claude Code talks straight to the mock (role anthropic). The per-run request
  logs under ``<out>/raw/`` are what ``test_cc_gateway.py`` replays.
* ``gateway``: Claude Code -> cc_gateway -> mock (role session, Miles validation on).
  Every run must exit 0 with the scripted final text and no request may fail validation.

    python3 capture_cc_sessions.py --claude <bin> --out <dir> [--phase raw|gateway|both]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVED = "glm-5.3-flash"
FINAL_TEXT = {
    "bash_basic": "Done: listed files.", "write_read_edit": "File written, read and edited.",
    "errors": "Errors handled.", "multi_tool": "Found alpha in a.txt.", "long": "Long run complete.",
    "text_only": "Nothing to do; answering directly.", "hip_like": "Kernel written to solution/kernel.hip.",
}
SCENARIOS = list(FINAL_TEXT)
# (scenario, Claude Code extended thinking on, mock replies with thinking blocks)
RUNS = [(s, False, False) for s in SCENARIOS] + [(s, True, True) for s in SCENARIOS] + [
    ("long", True, False), ("errors", False, True), ("multi_tool", True, True),
    ("write_read_edit", False, True), ("hip_like", True, False), ("bash_basic", True, True)]
assert len(RUNS) == 20


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except urllib.error.HTTPError:
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"{url} did not come up")


def start(cmd: list[str], log: Path) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env={**os.environ})


def claude_env(home: Path, base_url: str, api_key: str, thinking: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE"))}
    env.update(HOME=str(home), ANTHROPIC_BASE_URL=base_url, ANTHROPIC_API_KEY=api_key,
               DISABLE_TELEMETRY="1", DISABLE_ERROR_REPORTING="1", DISABLE_AUTOUPDATER="1",
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", NO_PROXY="127.0.0.1,localhost",
               no_proxy="127.0.0.1,localhost")
    if thinking:
        env["MAX_THINKING_TOKENS"] = "4096"
    return env


def run_claude(args, index: int, scenario: str, env: dict, workdir: Path, out_dir: Path) -> dict:
    prompt = f"SCENARIO:{scenario} Follow the scripted steps for this test run."
    started = time.time()
    proc = subprocess.run([args.claude, "-p", prompt, "--model", SERVED, "--output-format", "json",
                           "--dangerously-skip-permissions", "--max-turns", "30"],
                          cwd=workdir, env=env, capture_output=True, text=True, timeout=args.timeout)
    (out_dir / f"run{index:02d}.stdout.json").write_text(proc.stdout)
    (out_dir / f"run{index:02d}.stderr.txt").write_text(proc.stderr)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {}
    text = result.get("result", "") if isinstance(result, dict) else ""
    return {"run": index, "scenario": scenario, "exit": proc.returncode, "elapsed_s": round(time.time() - started, 1),
            "result_ok": text.strip() == FINAL_TEXT[scenario], "result": text[:200],
            "is_error": result.get("is_error") if isinstance(result, dict) else None}


def phase_raw(args, out: Path) -> list[dict]:
    out_dir = out / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (scenario, cc_thinking, mock_thinking) in enumerate(RUNS):
        port = free_port()
        log = out_dir / f"run{index:02d}.requests.jsonl"
        log.unlink(missing_ok=True)
        cmd = [sys.executable, str(HERE / "mock_anthropic.py"), "--role", "anthropic", "--port", str(port),
               "--log", str(log)] + (["--thinking"] if mock_thinking else [])
        mock = start(cmd, out_dir / f"run{index:02d}.mock.txt")
        try:
            wait_http(f"http://127.0.0.1:{port}/health")
            with tempfile.TemporaryDirectory() as tmp:
                home, work = Path(tmp, "home"), Path(tmp, "work")
                home.mkdir(), work.mkdir()
                env = claude_env(home, f"http://127.0.0.1:{port}", "mock-key-not-a-secret", cc_thinking)
                row = run_claude(args, index, scenario, env, work, out_dir)
        finally:
            mock.terminate()
            mock.wait()
        requests = [json.loads(line) for line in log.read_text().splitlines()]
        row.update(cc_thinking=cc_thinking, mock_thinking=mock_thinking, requests=len(requests),
                   served_calls=sum(1 for r in requests if (r.get("body") or {}).get("model") == SERVED
                                    and r["path"] == "/v1/messages"),
                   raw_rejected_by_miles=sum(1 for r in requests if r.get("miles_verdict", "ok") != "ok"))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def phase_gateway(args, out: Path) -> list[dict]:
    out_dir = out / "gateway"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    tmp_secrets = Path(tempfile.mkdtemp())
    admin_file = tmp_secrets / "admin_token"
    admin_file.touch(mode=0o600)
    admin_file.write_text(secrets.token_hex(32))
    admin = admin_file.read_text()
    rows = []
    for index, (scenario, cc_thinking, mock_thinking) in enumerate(RUNS):
        session_port, gw_port = free_port(), free_port()
        session_log = out_dir / f"run{index:02d}.session.jsonl"
        mock = start([sys.executable, str(HERE / "mock_anthropic.py"), "--role", "session", "--port",
                      str(session_port), "--log", str(session_log)] + (["--thinking"] if mock_thinking else []),
                     out_dir / f"run{index:02d}.mock.txt")
        gateway = start([sys.executable, str(HERE.parent / "cc_gateway.py"), "--host", "127.0.0.1", "--port",
                         str(gw_port), "--admin-token-file", str(admin_file), "--log-dir", str(out_dir / "gw_logs"),
                         "--served-model", SERVED], out_dir / f"run{index:02d}.gateway.txt")
        try:
            wait_http(f"http://127.0.0.1:{session_port}/health")
            wait_http(f"http://127.0.0.1:{gw_port}/healthz")
            episode, token = f"cap-{index:02d}", secrets.token_hex(24)
            _admin(gw_port, admin, "POST", "/episodes", {
                "episode": episode, "token": token, "model": SERVED, "context": 131072, "max_out": 8192,
                "session_url": f"http://127.0.0.1:{session_port}/sessions/s{index:02d}"})
            with tempfile.TemporaryDirectory() as tmp:
                home, work = Path(tmp, "home"), Path(tmp, "work")
                home.mkdir(), work.mkdir()
                env = claude_env(home, f"http://127.0.0.1:{gw_port}/e/{episode}", token, cc_thinking)
                row = run_claude(args, index, scenario, env, work, out_dir)
            summary = _admin(gw_port, admin, "DELETE", f"/episodes/{episode}", None)
        finally:
            for proc in (gateway, mock):
                proc.terminate()
                proc.wait()
        upstream = [json.loads(line) for line in session_log.read_text().splitlines()] if session_log.exists() else []
        row.update(cc_thinking=cc_thinking, mock_thinking=mock_thinking, gateway=summary,
                   upstream_calls=len(upstream),
                   upstream_rejected=sum(1 for r in upstream if r.get("miles_verdict", "ok") != "ok"),
                   upstream_non_served=sum(1 for r in upstream if (r.get("body") or {}).get("model") != SERVED))
        rows.append(row)
        print(json.dumps(row), flush=True)
    shutil.rmtree(tmp_secrets)
    return rows


def _admin(port: int, admin: str, method: str, path: str, body: dict | None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"authorization": f"Bearer {admin}", "content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claude", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=["raw", "gateway", "both"], default="both")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.phase in ("raw", "both"):
        report["raw"] = phase_raw(args, out)
    if args.phase in ("gateway", "both"):
        report["gateway"] = phase_gateway(args, out)
    ok = True
    for phase, rows in report.items():
        good = sum(1 for r in rows if r["exit"] == 0 and r["result_ok"])
        bad_upstream = sum(r.get("upstream_rejected", 0) for r in rows)
        print(f"{phase}: {good}/{len(rows)} runs completed with the scripted result; "
              f"upstream rejections {bad_upstream}", flush=True)
        ok &= good == len(rows) and bad_upstream == 0
    (out / f"capture-report-{args.phase}.json").write_text(json.dumps(report, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
