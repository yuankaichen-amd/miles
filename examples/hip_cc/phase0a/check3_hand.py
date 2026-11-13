"""Phase 0a check 3: one real episode, end to end, with the session kept for inspection.

Runs ``hip_cc_agent.Agent.run`` on the first task of CC_TASKS (as the driver does), but reads the
full session snapshot before deleting it, then checks:
* exactly 1 root in the session tree and no TITO mismatch;
* every recorded request's ``input_ids`` equals a request SGLang logged (``--log-requests`` level 3,
  json, under $RUN_DIR/sglang-requests);
* the checks reached Crusoe and returned (runner events.jsonl), and the final grade returned.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hip_cc_agent  # noqa: E402
from hip_cc_phase0 import load_rows, tree_summary  # noqa: E402


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def engine_requests(log_dir: Path) -> dict[str, int]:
    """sha256 of each logged input_ids list -> count."""
    seen: dict[str, int] = {}
    for path in sorted(log_dir.glob("*.log*")):
        with path.open(errors="replace") as handle:
            for line in handle:
                start = line.find("{")
                if start < 0 or '"request.received"' not in line:
                    continue
                try:
                    record = json.loads(line[start:])
                except ValueError:
                    continue
                ids = (record.get("obj") or {}).get("input_ids")
                if isinstance(ids, list) and ids and isinstance(ids[0], int):
                    digest = ids_digest(ids)
                    seen[digest] = seen.get(digest, 0) + 1
    return seen


async def run_one(row: dict) -> tuple[dict, dict]:
    session_server = os.environ["CC_SESSION_URLS"].split(",")[0].rstrip("/")
    agent = hip_cc_agent.Agent(hip_cc_agent.Config.from_env())
    hip_cc_agent._AGENT = agent
    async with httpx.AsyncClient() as client:
        sampling = {"temperature": float(os.environ.get("CC_TEMPERATURE", "1.0"))}
        session_id = (await client.post(f"{session_server}/sessions", json=sampling, timeout=30)).json()["session_id"]
        base_url = f"{session_server}/sessions/{session_id}"
        try:
            result = await agent.run(base_url, row)
            snapshot = (await client.get(base_url, timeout=120)).json()
        finally:
            await client.delete(base_url, timeout=30)
    return result, snapshot


def main() -> int:
    out = Path(os.environ["CC_OUT"])
    os.environ.setdefault("CC_EPISODE_LOG", str(out / "agent_episodes.jsonl"))
    row = load_rows(os.environ["CC_TASKS"], 1)[0]
    result, snapshot = asyncio.run(run_one(row))
    (out / "session_snapshot.json").write_text(json.dumps(snapshot))
    metadata = snapshot.get("metadata") or {}
    tree = tree_summary(metadata)
    records = snapshot.get("records") or []
    engine = engine_requests(Path(os.environ["RUN_DIR"]) / "sglang-requests")
    matched = [ids_digest(r["request"]["input_ids"]) in engine for r in records if r.get("request", {}).get("input_ids")]
    episode_log = [json.loads(line) for line in (out / "agent_episodes.jsonl").read_text().splitlines()]
    record = episode_log[-1] if episode_log else {}
    log_dir = Path(record.get("log_dir") or "")
    events = [json.loads(line) for line in (log_dir / "events.jsonl").read_text().splitlines()] \
        if (log_dir / "events.jsonl").is_file() else []
    checks = [e for e in events if e.get("event") == "check"]
    report = {
        "task_id": row["task_id"], "result": result, "tree": tree, "records": len(records),
        "input_ids_matched": sum(matched), "input_ids_records": len(matched),
        "engine_requests_logged": sum(engine.values()),
        "checks": [{k: c.get(k) for k in ("action", "kind", "status", "seconds", "charged", "passed", "grader_error")}
                   for c in checks],
        "grade": record.get("grade"), "dropped": record.get("dropped"), "log_dir": str(log_dir),
    }
    report["pass"] = bool(
        tree.get("roots") == 1 and not tree.get("tito_session_mismatch") and matched and all(matched)
        and record.get("grade") is not None and not record.get("dropped")
        and any(c.get("kind") == "ok" for c in checks)
        and all(c.get("kind") in ("ok", "cached", "budget") for c in checks))
    (out / "check3.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    print("CHECK 3:", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
