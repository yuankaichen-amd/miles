"""Phase 0 episode driver: no trainer, same agent path as RL (plan Phase 0a/0b).

For each task row and sample it opens a session on the standalone session server v2, runs
``hip_cc_agent.Agent.run`` against it, records the session tree summary and the episode result,
and deletes the session. Rows come from ``$CC_TASKS`` (``t2h_manifest.py workspaces`` output).

    CC_TASKS, CC_TASK_LIMIT (0 = all), CC_SAMPLES, CC_CONCURRENCY, CC_SESSION_URLS (comma list,
    one session server per rollout node; episodes go round-robin), CC_OUT,
    CC_RESUME (1 = skip finished task/sample pairs), CC_TEMPERATURE, CC_TOP_P
plus the ``hip_cc_agent`` variables. Results go to ``$CC_OUT/episodes.jsonl``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hip_cc_agent  # noqa: E402

logger = logging.getLogger("hip_cc_phase0")


def load_rows(path: str, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rows = [{**(row.get("label") or {}), **{k: v for k, v in row.items() if k != "label"}} for row in rows]
    return rows[:limit] if limit > 0 else rows


def finished_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if "dropped" not in record:
            keys.add((record["task_id"], int(record["sample"])))
    return keys


def tree_summary(metadata: dict) -> dict:
    tree = metadata.get("tree") or {}
    nodes = tree.get("nodes") or []
    return {"nodes": len(nodes), "roots": sum(1 for n in nodes if n.get("parent") is None),
            "leaves": len(tree.get("leaves") or []), "truncated": sum(1 for n in nodes if n.get("truncated")),
            "tokens": len(metadata.get("accumulated_token_ids") or []),
            "tito_session_mismatch": metadata.get("tito_session_mismatch")}


async def one(agent: hip_cc_agent.Agent, client: httpx.AsyncClient, session_url: str, row: dict, sample: int,
              sampling: dict) -> dict:
    response = await client.post(f"{session_url}/sessions", json=sampling, timeout=30)
    response.raise_for_status()
    session_id = response.json()["session_id"]
    base_url = f"{session_url}/sessions/{session_id}"
    try:
        result = await agent.run(base_url, row)
        info = await client.get(base_url, timeout=60)
        tree = tree_summary(info.json().get("metadata", {})) if info.status_code == 200 else {}
    finally:
        try:
            await client.delete(base_url, timeout=30)
        except httpx.HTTPError:
            pass
    return {"task_id": row["task_id"], "sample": sample, "session": tree, **result}


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    env = os.environ
    out = Path(env["CC_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CC_EPISODE_LOG", str(out / "agent_episodes.jsonl"))
    rows = load_rows(env["CC_TASKS"], int(env.get("CC_TASK_LIMIT", "0")))
    samples = int(env.get("CC_SAMPLES", "4"))
    sampling = {k: float(env[v]) for k, v in (("temperature", "CC_TEMPERATURE"), ("top_p", "CC_TOP_P")) if v in env}
    results = out / "episodes.jsonl"
    done = finished_keys(results) if env.get("CC_RESUME") == "1" else set()
    work = [(row, s) for row in rows for s in range(samples) if (row["task_id"], s) not in done]
    logger.info("%d tasks x %d samples, %d to run, %d already finished", len(rows), samples, len(work), len(done))
    agent = hip_cc_agent.Agent(hip_cc_agent.Config.from_env())
    hip_cc_agent._AGENT = agent
    semaphore = asyncio.Semaphore(int(env.get("CC_CONCURRENCY", "16")))
    client = httpx.AsyncClient()
    started = time.time()
    counts = {"done": 0, "dropped": 0, "reward_sum": 0.0}

    session_urls = [u.strip().rstrip("/") for u in env["CC_SESSION_URLS"].split(",") if u.strip()]

    async def guarded(index: int, row: dict, sample: int) -> None:
        async with semaphore:
            try:
                session_url = session_urls[index % len(session_urls)]
                record = await one(agent, client, session_url, row, sample, sampling)
            except Exception as exc:  # noqa: BLE001
                record = {"task_id": row["task_id"], "sample": sample, "dropped": f"driver:{type(exc).__name__}",
                          "error": str(exc)[:300]}
            if record.get("hip_cc_dropped"):
                record["dropped"] = record["hip_cc_dropped"]
            with results.open("a") as f:
                f.write(json.dumps(record) + "\n")
            counts["done"] += 1
            counts["dropped"] += "dropped" in record
            counts["reward_sum"] += float(record.get("reward") or 0.0)
            logger.info("%d/%d done (%d dropped), %s sample %d reward %s, %.0f s elapsed", counts["done"],
                        len(work), counts["dropped"], row["task_id"], sample, record.get("reward"),
                        time.time() - started)

    await asyncio.gather(*(guarded(i, row, s) for i, (row, s) in enumerate(work)))
    graded = counts["done"] - counts["dropped"]
    summary = {"episodes": counts["done"], "dropped": counts["dropped"],
               "mean_reward": counts["reward_sum"] / graded if graded else None,
               "breaker_trips": agent.breaker.trips, "wall_s": round(time.time() - started, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
