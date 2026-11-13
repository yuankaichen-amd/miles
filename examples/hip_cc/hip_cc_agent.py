"""Miles custom agent for Claude Code HIP episodes (plan §3.2, §5.4).

    --custom-generate-function-path hip_cc_agent.generate
    --custom-agent-function-path hip_cc_agent.run
    --custom-rollout-log-function-path hip_cc_agent.log_rollout_metrics

For each episode ``run``

1. waits while the circuit breaker is open (Crusoe pool below the healthy floor),
2. mints a per-episode gateway token and registers the episode with the cc-gateway of the node
   it picks (only the token's sha256 goes to the gateway; the token itself goes to the runner,
   which writes it to the container's tmpfs home),
3. posts the episode to that node's cc-runner and long-polls for the final file,
4. queues the final grade on Crusoe with backoff (30 s, 1, 2, 4, 8 min, at most 15 min),
5. returns ``{"reward": score, ...}``; the session server assigns that one reward to every kept
   sample of the episode.

An episode whose runner failed for infrastructure reasons, or whose final grade never came back,
is dropped: ``generate`` marks its samples ABORTED, and Miles' aborted filter drops the group. It
is never scored 0.

One advantage per episode: under session server v2 ``agentic_tool_call.generate`` stamps every
sample of an episode with the input sample's ``rollout_id``, and
``train_data_conversion._normalize_rewards_by_rollout`` normalizes one reward per rollout inside
each prompt group, so an episode with many kept leaves weighs the same as one with a single leaf.
``tests/test_hip_cc_agent.py`` pins that behavior.

Configuration (environment; files only, never secrets):
    CC_NODES_FILE                lines "<runner_url> <gateway_url>", one per node running cc_node.sh
    CC_RUNNER_TOKEN_FILE         0600 bearer for the runners
    CC_GATEWAY_ADMIN_TOKEN_FILE  0600 admin bearer for the gateways
    HIP_SF_URL_FILE, HIP_SANDBOX_TOKEN_FILE   Crusoe relay URLs and grader token
    CC_CONTEXT, CC_MAX_OUT, CC_STOP_MARGIN, CC_MAX_CALLS, CC_WALL_S, CC_CHECKS_CORRECTNESS,
    CC_CHECKS_PERF, CC_CHECKS_COUNTERS, CC_CHECKS_COMPILE, CC_COUNTERS (0/1), CC_SERVED_MODEL,
    CC_HEALTHY_FLOOR, CC_EPISODE_LOG
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import secrets
import stat
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from hip_grader_client import GraderClient, Result

logger = logging.getLogger("hip_cc_agent")

GRADE_BACKOFF_S = (30, 60, 120, 240, 480)
GRADE_DEADLINE_S = 15 * 60
RUNNER_POLL_S = 120
DROP_REASONS = ("error", "cancelled")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _read_secret(path: str | None) -> str | None:
    if not path:
        return None
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode & 0o077:
        raise RuntimeError(f"{path} must be mode 0600, is {oct(mode)}")
    return Path(path).read_text().strip()


@dataclass
class Config:
    nodes_file: str = ""
    runner_token_file: str | None = None
    gateway_token_file: str | None = None
    served_model: str = "glm-5.3-flash"
    context: int = 131072
    max_out: int = 16384
    stop_margin: int = 4096
    max_calls: int = 40
    wall_s: int = 1200
    budgets: dict = field(default_factory=dict)
    counters: bool = True
    healthy_floor: int = 8
    breaker_probe_s: float = 30.0
    grade_concurrency: int = 64
    episode_log: str | None = None

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ
        return cls(
            nodes_file=env.get("CC_NODES_FILE", ""),
            runner_token_file=env.get("CC_RUNNER_TOKEN_FILE"),
            gateway_token_file=env.get("CC_GATEWAY_ADMIN_TOKEN_FILE"),
            served_model=env.get("CC_SERVED_MODEL", "glm-5.3-flash"),
            context=_env_int("CC_CONTEXT", 131072),
            max_out=_env_int("CC_MAX_OUT", 16384),
            stop_margin=_env_int("CC_STOP_MARGIN", 4096),
            max_calls=_env_int("CC_MAX_CALLS", 40),
            wall_s=_env_int("CC_WALL_S", 1200),
            budgets={"compile": _env_int("CC_CHECKS_COMPILE", 20),
                     "correctness": _env_int("CC_CHECKS_CORRECTNESS", 10),
                     "performance": _env_int("CC_CHECKS_PERF", 4),
                     "counters": _env_int("CC_CHECKS_COUNTERS", 1)},
            counters=env.get("CC_COUNTERS", "1") != "0",
            healthy_floor=_env_int("CC_HEALTHY_FLOOR", 8),
            grade_concurrency=_env_int("CC_GRADE_CONCURRENCY", 64),
            episode_log=env.get("CC_EPISODE_LOG"),
        )


@dataclass
class Node:
    runner_url: str
    gateway_url: str
    capacity: int = 0
    in_flight: int = 0
    cooldown_until: float = 0.0


def read_nodes(path: str) -> list[Node]:
    nodes = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and not line.startswith("#"):
            nodes.append(Node(parts[0].rstrip("/"), parts[1].rstrip("/")))
    return nodes


class Breaker:
    """Stops new episodes while the Crusoe pool reports fewer healthy GPUs than the floor."""

    def __init__(self, client: GraderClient, floor: int, probe_s: float, clock=time.monotonic):
        self.client = client
        self.floor = floor
        self.probe_s = probe_s
        self.clock = clock
        self.checked_at = -1e18
        self.healthy: int | None = None
        self.open = False
        self.reason = ""
        self.trips = 0
        self._lock = asyncio.Lock()

    def _evaluate(self, result: Result) -> None:
        was_open = self.open
        if result.kind != "ok":
            self.open, self.reason, self.healthy = True, f"health {result.kind} {result.error}".strip(), None
        else:
            healthy = (result.body or {}).get("healthy")
            self.healthy = int(healthy) if isinstance(healthy, (int, float)) else None
            self.open = self.healthy is not None and self.healthy < self.floor
            self.reason = f"healthy {self.healthy} < floor {self.floor}" if self.open else ""
        if self.open and not was_open:
            self.trips += 1
            logger.warning("circuit breaker open: %s", self.reason)
        elif was_open and not self.open:
            logger.warning("circuit breaker closed: healthy %s", self.healthy)

    async def refresh(self, force: bool = False) -> None:
        async with self._lock:
            if not force and self.clock() - self.checked_at < self.probe_s:
                return
            result = await asyncio.to_thread(self.client.call, "GET", "/health")
            self.checked_at = self.clock()
            self._evaluate(result)

    async def wait_closed(self, sleep=asyncio.sleep) -> float:
        waited = 0.0
        await self.refresh()
        while self.open:
            await sleep(self.probe_s)
            waited += self.probe_s
            await self.refresh(force=True)
        return waited


class GradeQueue:
    """Final grades with backoff; ``None`` after the deadline means drop the episode."""

    def __init__(self, client: GraderClient, concurrency: int, backoff=GRADE_BACKOFF_S,
                 deadline_s: float = GRADE_DEADLINE_S, sleep=asyncio.sleep, clock=time.monotonic):
        self.client = client
        self.semaphore = asyncio.Semaphore(concurrency)
        self.backoff = backoff
        self.deadline_s = deadline_s
        self.sleep = sleep
        self.clock = clock

    async def grade(self, payload: dict) -> tuple[dict | None, list[dict]]:
        attempts: list[dict] = []
        start = self.clock()
        for attempt in range(len(self.backoff) + 1):
            async with self.semaphore:
                result = await asyncio.to_thread(self.client.call, "POST", "/grade", payload)
            attempts.append({"kind": result.kind, "status": result.status, "error": result.error,
                             "s": round(result.seconds, 2)})
            if result.kind == "ok" and isinstance((result.body or {}).get("score"), (int, float)):
                return result.body, attempts
            if result.kind == "rejected":
                logger.error("final grade rejected for %s: %s %s", payload.get("episode"), result.status,
                             str(result.body)[:300])
                return None, attempts
            if attempt == len(self.backoff):
                break
            delay = self.backoff[attempt]
            if self.clock() - start + delay > self.deadline_s:
                break
            await self.sleep(delay)
        return None, attempts


class RunnerPool:
    def __init__(self, nodes: list[Node], token: str | None, client: httpx.AsyncClient):
        self.nodes = nodes
        self.token = token
        self.client = client

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    async def _capacity(self, node: Node) -> None:
        if node.capacity:
            return
        try:
            response = await self.client.get(f"{node.runner_url}/health", headers=self.headers(), timeout=10)
            node.capacity = int(response.json().get("capacity", 0))
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("runner %s health failed: %s", node.runner_url, type(exc).__name__)
            node.cooldown_until = time.monotonic() + 30

    async def acquire(self, spec: dict, register=None, unregister=None, sleep=asyncio.sleep) -> Node:
        """Start ``spec`` on the least-loaded runner with room; waits while all are full.

        ``register(node)`` runs before the episode starts (the container calls the gateway at once);
        ``unregister(node)`` undoes it when that runner turns the episode down."""
        while True:
            now = time.monotonic()
            for node in self.nodes:
                if node.cooldown_until <= now:
                    await self._capacity(node)
            ready = [n for n in self.nodes if n.cooldown_until <= now and n.capacity and n.in_flight < n.capacity]
            random.shuffle(ready)
            ready.sort(key=lambda n: n.in_flight / n.capacity)
            for node in ready:
                node.in_flight += 1
                if register is not None:
                    try:
                        await register(node)
                    except BaseException:
                        node.in_flight -= 1
                        raise
                try:
                    body = {**spec, "gateway_url": node.gateway_url}
                    response = await self.client.post(f"{node.runner_url}/episodes", json=body,
                                                      headers=self.headers(), timeout=30)
                except httpx.HTTPError as exc:
                    response = None
                    node.cooldown_until = time.monotonic() + 30
                    logger.warning("runner %s unreachable: %s", node.runner_url, type(exc).__name__)
                if response is not None and response.status_code == 202:
                    return node
                node.in_flight -= 1
                if unregister is not None:
                    await unregister(node)
                if response is None:
                    continue
                if response.status_code == 503:
                    node.cooldown_until = time.monotonic() + 5
                    continue
                raise RuntimeError(f"runner {node.runner_url} refused episode: {response.status_code} "
                                   f"{response.text[:200]}")
            await sleep(5)

    def release(self, node: Node) -> None:
        node.in_flight = max(0, node.in_flight - 1)

    async def wait(self, node: Node, episode: str) -> dict:
        while True:
            response = await self.client.get(f"{node.runner_url}/episodes/{episode}",
                                             params={"wait": RUNNER_POLL_S, "forget": "1"},
                                             headers=self.headers(), timeout=RUNNER_POLL_S + 30)
            response.raise_for_status()
            body = response.json()
            if "result" in body:
                return body["result"]

    async def cancel(self, node: Node, episode: str) -> None:
        try:
            await self.client.delete(f"{node.runner_url}/episodes/{episode}", headers=self.headers(), timeout=10)
        except httpx.HTTPError:
            pass


class Agent:
    def __init__(self, config: Config, grader: GraderClient | None = None, client: httpx.AsyncClient | None = None):
        self.config = config
        self.client = client or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128))
        self.grader = grader or GraderClient(pool_size=config.grade_concurrency)
        self.runners = RunnerPool(read_nodes(config.nodes_file), _read_secret(config.runner_token_file), self.client)
        self.gateway_token = _read_secret(config.gateway_token_file)
        self.breaker = Breaker(self.grader, config.healthy_floor, config.breaker_probe_s)
        self.grades = GradeQueue(self.grader, config.grade_concurrency)
        self.dropped = 0
        self.in_flight: dict[str, Node] = {}

    async def cancel_all(self) -> int:
        pending = list(self.in_flight.items())
        await asyncio.gather(*(self.runners.cancel(node, episode) for episode, node in pending))
        return len(pending)

    def _gw_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.gateway_token}"} if self.gateway_token else {}

    def _log(self, record: dict) -> None:
        if self.config.episode_log:
            with open(self.config.episode_log, "a") as f:
                f.write(json.dumps(record) + "\n")

    def episode_spec(self, episode: str, token: str, label: dict) -> dict:
        return {
            "episode": episode, "task_id": label["task_id"], "task_version": label["task_version"],
            "workspace_dir": label["workspace_dir"], "gateway_token": token, "model": self.config.served_model,
            "split": label.get("split", "train"), "max_turns": self.config.max_calls, "wall_s": self.config.wall_s,
            "max_output_tokens": self.config.max_out, "budgets": self.config.budgets,
            "counters_enabled": self.config.counters,
        }

    async def run(self, base_url: str, label: dict) -> dict:
        started = time.time()
        waited = await self.breaker.wait_closed()
        task = label["task_id"].rsplit("/", 1)[-1][:40]
        episode = f"{task}-{secrets.token_hex(6)}"
        token = secrets.token_urlsafe(32)
        spec = self.episode_spec(episode, token, label)
        record: dict[str, Any] = {"episode": episode, "task_id": label["task_id"], "split": spec["split"],
                                  "breaker_wait_s": waited}
        node = None
        result: dict | None = None
        gateway: dict = {}
        try:
            registration = {"episode": episode, "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                            "session_url": base_url, "model": self.config.served_model,
                            "context": self.config.context, "max_out": self.config.max_out,
                            "stop_margin": self.config.stop_margin, "max_calls": self.config.max_calls,
                            "wall_s": float(self.config.wall_s)}

            async def register(target: Node) -> None:
                response = await self.client.post(f"{target.gateway_url}/episodes", json=registration,
                                                  headers=self._gw_headers(), timeout=30)
                if response.status_code != 200:
                    raise RuntimeError(f"gateway register failed: {response.status_code} {response.text[:200]}")

            async def unregister(target: Node) -> None:
                try:
                    await self.client.delete(f"{target.gateway_url}/episodes/{episode}", headers=self._gw_headers(),
                                             timeout=30)
                except httpx.HTTPError:
                    pass

            node = await self.runners.acquire(spec, register, unregister)
            self.in_flight[episode] = node
            record["node"] = urllib.parse.urlsplit(node.runner_url).hostname
            result = await self.runners.wait(node, episode)
        except asyncio.CancelledError:
            if node is not None:
                await asyncio.shield(self.runners.cancel(node, episode))
            raise
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"[:300]
            if node is not None:
                await self.runners.cancel(node, episode)
        finally:
            if node is not None:
                self.in_flight.pop(episode, None)
                self.runners.release(node)
                try:
                    response = await self.client.delete(f"{node.gateway_url}/episodes/{episode}",
                                                        headers=self._gw_headers(), timeout=30)
                    gateway = response.json() if response.status_code == 200 else {}
                except (httpx.HTTPError, ValueError):
                    gateway = {}
        metrics = (result or {}).get("metrics", {})
        record.update({"gateway": gateway, "runner": {k: metrics.get(k) for k in
                                                        ("reason", "exit", "wall_s", "checks", "gateway_requests",
                                                         "gateway_denials", "assistant_messages", "tool_calls")},
                       "log_dir": (result or {}).get("log_dir")})
        reason = metrics.get("reason")
        if result is None or reason in DROP_REASONS:
            return self._drop(record, f"runner:{reason or 'no_result'}", started)
        code = result.get("final_code") or ""
        grade, attempts = await self.grades.grade({"episode": episode, "task_id": label["task_id"],
                                                    "task_version": label["task_version"], "code": code})
        record["grade_attempts"] = attempts
        if grade is None:
            return self._drop(record, "final_grade_unavailable", started)
        score = float(grade["score"])
        record.update({"grade": {k: grade.get(k) for k in ("compiled", "correct", "audit", "method_consistent",
                                                           "speedup", "score")},
                       "reward": score, "wall_s": round(time.time() - started, 1)})
        self._log(record)
        checks = (metrics.get("checks") or {}).get("used", {})
        return {
            "reward": score,
            "hip_cc_episode": episode,
            "hip_cc_stop": gateway.get("stop_reason") or reason,
            "hip_cc_calls": gateway.get("calls"),
            "hip_cc_compiled": bool(grade.get("compiled")),
            "hip_cc_correct": bool(grade.get("correct")),
            "hip_cc_speedup": grade.get("speedup"),
            "hip_cc_checks": checks,
            "hip_cc_denials": metrics.get("gateway_denials", 0),
        }

    def _drop(self, record: dict, why: str, started: float) -> dict:
        self.dropped += 1
        record.update({"dropped": why, "wall_s": round(time.time() - started, 1)})
        logger.warning("dropping episode %s: %s", record["episode"], why)
        self._log(record)
        return {"reward": None, "hip_cc_episode": record["episode"], "hip_cc_dropped": why}


_AGENT: Agent | None = None


def get_agent() -> Agent:
    global _AGENT
    if _AGENT is None:
        _AGENT = Agent(Config.from_env())
    return _AGENT


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return {}


async def run(base_url: str, prompt: Any, request_kwargs: dict | None = None, metadata: dict | None = None,
              **kwargs: Any) -> dict:
    label = {**_as_dict(prompt), **(metadata or {})}
    return await get_agent().run(base_url, label)


async def generate(fn_input):
    """Merge the jsonl label into metadata, run the session-traced episode, then drop failed episodes."""
    from miles.rollout.generate_hub.agentic_tool_call import generate as agentic_generate
    from miles.utils.types import Sample

    sample = fn_input.sample
    sample.metadata = {**_as_dict(getattr(sample, "label", None)), **_as_dict(getattr(sample, "metadata", None))}
    output = await agentic_generate(fn_input)
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    for s in samples:
        meta = s.metadata or {}
        if meta.get("hip_cc_dropped") or s.reward is None:
            s.status = Sample.Status.ABORTED
    return output


def _add_arguments(parser):
    from miles.rollout.generate_hub.agentic_tool_call import generate as agentic_generate

    agentic_generate.add_arguments(parser)


generate.add_arguments = _add_arguments


def log_rollout_metrics(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """Print per-step episode stats; Miles' default logging still runs."""
    episodes: dict[str, dict] = {}
    for s in samples:
        meta = s.metadata or {}
        if "hip_cc_episode" in meta:
            episodes.setdefault(meta["hip_cc_episode"], meta)
    if episodes:
        metas = list(episodes.values())
        stops: dict[str, int] = {}
        for m in metas:
            stops[str(m.get("hip_cc_stop"))] = stops.get(str(m.get("hip_cc_stop")), 0) + 1
        n = len(metas)
        stats = {
            "episodes": n,
            "samples_per_episode": round(len(samples) / n, 2),
            "compiled": sum(bool(m.get("hip_cc_compiled")) for m in metas) / n,
            "correct": sum(bool(m.get("hip_cc_correct")) for m in metas) / n,
            "denials": sum(int(m.get("hip_cc_denials") or 0) for m in metas),
            "stop": {k: v / n for k, v in sorted(stops.items())},
        }
        if _AGENT is not None:
            stats.update({"dropped_total": _AGENT.dropped, "breaker_trips": _AGENT.breaker.trips,
                          "breaker_open": _AGENT.breaker.open})
        print(f"hip_cc {rollout_id}: {json.dumps(stats)}", flush=True)
    return False


async def abort(args) -> None:
    """Oversampling abort: stop every in-flight container; those episodes then drop."""
    if _AGENT is not None:
        cancelled = await _AGENT.cancel_all()
        logger.info("abort: cancelled %d in-flight episodes", cancelled)
