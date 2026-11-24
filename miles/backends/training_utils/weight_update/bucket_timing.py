"""Opt-in per-batch weight-sync timing, enabled with MILES_WEIGHT_UPDATE_BUCKET_TIMING=1.

Every rank times its own phases of each gather batch (load, PP broadcast, ETP gather, EP name
exchange, EP gather); the sender also times its NCCL broadcast to the engines and the wait for the
engines' HTTP acks after that broadcast completes (engine-side load). Each timed phase ends with
torch.cuda.synchronize() so asynchronous collectives are charged to the phase that issued them,
which serializes the sync slightly: leave it off outside diagnosis.

Each rank logs a one-line summary per sync; with MILES_WEIGHT_UPDATE_BUCKET_TIMING_DIR set it also
writes sync<version>-rank<rank>.jsonl with one record per batch. Recording never raises.
"""

import json
import logging
import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MILES_WEIGHT_UPDATE_BUCKET_TIMING") == "1"
_OUT_DIR = os.environ.get("MILES_WEIGHT_UPDATE_BUCKET_TIMING_DIR") or None
_TOP_N = 5


class _Recorder:
    def __init__(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._version: int | None = None
        self._t0 = 0.0
        self._batches: list[dict] = []

    @property
    def active(self) -> bool:
        return ENABLED and self._version is not None

    def begin_sync(self, version: int) -> None:
        if not ENABLED:
            return
        self._reset()
        self._version = version
        self._t0 = time.perf_counter()

    def next_batch(self, kind: str, nbytes: int) -> None:
        if self.active:
            self._batches.append({"i": len(self._batches), "kind": kind, "bytes": int(nbytes)})

    @contextmanager
    def phase(self, name: str):
        if not self.active:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            try:
                torch.cuda.synchronize()
                if not self._batches:
                    self._batches.append({"i": 0, "kind": "pre", "bytes": 0})
                batch = self._batches[-1]
                batch[name] = batch.get(name, 0.0) + time.perf_counter() - start
            except Exception:
                logger.warning("bucket_timing: phase %s not recorded", name, exc_info=True)

    def end_sync(self) -> None:
        if not self.active:
            return
        try:
            self._emit()
        except Exception:
            logger.warning("bucket_timing: summary not written", exc_info=True)
        finally:
            self._reset()

    def _emit(self) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        wall = time.perf_counter() - self._t0
        totals: dict[str, float] = {}
        for batch in self._batches:
            for key, value in batch.items():
                if isinstance(value, float):
                    totals[key] = totals.get(key, 0.0) + value
        slowest = sorted(
            self._batches, key=lambda b: sum(v for v in b.values() if isinstance(v, float)), reverse=True
        )[:_TOP_N]
        summary = {
            "version": self._version,
            "rank": rank,
            "batches": len(self._batches),
            "gbytes": round(sum(b["bytes"] for b in self._batches) / 2**30, 2),
            "wall_s": round(wall, 2),
            "phase_s": {k: round(v, 2) for k, v in sorted(totals.items())},
            "slowest": [{k: round(v, 3) if isinstance(v, float) else v for k, v in b.items()} for b in slowest],
        }
        logger.info("bucket_timing %s", json.dumps(summary, separators=(",", ":")))
        if _OUT_DIR:
            os.makedirs(_OUT_DIR, exist_ok=True)
            path = os.path.join(_OUT_DIR, f"sync{self._version:04d}-rank{rank:03d}.jsonl")
            with open(path, "w") as f:
                for batch in self._batches:
                    f.write(json.dumps(batch, separators=(",", ":")) + "\n")
                f.write(json.dumps({"summary": summary}, separators=(",", ":")) + "\n")


recorder = _Recorder()
