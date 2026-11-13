"""t2h_manifest tests: the passed-all-runs filter and the eval128 top-up plan.

    python3 -m pytest -q examples/hip_cc/tests/test_t2h_manifest.py
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import t2h_manifest  # noqa: E402

VERSION = "v0"


def passed(task_id, split, source):
    return {"task_id": task_id, "split": split, "source": source, "status": "valid",
            "timing_nodes": ["a", "b", "c"],
            "baseline": {"w0": {"runs_ms": [1.0, 1.0, 1.0]}, "w1": {"runs_ms": [2.0, 2.0, 2.0]}}}


def test_passed_all_runs():
    good = passed("t", "train", "gpumode")
    assert t2h_manifest.passed_all_runs(good, 3)
    assert not t2h_manifest.passed_all_runs({**good, "status": "pending"}, 3)
    assert not t2h_manifest.passed_all_runs({**good, "timing_nodes": ["a", "b"]}, 3)
    short = {**good, "baseline": {"w0": {"runs_ms": [1.0, 1.0]}, "w1": {"runs_ms": [2.0, 2.0, 2.0]}}}
    assert not t2h_manifest.passed_all_runs(short, 3)
    assert not t2h_manifest.passed_all_runs({**good, "baseline": {}}, 3)
    assert not t2h_manifest.passed_all_runs(None, 3)


def make_store(tmp_path, tasks):
    """tasks: (task_id, split, source, difficulty, rank) -> tarball with t2h.json and a hip_ref decoy."""
    root = tmp_path / "t2h"
    (root / "store" / VERSION).mkdir(parents=True)
    index = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for task_id, split, source, difficulty, rank in tasks:
            path = f"{VERSION}/tasks/{split}/{source}/{task_id.rsplit('/', 1)[-1]}"
            meta = {"task_id": task_id, "split": split, "source": source, "difficulty": difficulty}
            if rank is not None:
                meta["eval128_topup_rank"] = rank
            for name, data in (("t2h.json", json.dumps(meta)), ("hip_ref.hip", "SECRET_HIP_REF")):
                raw = data.encode()
                info = tarfile.TarInfo(f"{path}/{name}")
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
            index.append({"task_id": task_id, "split": split, "source": source, "path": path})
    (root / f"t2h-{VERSION}.tar.gz").write_bytes(buffer.getvalue())
    (root / "store" / VERSION / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    return root


def test_topup_takes_rank_order_same_difficulty_and_pending_holds_the_slot(tmp_path, monkeypatch):
    tasks = [
        ("t2h/cudaagent/e1", "t2h-eval128", "cudaagent", "easy", None),
        ("t2h/cudaagent/e2", "t2h-eval128", "cudaagent", "hard", None),
        ("t2h/cudaagent/e3", "t2h-eval128", "cudaagent", "easy", None),
        ("t2h/cudaagent/r0", "reserve", "cudaagent", "easy", 0),   # dropped: skipped
        ("t2h/cudaagent/r1", "reserve", "cudaagent", "easy", 1),   # pending: holds e1's slot
        ("t2h/cudaagent/r2", "reserve", "cudaagent", "easy", 2),   # passed but ranked after r1
        ("t2h/cudaagent/r3", "reserve", "cudaagent", "hard", 3),   # passed, same difficulty as e2
        ("t2h/cudaagent/r9", "reserve", "cudaagent", "hard", None),  # not in the ranked pool
        ("t2h/gpumode/g1", "t2h-eval128", "gpumode", "easy", None),
    ]
    root = make_store(tmp_path, tasks)
    monkeypatch.setattr(t2h_manifest, "DATA", tmp_path)
    monkeypatch.setattr(t2h_manifest, "TOPUP_PLAN", root / "eval128-topup-latest.json")
    manifest = {"version": VERSION, "generated_at": "T1", "rules": {"runs": 3}, "tasks": [
        {"task_id": "t2h/cudaagent/e1", "status": "dropped"},
        {"task_id": "t2h/cudaagent/e2", "status": "dropped"},
        passed("t2h/cudaagent/e3", "t2h-eval128", "cudaagent"),
        {"task_id": "t2h/cudaagent/r0", "status": "dropped"},
        {"task_id": "t2h/cudaagent/r1", "status": "pending"},
        passed("t2h/cudaagent/r2", "reserve", "cudaagent"),
        passed("t2h/cudaagent/r3", "reserve", "cudaagent"),
        passed("t2h/cudaagent/r9", "reserve", "cudaagent"),
        {"task_id": "t2h/gpumode/g1", "status": "pending"},
    ]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    plan = t2h_manifest.topup(path, VERSION)
    slots = {s["replaces"]: s for s in plan["slots"]}
    assert slots["t2h/cudaagent/e1"]["task_id"] == "t2h/cudaagent/r1"
    assert slots["t2h/cudaagent/e1"]["state"] == "pending"
    assert slots["t2h/cudaagent/e2"]["task_id"] == "t2h/cudaagent/r3"
    assert slots["t2h/cudaagent/e2"]["state"] == "passed" and slots["t2h/cudaagent/e2"]["same_difficulty"]
    assert plan["eval128_pending"] == 1 and plan["complete"] is False
    assert "SECRET_HIP_REF" not in t2h_manifest.TOPUP_PLAN.read_text()

    # r1 drops later: the slot moves to the next ranked easy candidate, r2.
    manifest["tasks"][4] = {"task_id": "t2h/cudaagent/r1", "status": "dropped"}
    manifest["tasks"][8] = passed("t2h/gpumode/g1", "t2h-eval128", "gpumode")
    manifest["generated_at"] = "T2"
    path.write_text(json.dumps(manifest))
    plan = t2h_manifest.topup(path, VERSION)
    slots = {s["replaces"]: s for s in plan["slots"]}
    assert slots["t2h/cudaagent/e1"]["task_id"] == "t2h/cudaagent/r2"
    assert plan["complete"] is True
