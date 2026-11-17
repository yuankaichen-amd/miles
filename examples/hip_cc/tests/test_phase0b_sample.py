"""Phase 0b sample: stratum allocation and the four ablation cells.

    python3 -m pytest -q examples/hip_cc/tests/test_phase0b_sample.py
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "phase0b")]

import sample_tasks  # noqa: E402

KEYS = ("source", "difficulty")


def make_tasks():
    sizes = {("cudaagent", "L1"): 600, ("cudaagent", "L2"): 400, ("cudaagent", "L3"): 171,
             ("gpumode", "L1"): 1500, ("gpumode", "L2"): 600, ("gpumode", "L3"): 90}
    tasks = []
    for (source, level), size in sizes.items():
        for i in range(size):
            task_id = f"t2h/{source}/{level}_{i}"
            tasks.append({"task_id": task_id, "source": source, "difficulty": level,
                          "gfx950": "known" if i % 5 else "none",
                          "row": {"prompt": task_id, "label": {"task_id": task_id, "split": "train", "source": source}}})
    return tasks


def test_proportional_allocation_sums_and_tracks_sizes():
    sizes = {"a": 1171, "b": 2190}
    out = sample_tasks.allocate(sizes, 512, "proportional")
    assert sum(out.values()) == 512
    assert out == {"a": 178, "b": 334}


def test_equal_allocation_water_fills_small_strata():
    out = sample_tasks.allocate({"a": 10, "b": 1000, "c": 1000}, 300, "equal")
    assert out == {"a": 10, "b": 145, "c": 145}
    assert sample_tasks.allocate({"a": 100, "b": 100}, 6, "equal") == {"a": 3, "b": 3}


def test_allocation_refuses_more_than_eligible():
    with pytest.raises(ValueError):
        sample_tasks.allocate({"a": 3}, 4, "proportional")
    with pytest.raises(ValueError):
        sample_tasks.allocate({"a": 3}, 2, "random")


@pytest.mark.parametrize("allocation", sample_tasks.ALLOCATIONS)
def test_sample_cells_are_balanced(allocation):
    result = sample_tasks.sample(make_tasks(), 512, KEYS, allocation, "any", seed=1)
    picks = result["picks"]
    assert len(picks) == len({p["task_id"] for p in picks}) == 512
    cells = collections.Counter(sample_tasks.cell_name(p["cell"]) for p in picks)
    assert sorted(cells.values()) == [128] * 4
    assert {(c["counters"], c["kb"]) for c in sample_tasks.CELLS} == {(a, b) for a in (True, False) for b in (True, False)}
    for counts in result["record"]["cells_by_stratum"].values():
        assert max(counts.values()) - min(counts.values()) <= 1
    picked = {s: v["picked"] for s, v in result["record"]["strata"].items()}
    assert sum(picked.values()) == 512


def test_sample_is_seeded():
    a = sample_tasks.sample(make_tasks(), 512, KEYS, "proportional", "any", seed=7)["picks"]
    b = sample_tasks.sample(make_tasks(), 512, KEYS, "proportional", "any", seed=7)["picks"]
    c = sample_tasks.sample(make_tasks(), 512, KEYS, "proportional", "any", seed=8)["picks"]
    assert [(p["task_id"], p["cell"]) for p in a] == [(p["task_id"], p["cell"]) for p in b]
    assert {p["task_id"] for p in a} != {p["task_id"] for p in c}


def test_sample_excludes_and_filters():
    tasks = make_tasks()
    excluded = {t["task_id"] for t in tasks[:50]}
    result = sample_tasks.sample(tasks, 512, KEYS, "proportional", "known", seed=1, exclude=excluded)
    assert not excluded & {p["task_id"] for p in result["picks"]}
    assert all(p["gfx950"] == "known" for p in result["picks"])
    assert result["record"]["excluded"] == sorted(excluded)


def test_source_only_strata():
    result = sample_tasks.sample(make_tasks(), 512, ("source",), "proportional", "any", seed=1)
    assert set(result["record"]["strata"]) == {"cudaagent", "gpumode"}
