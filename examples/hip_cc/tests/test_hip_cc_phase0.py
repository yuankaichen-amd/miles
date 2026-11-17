"""Phase 0 task gate: 0a runs train tasks only; 0b also runs the three eval sets; reserve never runs.

Both copies of the allowlist are tested: ``hip_cc_phase0.load_rows`` and the sbatch's host-side gate.

    python3 -m pytest -q examples/hip_cc/tests/test_hip_cc_phase0.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import hip_cc_phase0  # noqa: E402

EVAL_SPLITS = ("t2h-eval128", "heldout150-t2h", "kernelbench-t2h")
SBATCH = HERE.parent / "scripts" / "run_glm53_hip_cc_phase0.sbatch"


def write_tasks(path: Path, splits) -> Path:
    path.write_text("".join(
        json.dumps({"prompt": f"t2h/x/{split}_{i}", "label": {"task_id": f"t2h/x/{split}_{i}", "split": split}}) + "\n"
        for i, split in enumerate(splits)))
    return path


def sbatch_gate() -> str:
    text = SBATCH.read_text()
    match = re.search(r'python3 - "\$\{CC_TASKS\}" "\$\{CC_STEP\}" <<\'EOF\'\n(.*?)\nEOF\n', text, re.S)
    assert match, "task gate heredoc not found in the sbatch"
    return match.group(1)


def run_gate(tasks: Path, step: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-", str(tasks), step], input=sbatch_gate(), text=True,
                          capture_output=True, check=False)


def test_0a_runs_train(tmp_path):
    tasks = write_tasks(tmp_path / "t.jsonl", ["train", "train"])
    assert len(hip_cc_phase0.load_rows(str(tasks), 0)) == 2
    assert len(hip_cc_phase0.load_rows(str(tasks), 1, "0a")) == 1
    assert run_gate(tasks, "0a").returncode == 0


@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_0a_refuses_eval_sets(tmp_path, split):
    tasks = write_tasks(tmp_path / "t.jsonl", ["train", split])
    with pytest.raises(SystemExit, match=split):
        hip_cc_phase0.load_rows(str(tasks), 0)
    with pytest.raises(SystemExit, match=split):
        hip_cc_phase0.load_rows(str(tasks), 0, "0a")
    assert run_gate(tasks, "0a").returncode != 0


@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_0b_runs_each_eval_set(tmp_path, split):
    tasks = write_tasks(tmp_path / "t.jsonl", [split] * 3)
    rows = hip_cc_phase0.load_rows(str(tasks), 0, "0b")
    assert [row["split"] for row in rows] == [split] * 3
    assert run_gate(tasks, "0b").returncode == 0


def test_0b_runs_train_and_eval_together(tmp_path):
    tasks = write_tasks(tmp_path / "t.jsonl", ["train", *EVAL_SPLITS])
    assert len(hip_cc_phase0.load_rows(str(tasks), 0, "0b")) == 4
    assert run_gate(tasks, "0b").returncode == 0


@pytest.mark.parametrize("step", ["0a", "0b"])
def test_reserve_never_runs(tmp_path, step):
    tasks = write_tasks(tmp_path / "t.jsonl", ["train", "reserve"])
    with pytest.raises(SystemExit, match="reserve"):
        hip_cc_phase0.load_rows(str(tasks), 0, step)
    assert run_gate(tasks, step).returncode != 0


def test_unknown_step_refused(tmp_path):
    tasks = write_tasks(tmp_path / "t.jsonl", ["train"])
    with pytest.raises(SystemExit, match="unknown Phase 0 step"):
        hip_cc_phase0.load_rows(str(tasks), 0, "1")
    assert run_gate(tasks, "1").returncode != 0


def test_empty_task_list_refused_by_the_gate(tmp_path):
    tasks = tmp_path / "t.jsonl"
    tasks.write_text("")
    assert run_gate(tasks, "0b").returncode != 0


def test_gate_allowlist_matches_the_driver():
    allowed = re.search(r"allowed = (\{.*?\})\.get", sbatch_gate()).group(1)
    gate = {step: set(splits) for step, splits in eval(allowed).items()}  # noqa: S307
    assert gate == {step: set(splits) for step, splits in hip_cc_phase0.PHASE0_SPLITS.items()}
