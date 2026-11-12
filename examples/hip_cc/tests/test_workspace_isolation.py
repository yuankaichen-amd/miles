"""Agents see only rendered workspaces: no hip_ref content or path may reach a rollout container.

Host-side part (this file, run in the Miles image): renders workspaces from the real task
store tarball, which in task_version 6e0402dbba69efa9 still carries a top-level ``hip_ref/``
directory, and checks the workspace files, the tarball reads and the docker-argument guard.
With ``ISOLATION_OUT`` set it also writes the rendered workspaces plus ``needles.json``
(sha256 of every distinctive hip_ref line, never the text) for the container part,
``run_container_isolation.py``, which needs docker and so runs on the host.

    PYTHONPATH=<miles> python3 -m pytest -q examples/hip_cc/tests/test_workspace_isolation.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import cc_workspace  # noqa: E402

DATA = Path("/shared_inference/yuankach/glm53/datasets/hip")
VERSION = os.environ.get("T2H_VERSION", "6e0402dbba69efa9")
PACKAGE = DATA / "t2h" / f"t2h-{VERSION}.tar.gz"
FEASIBILITY = DATA / "t2h" / f"t2h-{VERSION}-feasibility.tar.gz"
SPLITS = ("train", "t2h-eval128", "heldout150-t2h", "reserve", "kernelbench-t2h")
PER_SPLIT = int(os.environ.get("ISOLATION_PER_SPLIT", "4"))

pytestmark = pytest.mark.skipif(not PACKAGE.exists(), reason=f"{PACKAGE} not present")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def distinctive_lines(text: str) -> set[str]:
    return {line.strip() for line in text.splitlines() if len(line.strip()) >= 24}


@pytest.fixture(scope="module")
def chosen():
    index = [json.loads(line) for line in (DATA / "t2h/store" / VERSION / "index.jsonl").read_text().splitlines()]
    tasks = []
    for split in SPLITS:
        tasks += [row for row in index if row["split"] == split][:PER_SPLIT]
    return tasks


@pytest.fixture(scope="module")
def hip_ref_corpus():
    """Every hip_ref in the package, read by the test (never by rollout code) from its hip_ref/."""
    corpus = {}
    with tarfile.open(PACKAGE) as tar:
        for member in tar:
            if member.isfile() and member.name.startswith(f"{VERSION}/hip_ref/"):
                corpus[Path(member.name).stem.replace("__", "/")] = tar.extractfile(member).read().decode()
    return corpus


@pytest.fixture(scope="module")
def hip_refs(chosen, hip_ref_corpus):
    return {t["task_id"]: hip_ref_corpus[t["task_id"]] for t in chosen if t["task_id"] in hip_ref_corpus}


@pytest.fixture(scope="module")
def needles(hip_refs, hip_ref_corpus):
    """sha256 of lines specific to the chosen tasks' hip_refs. Lines found in more than one hip_ref
    (``#include <hip/hip_runtime.h>``, pybind boilerplate) are public HIP idiom, not leaked content."""
    counts = Counter(line for text in hip_ref_corpus.values() for line in distinctive_lines(text))
    return {sha(line) for text in hip_refs.values() for line in distinctive_lines(text) if counts[line] == 1}


@pytest.fixture(scope="module")
def rendered(chosen, tmp_path_factory):
    out = Path(os.environ["ISOLATION_OUT"]) if os.environ.get("ISOLATION_OUT") else tmp_path_factory.mktemp("ws")
    workspaces = {}
    for task in chosen:
        dest = out / "workspaces" / task["task_id"].replace("/", "__")
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        cc_workspace.render_workspace(PACKAGE, task["task_id"], dest, task_version=VERSION, split=task["split"])
        workspaces[task["task_id"]] = dest
    return out, workspaces


def test_package_still_has_hip_ref_so_the_check_is_meaningful(hip_refs, chosen):
    assert len(hip_refs) >= len([t for t in chosen if t["split"] != "kernelbench-t2h"]) > 0


def test_workspace_contains_only_the_allowed_files(rendered):
    _, workspaces = rendered
    for workspace in workspaces.values():
        files = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file())
        assert files == sorted(cc_workspace.WORKSPACE_FILES)
        assert not any(cc_workspace.FORBIDDEN_PATH.search(f) for f in files)


def test_needles_cover_every_chosen_hip_ref(hip_refs, needles):
    for task_id, text in hip_refs.items():
        assert {sha(line) for line in distinctive_lines(text)} & needles, task_id


def leaks(text: str, needles: set, whole: set, row_hashes: set) -> list[str]:
    found = []
    if sha(text) in whole:
        found.append("whole_file")
    if {sha(line.strip()) for line in text.splitlines()} & needles:
        found.append("hip_ref_line")
    if any(h in text for h in row_hashes) or "hip_ref" in text or str(DATA) in text:
        found.append("marker")
    return found


def test_leak_check_catches_a_planted_hip_ref(hip_refs, needles):
    text = next(iter(hip_refs.values()))
    assert leaks(text, needles, {sha(text)}, set()) == ["whole_file", "hip_ref_line"]
    one_line = next(line for line in distinctive_lines(text) if sha(line) in needles)
    assert leaks(f"// stub\n  {one_line}\n", needles, set(), set()) == ["hip_ref_line"]


def test_no_hip_ref_content_in_any_workspace(rendered, hip_refs, chosen, needles):
    _, workspaces = rendered
    whole = {sha(text) for text in hip_refs.values()}
    row_hashes = {t["row_sha256"] for t in chosen}
    for task_id, workspace in workspaces.items():
        for path in workspace.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                assert not leaks(text, needles, whole, row_hashes), (task_id, path.relative_to(workspace).as_posix())


def test_contract_summary_has_no_seeds_or_baseline(rendered):
    _, workspaces = rendered
    for workspace in workspaces.values():
        contract = json.loads((workspace / "_aka_eval/contract.json").read_text())
        assert "baseline" not in contract and all("seeds" not in c for c in contract["cases"])


def test_extract_reads_only_allowlisted_members(chosen, tmp_path, monkeypatch):
    opened = []
    original = tarfile.TarFile.extractfile

    def spy(self, member):
        opened.append(member if isinstance(member, str) else member.name)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", spy)
    task = chosen[0]
    cc_workspace.extract_task(PACKAGE, task["task_id"], tmp_path / "t")
    assert len(opened) == 3 and all(Path(n).name in cc_workspace.PACKAGE_FILES for n in opened)
    assert not any("hip_ref" in n or "t2h.json" in n or "/baseline/" in n for n in opened)


def test_forbidden_paths_are_refused():
    for path in (FEASIBILITY, DATA / "t2h" / "hip_ref", f"{VERSION}/hip_ref/t2h__gpumode__x.hip"):
        with pytest.raises(cc_workspace.IsolationError):
            cc_workspace.check_path_allowed(path)


@pytest.mark.parametrize("bad", [
    ["-v", "/shared_inference/yuankach/glm53/datasets/hip/t2h:/data:ro"],
    ["-v", "/home/yuankach:/h"],
    ["--volume=/shared_inference:/s"],
    ["--mount", "type=bind,source=/home/yuankach/hip-cc,target=/x"],
    ["-v", "hip_ref_cache:/x"],
    ["--privileged"],
    ["-e", "TASKS=/shared_inference/yuankach/glm53/datasets/hip/t2h"],
])
def test_docker_guard_rejects_host_paths(bad):
    with pytest.raises(cc_workspace.IsolationError):
        cc_workspace.check_docker_args(["docker", "run", *bad, "img"])


def test_container_args_pass_the_guard():
    args = cc_workspace.container_args("c", "img", 1000, 1000)
    assert "--read-only" in args and "--network" in args and not any(a in ("-v", "--volume", "--mount") for a in args)


def test_write_needles_for_container_check(rendered, hip_refs, chosen, needles):
    out, workspaces = rendered
    if not os.environ.get("ISOLATION_OUT"):
        pytest.skip("container part not requested")
    (out / "needles.json").write_text(json.dumps({
        "line_sha256": sorted(needles), "whole_sha256": sorted(sha(t) for t in hip_refs.values()),
        "row_sha256": sorted(t["row_sha256"] for t in chosen),
        "workspaces": {k: str(v) for k, v in workspaces.items()},
        "planted": str(plant_control(out, workspaces, hip_refs)),
    }))


def plant_control(out: Path, workspaces: dict, hip_refs: dict) -> Path:
    """A copy of one workspace with its hip_ref as kernel.hip: the container scan must flag it.
    run_container_isolation.py deletes it after the scan."""
    import shutil

    task_id, text = next(iter(hip_refs.items()))
    planted = out / "planted"
    if planted.exists():
        shutil.rmtree(planted)
    shutil.copytree(workspaces[task_id], planted)
    (planted / "solution/kernel.hip").write_text(text)
    return planted
