"""Build the read-only /kb for hip-cc-rollout from the Hyperloom cards (plan §10 P5, Hyperloom study §2.3).

    python3 build_kb.py [--hyperloom <clone>] [--out <dir>]

Copies the kept cards unchanged from Hyperloom ``8fc5514a``, drops the 7 that need the forge loop, a
GPU or headers we don't ship, adds the rewritten maps and cards from ``overrides/`` and the MIT
LICENSE, and writes ``<out>/<sha16>/`` plus a ``current`` symlink. The sha covers every file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMMIT = "8fc5514ae06dbd320d64d3c2524c379b8793d86e"
RESOURCES = "src/kernelforge/knowledge/local_wiki/resources"
KEEP = [
    *(f"hardware/{n}" for n in ("INDEX.md", "mi350_overview.md", "mi350_execution.md", "mi350_matrix_core.md",
                                "mi350_dtypes.md", "mi350_lds.md", "mi350_memory.md", "mi350_chiplet.md",
                                "mi350_isa.md", "mi350_clocks.md")),
    "common_methodology/optimization/LEVERS.md",
    *(f"common_methodology/optimization/lever_{n}.md" for n in (
        "bottleneck_class", "coalescing", "fusion", "grid_sizing", "lds_banks", "mfma_sched", "numerics",
        "occupancy", "prefetch", "xcd_locality")),
    *(f"common_methodology/profiling/measure_{n}.md" for n in ("protocol", "roofline", "triage")),
    *(f"languages/hip/API_docs/{n}.md" for n in ("compilation_and_build", "dtypes", "kernel_language",
                                                  "runtime_api")),
    "languages/hip/skills/bottleneck/debug-hip-kernel.md",
    *(f"languages/hip/skills/optimize/hip_levers/hip_{n}.md" for n in (
        "authoring_model", "builtins", "lds_staging", "templates", "traps")),
]
DROPPED = [
    "common_methodology/optimization/lever_edit_surface.md",
    "common_methodology/optimization/lever_cheap_sweeps.md",
    "common_methodology/optimization/lever_autotune.md",
    "common_methodology/optimization/lever_loop_form.md",
    "common_methodology/profiling/measure_rocpc_workflow.md",
    "common_methodology/profiling/rocpc_profile.py",
    "languages/hip/skills/optimize/hip_levers/hipkittens.md",
]
REWRITTEN = [
    "INDEX.md",
    "common_methodology/INDEX.md",
    "languages/hip/INDEX.md",
    "languages/hip/skills/profile/profiling-hip.md",
    "common_methodology/profiling/counters.md",
]


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperloom", default="/shared_inference/yuankach/repos/Hyperloom")
    parser.add_argument("--out", default="/shared_inference/yuankach/glm53/hip-cc/kb")
    args = parser.parse_args()
    clone = Path(args.hyperloom)
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        sys.exit(f"Hyperloom is at {head}, expected {COMMIT}")
    source = clone / RESOURCES
    out = Path(args.out)
    staging = out / ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    for relative in KEEP:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    for relative in REWRITTEN:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "overrides" / relative, target)
    shutil.copyfile(clone / "LICENSE", staging / "LICENSE")
    for relative in DROPPED:
        if (staging / relative).exists():
            sys.exit(f"dropped card {relative} ended up in /kb")
    sha = tree_sha(staging)[:16]
    final = out / sha
    if final.exists():
        shutil.rmtree(staging)
    else:
        staging.rename(final)
    for path in final.rglob("*"):
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    link = out / "current"
    tmp_link = out / ".current.tmp"
    if tmp_link.is_symlink():
        tmp_link.unlink()
    tmp_link.symlink_to(sha)
    tmp_link.replace(link)
    files = sorted(p for p in final.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files)
    print(f"/kb {sha}: {len(files)} files, {size:,} bytes -> {final} (current)")


if __name__ == "__main__":
    main()
