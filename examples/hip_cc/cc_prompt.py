"""The Claude Code task prompt for one rendered workspace (plan §1 "Workspace and prompt", Hyperloom study §2.3).

AKA's structure (task, workspace, editable file, instructions, checks, hardware context) with this
harness's check commands, AKA's two gfx950 cheat sheets inline (numbers reconciled with /kb,
self-profiling replaced by ``aka-eval counters``) and a one-line pointer to /kb.
"""

from __future__ import annotations

import re
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
CHEATSHEETS = ("MI355X_architecture.md", "hip_cheatsheet.md")


def hardware_context(with_kb: bool = True) -> str:
    parts = [re.sub(r"^<!--.*?-->\n", "", (PROMPT_DIR / name).read_text(), flags=re.S) for name in CHEATSHEETS]
    text = "\n\n---\n\n".join(parts)
    if with_kb:
        text += ("\n\n---\n\nDeeper reference cards on MI355X hardware, bottleneck triage, optimization levers and HIP "
                 "are in /kb (read-only). Start at /kb/INDEX.md and open a card only when it is relevant.\n")
    return text


def build_prompt(workspace: Path, *, with_kb: bool = True) -> str:
    readme = (workspace / "README.md").read_text()
    task_id = readme.splitlines()[0].removeprefix("# ").strip()
    lines = [
        f"Task: {task_id}",
        "Workspace: /workspace",
        "Target GPU: AMD Instinct MI355X (gfx950). Your container has no GPU; checks run on MI355X remotely.",
        "Required final implementation backend: hip",
        "",
        "Editable implementation:",
        "- solution/kernel.hip: implementation file (the only file that is graded)",
        "",
        "Task instructions from README.md:",
        readme,
        "Candidate checks (run from /workspace):",
        "- aka-eval compile: local gfx950 build, source audit and per-kernel register/LDS/occupancy report "
        "(up to 20 per episode)",
        "- aka-eval correctness: MI355X correctness against the PyTorch reference (up to 10; the same code twice "
        "is not charged)",
        "- aka-eval performance: MI355X time per workload against the PyTorch baseline (up to 4)",
        "- aka-eval counters: MI355X hardware counters, once correctness passes on that code (1; may be unavailable)",
        "- aka-eval status: remaining budgets",
        "A check reported as unavailable is not counted. Re-run compile and correctness after changing the kernel.",
        "",
        "Keep reference/, _aka_eval/ and config.yaml unchanged. Do not call PyTorch, rocBLAS or hipBLAS from the "
        "kernel, memoize outputs, or special-case inputs. The final solution/kernel.hip is compiled, checked for "
        "correctness and timed against the PyTorch baseline on MI355X after you finish; only a geometric-mean "
        "speedup above 1.05x earns reward.",
        "",
        "Hardware and implementation reference:",
        hardware_context(with_kb),
    ]
    return "\n".join(lines)
