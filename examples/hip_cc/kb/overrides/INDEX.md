---
title: /kb — HIP on MI355X (gfx950) reference notes for this task
kind: index
scope: kb
gens: [gfx950]
---

# /kb — knowledge map

Read-only reference cards for writing and tuning a HIP kernel on **MI355X (gfx950, CDNA4)**. Open a
card with `Read` only when it is relevant; nothing here is required reading, and nothing here is
graded. The task prompt already carries the short MI355X and HIP cheat sheets; these cards are the
depth behind them.

## Your tools, not the cards' tools

You have no GPU and no profiler in this container. The only measurements are:

| Command | What it gives you | Card |
|---|---|---|
| `aka-eval compile` | local gfx950 build, the source audit, and per-kernel VGPR/AGPR/SGPR, scratch, spills, LDS bytes, compiler occupancy and `occ_lds` | `common_methodology/profiling/counters.md` §1 |
| `aka-eval correctness` | the MI355X correctness verdict against the PyTorch reference | — |
| `aka-eval performance` | MI355X time per workload against the PyTorch baseline | `common_methodology/profiling/measure_protocol.md` |
| `aka-eval counters` | MI355X hardware counters and one triage verdict (after correctness passes; may be unavailable) | `common_methodology/profiling/counters.md` |

Where a card says `rocprof-compute`, `rocprofv3`, `omniperf`, `rocm-smi`, `--save-temps`, a forge loop
or a driver script, read it as background: run `aka-eval compile` or `aka-eval counters` instead.
Paths written as `local_knowledge/...` mean `/kb/...`. Some cards link to cards that are not in this
`/kb` (`lever_autotune`, `lever_loop_form`, `lever_cheap_sweeps`, `lever_edit_surface`,
`measure_rocpc_workflow`, `rocpc_profile.py`, `hipkittens`, `framework/`, other languages); skip those
links. The build is fixed (`torch.utils.cpp_extension`, `-O3`, gfx950): compiler flags in a card are
not available to you.

Host-side snippets in the cards (`hipMalloc`/`hipFree`, blocking `hipMemcpy`/`hipMemset`,
`hip*Synchronize`, `hipStreamCreate`, HIP graphs, `hipHostMalloc`) are background too: the source
audit rejects them in `kernel.hip`. Allocate outputs as torch tensors in `run`, launch on the current
stream, and use only the `*Async` copy and memset forms if you need them.

## Layout

```
/kb/
├── INDEX.md                         ← this map
├── hardware/INDEX.md                # MI355X numbers: execution, LDS, memory, chiplet, MFMA, dtypes, ISA, clocks
├── common_methodology/INDEX.md      # classify the bottleneck, then pick one lever
└── languages/hip/INDEX.md           # HIP authoring, builtins, LDS staging, templates, CUDA→HIP traps, debugging
```

## Start here

| Question | Read |
|---|---|
| "What are the MI355X numbers?" | `hardware/mi350_overview.md` |
| "How many waves will this kernel get?" | `aka-eval compile`, then `hardware/mi350_execution.md` |
| "Is it bandwidth- or compute-bound, and what do I change?" | `common_methodology/INDEX.md` → `common_methodology/profiling/counters.md` |
| "LDS bank conflicts on 64 banks" | `hardware/mi350_lds.md` → `languages/hip/skills/optimize/hip_levers/hip_lds_staging.md` |
| "Kernel is wrong / slow after a CUDA-style port" | `languages/hip/skills/bottleneck/debug-hip-kernel.md` → `languages/hip/skills/optimize/hip_levers/hip_traps.md` |
| "Reduction / grid-stride / tiled skeletons" | `languages/hip/skills/optimize/hip_levers/hip_templates.md` |
| "MFMA intrinsics" | `hardware/mi350_matrix_core.md` → `languages/hip/skills/optimize/hip_levers/hip_builtins.md` |

## Source and license

Cards are from AMD-AGI/Hyperloom `src/kernelforge/knowledge/local_wiki/resources` at
`8fc5514ae06dbd320d64d3c2524c379b8793d86e`, MIT License (`/kb/LICENSE`). This file, the two folder
INDEX maps, `languages/hip/skills/profile/profiling-hip.md` and
`common_methodology/profiling/counters.md` are rewritten for this task; the other cards are unchanged.
