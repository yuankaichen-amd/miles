---
name: profiling-hip
description: >
  Decide what to optimize next on a HIP kernel for MI355X from measured evidence: the aka-eval
  compile resource report (registers, spills, LDS, occupancy), aka-eval performance (time per
  workload against the baseline) and aka-eval counters (hardware counters and a verdict), each tied
  to a HIP lever.
---

# Profiling HIP kernels here

There is no GPU or profiler in your container. Evidence comes from three commands; use them in this
order, and change one thing at a time between measurements.

## 1. `aka-eval compile` — free, local, every edit

The resource report lists each `__global__` kernel with VGPR, AGPR, SGPR, scratch bytes per lane,
spills, LDS bytes per workgroup, the compiler's occupancy (waves per SIMD) and `occ_lds` (workgroups
per CU that fit in 160 KiB of LDS). Read it before spending a GPU check:

| Report shows | Means | Lever |
|---|---|---|
| VGPR spills > 0 or scratch > 0 | registers spilled to memory (hundreds of cycles per access) | `../optimize/hip_levers/hip_traps.md`, `/kb/common_methodology/optimization/lever_occupancy.md` (`__launch_bounds__`, smaller per-thread tiles) |
| occupancy ≤ 2 waves/SIMD on a memory-bound kernel | too few waves to hide memory latency | cut VGPRs below the next 16-register step (`/kb/hardware/mi350_execution.md`) |
| `occ_lds` = 1 | LDS allows one workgroup per CU | shrink tiles or stage less (`../optimize/hip_levers/hip_lds_staging.md`) |
| AGPR > 0 with VGPR copies in the loop | MFMA accumulators bounce between AGPRs and VGPRs | `../optimize/hip_levers/hip_builtins.md` (stable accumulator variables) |

## 2. `aka-eval performance` — the number that is graded

It reports candidate and baseline milliseconds per workload and the geometric-mean speedup. Only a
geometric-mean speedup above 1.05× on the final file counts, so compare against the baseline per
workload:

- **One workload much slower than the others:** usually a shape your launch configuration handles
  badly (too few workgroups for a small shape, or a tail loop for an odd size). See
  `/kb/common_methodology/optimization/lever_grid_sizing.md`.
- **All workloads near 1.0×:** you're probably doing what PyTorch does. Estimate the bytes moved and
  the memory floor (`/kb/common_methodology/profiling/measure_roofline.md`); if the baseline is already
  near it, the remaining gain is fusion or fewer passes, not tuning.
- **Small differences between runs** are noise (per-XCD clock variance is 3–10%).

## 3. `aka-eval counters` — one per episode, after correctness passes

It returns the top candidate kernels by time, fabric bandwidth against the measured Infinity Cache and
HBM roofs, L2 hit rate, the instruction mix, LDS bank conflicts, grid fill and one verdict with the
card to read. `/kb/common_methodology/profiling/counters.md` explains each field. It may report
"unavailable"; then classify from sections 1 and 2 with
`/kb/common_methodology/profiling/measure_triage.md`.

## 4. From the verdict to a HIP lever

- **Bandwidth-bound** → vectorize and coalesce (`../optimize/hip_levers/hip_templates.md`), stage
  reused data in LDS (`../optimize/hip_levers/hip_lds_staging.md`), fuse passes.
- **Compute-bound** → MFMA shape and scheduling (`../optimize/hip_levers/hip_builtins.md`).
- **Occupancy-limited** → registers and LDS per workgroup (`../optimize/hip_levers/hip_traps.md`,
  `/kb/common_methodology/optimization/lever_occupancy.md`).
- **Latency-bound** → more independent work in flight per thread, prefetch the next tile
  (`/kb/common_methodology/optimization/lever_prefetch.md`).
- **LDS-bound** → re-derive padding or the XOR swizzle for 64 banks (`/kb/hardware/mi350_lds.md`).

A change that doesn't move the counter it targets, or the per-workload time, didn't work; revert it
rather than stacking more changes on top.
