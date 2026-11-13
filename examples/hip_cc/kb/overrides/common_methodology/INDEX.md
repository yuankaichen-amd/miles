---
title: Common optimization methodology — index and problem routing
kind: index
scope: common_methodology
gens: [gfx950]
---

# Common optimization methodology — knowledge map

Entry index for `common_methodology/`: how to find the real bottleneck of a slow kernel and which
general lever fixes it, independent of the kernel language. It sits above `hardware/` (the numbers)
and below `languages/hip/` (turning a lever into code). Everything is grounded in gfx950 (MI355X).

Two folders, diagnosis then treatment:

- **`profiling/`** — measure and classify. Here that means `aka-eval performance` (time per workload
  against the baseline), `aka-eval compile` (registers, LDS, occupancy) and `aka-eval counters`
  (hardware counters and a verdict). `counters.md` maps every field of those replies to a verdict
  and a lever.
- **`optimization/`** — the levers, one card per technique, each tied to the hardware fact it exploits
  and the bottleneck class it addresses. `optimization/LEVERS.md` is its index.

**Classify before you optimize.** Don't pull a lever until you know which roof the kernel sits under:
tuning MFMA on a bandwidth-bound norm, or coalescing a compute-bound GEMM, is wasted effort.

## Golden rules

- **Four failure modes:** compute-bound, bandwidth-bound, occupancy-limited, latency-bound (plus the
  LDS-bound sub-case). Different roofs, different levers.
- **Most elementwise, norm and reduction kernels are bandwidth-bound:** optimize bytes moved, not
  FLOPs. Count the bytes the kernel must move and divide by about 8 TB/s (less in practice) for a
  floor; the Infinity Cache (256 MiB) can beat that for small working sets.
- **Peak is not achievable.** Tuned GEMM sustains about 45–55% of the matrix peak.
- **Measure honestly.** A speedup smaller than run-to-run noise (MI355X per-XCD clocks vary 3–10%) is
  not a result. `aka-eval performance` uses the grader's timing; compare ratios, not single numbers.
- **Apply levers top-down:** algorithm and fusion, then feeding the matrix cores, then parallelism and
  grid fill, then memory access, and keep it correct throughout.
- **Accumulate in fp32** and check every fast path against the reference; watch the fp8 OCP (gfx950)
  vs FNUZ (gfx942) trap.

## Start here — problem → files → order

| Task / symptom | Read in this order |
|---|---|
| "Where do I start on this kernel?" | `profiling/measure_protocol.md` → `profiling/measure_roofline.md` → `profiling/measure_triage.md` |
| "I have an `aka-eval counters` or `compile` reply; what does it say?" | `profiling/counters.md` → the lever it names |
| "Classify it: compute / bandwidth / occupancy / latency" | `profiling/measure_triage.md` → `optimization/lever_bottleneck_class.md` → `profiling/measure_roofline.md` |
| "Compute-bound GEMM — feed the matrix cores" | `optimization/lever_mfma_sched.md` → `optimization/lever_prefetch.md` → `optimization/lever_lds_banks.md` → `optimization/lever_occupancy.md` |
| "Bandwidth-bound (norm, elementwise, reduction, GEMV)" | `optimization/lever_coalescing.md` → `optimization/lever_fusion.md` → `optimization/lever_xcd_locality.md` |
| "Low occupancy, register pressure, spilling" | `optimization/lever_occupancy.md` → `optimization/lever_grid_sizing.md` |
| "Latency-bound: both roofs far, occupancy fine" | `optimization/lever_prefetch.md` → `optimization/lever_grid_sizing.md` |
| "LDS bank conflicts, `ds_read`/`ds_write` stalls, tile won't fit" | `optimization/lever_lds_banks.md` → `optimization/lever_prefetch.md` |
| "Chiplet locality, L2 reuse, fewer than 1024 workgroups" | `optimization/lever_xcd_locality.md` → `optimization/lever_grid_sizing.md` |
| "Should I fuse these ops?" | `optimization/lever_bottleneck_class.md` → `optimization/lever_fusion.md` |
| "Workgroup and grid sizing, `__launch_bounds__`, persistent kernels" | `optimization/lever_grid_sizing.md` → `optimization/lever_xcd_locality.md` |
| "Accuracy regressed: fp8 mismatch, softmax overflow, norm drift" | `optimization/lever_numerics.md` |
| "Did my change actually help?" | `profiling/measure_protocol.md` |

## Folder structure

```
common_methodology/
├── INDEX.md                      ← this map
├── profiling/                    ← DIAGNOSIS
│   ├── counters.md               # the aka-eval compile and counters replies → verdict → lever
│   ├── measure_protocol.md       # warmup, repeats, noise band, A/B discipline
│   ├── measure_roofline.md       # building and reading a roofline, per-dtype roofs
│   └── measure_triage.md         # decision flow: compute / bandwidth / occupancy / latency
└── optimization/                 ← TREATMENT
    ├── LEVERS.md                 # lever hierarchy and per-file table
    ├── lever_bottleneck_class.md # arithmetic intensity, machine balance, bottleneck → lever
    ├── lever_occupancy.md        # 512 VGPR/SIMD, 16-register granule, AGPRs, spill cliff
    ├── lever_lds_banks.md        # 160 KiB over 64 banks, padding vs XOR swizzle, double buffering
    ├── lever_prefetch.md         # direct-to-LDS loads, software pipelining
    ├── lever_mfma_sched.md       # 16×16 vs 32×32 MFMA, AGPR accumulators, issue cadence
    ├── lever_coalescing.md       # 128-bit loads, alignment, grid-stride access
    ├── lever_grid_sizing.md      # wave64, workgroup size, __launch_bounds__, 256 CUs, 8 XCDs
    ├── lever_xcd_locality.md     # per-XCD L2, ≥1024 workgroups, multiples of 8, CTA swizzle
    ├── lever_fusion.md           # when to fuse and when not to
    └── lever_numerics.md         # fp32 accumulation, online softmax, Welford, fp8 scale trap
```

## Reading depth

- **Just diagnosing:** `profiling/counters.md` and `profiling/measure_triage.md`.
- **Applying one lever:** the single `optimization/` card; it names the `hardware/` cards it relies on.
- **Correctness:** `optimization/lever_numerics.md` before trusting a fast path.

Hardware numbers live in `/kb/hardware/`; HIP mechanics in `/kb/languages/hip/`.
