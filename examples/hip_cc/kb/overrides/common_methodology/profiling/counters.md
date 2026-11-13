---
title: counters — reading the aka-eval compile and counters replies
kind: measure
measure: counters
gens: [gfx950]
---

# Reading `aka-eval compile` and `aka-eval counters`

Both replies are observations only: they never change the grade. Each field below maps to a verdict
from `measure_triage.md` and a lever card. Read the fields in order; stop at the first one that
explains the time.

## 1. The compile resource report (local, every build)

| Field | How to read it on gfx950 | Lever |
|---|---|---|
| `VGPR` | 512 per SIMD, allocated in steps of 16: `occupancy = min(8, floor(512 / round16(VGPR)))` waves/SIMD. 64 → 8, 96 → 5, 128 → 4, 256 → 2 | `../optimization/lever_occupancy.md` |
| `AGPR` | drawn from the same 512-entry pool as VGPRs; MFMA accumulators live here | `../optimization/lever_mfma_sched.md` |
| `SGPR` | rarely the limit; a large count hints at heavy uniform address math | — |
| `scratch`, `spills` | any non-zero value is a red flag: spills go to memory | `../optimization/lever_occupancy.md` |
| `LDS` | bytes per workgroup; `occ_lds = floor(163840 / round_up(LDS, 1280))` workgroups per CU | `../optimization/lever_lds_banks.md` |
| `occupancy` | the compiler's waves per SIMD after VGPR, AGPR and LDS limits | `../optimization/lever_occupancy.md` |

Workgroups per CU are also capped by wave slots: 32 waves per CU, so a 256-thread workgroup (4 waves)
fits at most 8 per CU (`/kb/hardware/mi350_execution.md`).

## 2. The counters reply (MI355X, after correctness passes)

The grader runs your kernel under `rocprofv3` in fixed counter passes and summarizes the top candidate
kernels by time. Kernel names are shortened. Fields, with the counters behind them:

| Field | Derived from | How to read it |
|---|---|---|
| **time share** | kernel trace | Work on the kernel with the largest share first; a kernel under 10% isn't worth tuning |
| **fabric GB/s vs MALL and HBM roofs** | `TCC_EA0_RDREQ*`, `TCC_EA0_WRREQ*` (L2→fabric requests × 32/64 B) over kernel time | Near either roof → **bandwidth-bound**. Small working sets are served by the 256 MiB Infinity Cache (MALL), so compare against both roofs; never quote "% of HBM" alone |
| **L2 hit %** | `TCC_HIT_sum`, `TCC_MISS_sum` | Low hit rate on data that is reused → locality problem (`../optimization/lever_xcd_locality.md`); L2 is per XCD |
| **instruction mix** | `SQ_INSTS_VALU`, `SALU`, `VMEM`, `LDS`, `MFMA` per wave | VALU-heavy with low bandwidth → address or math overhead (vectorize, cut index math); MFMA present but bandwidth low → matrix cores starved (`../optimization/lever_mfma_sched.md`) |
| **LDS bank conflicts** | `SQ_LDS_BANK_CONFLICT` relative to `SQ_INSTS_LDS` | Non-zero and rising with tile size → **LDS-bound** (`../optimization/lever_lds_banks.md`, 64 banks) |
| **busy / waves** | `SQ_BUSY_CYCLES`, `SQ_WAVES`, `GRBM_GUI_ACTIVE` | Low busy fraction with short kernels → launch-bound or too little work per launch |
| **grid fill** | launch dimensions | Fewer than 1024 workgroups, or a count that isn't a multiple of 8 XCDs, leaves CUs idle (`../optimization/lever_grid_sizing.md`, `../optimization/lever_xcd_locality.md`) |
| **verdict** | the rules above | One of compute-bound, bandwidth-bound, occupancy-limited, latency-bound, LDS-bound, with the card to read |

## 3. Verdict → lever

| Verdict | First lever | Then |
|---|---|---|
| bandwidth-bound | `../optimization/lever_coalescing.md` (128-bit loads, contiguous lanes) | `../optimization/lever_fusion.md`, `../optimization/lever_xcd_locality.md` |
| compute-bound | `../optimization/lever_mfma_sched.md` | `../optimization/lever_prefetch.md` |
| occupancy-limited | `../optimization/lever_occupancy.md` | `../optimization/lever_grid_sizing.md` |
| latency-bound | `../optimization/lever_prefetch.md` (more loads in flight) | `../optimization/lever_grid_sizing.md` |
| LDS-bound | `../optimization/lever_lds_banks.md` | `/kb/languages/hip/skills/optimize/hip_levers/hip_lds_staging.md` |

## When counters are unavailable

The reply may say "unavailable" (the counter tool is down or this run doesn't offer it); that check is
not counted. Classify from the compile report and `aka-eval performance` instead: compute the bytes
each workload must move, divide by the measured time, and compare with about 8 TB/s (HBM) or several
times that for working sets that fit the Infinity Cache. Far below both with high occupancy suggests
latency; far below with low occupancy suggests registers or LDS.
