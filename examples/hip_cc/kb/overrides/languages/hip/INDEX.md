---
title: HIP / C++ knowledge map — index and problem routing
kind: index
scope: languages/hip
gens: [gfx950]
---

# HIP / C++ — knowledge map

Entry index for `languages/hip/`: how to author and debug a hand-written HIP kernel for gfx950. Your
kernel is one `solution/kernel.hip`, built with `torch.utils.cpp_extension` for gfx950 and registered
as `run` through `TORCH_LIBRARY_EXPAND` (see the task's `_aka_eval/CANDIDATE.md`). Hardware numbers
are in `/kb/hardware/`; this folder doesn't repeat them.

## The two facts behind most HIP bugs

1. **A wavefront is 64 lanes, not 32.** `warpSize == 64`. Every `__shfl`/`__ballot`/manual reduction,
   lane mask (`unsigned long long` with `__popcll`), occupancy calculation and block size (a multiple
   of 64) depends on it. 32-lane CUDA code runs but uses half the machine.
2. **LDS is 160 KiB per CU with 64 banks on gfx950** (64 KiB and 32 banks on gfx942). Padding or
   swizzles tuned for 32 banks are not conflict-free on 64.

Also load-bearing: launch **≥ 1024 workgroups** to fill 256 CUs; keep MFMA accumulators in a stable
vector variable so they stay in AGPRs; fp8 is **OCP** on gfx950 (FNUZ on gfx942).

## Start here — problem → files → order

| Task / symptom | Read in this order |
|---|---|
| "Orient me on authoring HIP kernels" | `skills/optimize/hip_levers/hip_authoring_model.md` → `API_docs/kernel_language.md` |
| "Write or tune an MFMA (matrix-core) kernel" | `skills/optimize/hip_levers/hip_authoring_model.md` → `skills/optimize/hip_levers/hip_builtins.md` → `skills/optimize/hip_levers/hip_lds_staging.md` → `skills/optimize/hip_levers/hip_templates.md` |
| "LDS bank conflicts, double buffering, direct-to-LDS" | `skills/optimize/hip_levers/hip_lds_staging.md` → `skills/optimize/hip_levers/hip_builtins.md` |
| "Wave reductions, grid-stride loops, tiled-GEMM skeleton" | `skills/optimize/hip_levers/hip_templates.md` (skip its host-side stream and graph snippets) |
| "CUDA-style code is wrong or slow, static-assert on a mask" | `skills/optimize/hip_levers/hip_traps.md` → `skills/bottleneck/debug-hip-kernel.md` |
| "Kernel is wrong, crashes, hangs or underperforms" | `skills/bottleneck/debug-hip-kernel.md` |
| "What should I optimize next?" | `skills/profile/profiling-hip.md` → the lever it points to |
| "Qualifiers, launch, `__shfl`/`__ballot`, atomics" | `API_docs/kernel_language.md` |
| "Host API: memory, streams, occupancy query, errors" | `API_docs/runtime_api.md` (the task forbids allocation, streams and sync in `run`) |
| "PyTorch extension build, arch guards" | `API_docs/compilation_and_build.md` (your build flags are fixed) |
| "fp8/fp6/fp4/bf16 headers, MFMA vector types" | `API_docs/dtypes.md` |

## Folder structure

```
languages/hip/
├── INDEX.md                               ← this map
├── API_docs/
│   ├── kernel_language.md                 # qualifiers, launch, indexing, sync, wave64 cross-lane, atomics
│   ├── runtime_api.md                     # memory, streams, events, graphs, occupancy query, errors
│   ├── compilation_and_build.md           # hipcc flags, arch guards, PyTorch extensions
│   └── dtypes.md                          # fp8/fp6/fp4/bf16/fp16 headers, MXFP E8M0, MFMA operand types
└── skills/
    ├── profile/profiling-hip.md           # read aka-eval compile/counters → classify → pick a lever
    ├── bottleneck/debug-hip-kernel.md     # wrong/crash/slow: symptom → cause, wave64, LDS, AGPR, fp8
    └── optimize/hip_levers/
        ├── hip_authoring_model.md         # gfx950 constants, toolchain, capability map (read first)
        ├── hip_builtins.md                # __builtin_amdgcn_mfma_*, buffer loads, cross-lane, sched barriers
        ├── hip_lds_staging.md             # 64-bank conflicts, swizzle, direct-to-LDS, double buffering
        ├── hip_templates.md               # wave64 reductions, grid-stride, tiled GEMM
        └── hip_traps.md                   # CUDA→HIP traps by symptom, occupancy prediction
```

## Reading depth

- **A single API fact:** the one `API_docs/` card.
- **Authoring a matrix-core kernel:** `hip_authoring_model` → `hip_builtins` → `hip_lds_staging` →
  `hip_templates`, then `hip_traps` before trusting a result.
- **A specific failure:** `skills/bottleneck/debug-hip-kernel.md` and its symptom table.

## Pinned reference sources

- ROCm/HIP — https://github.com/ROCm/HIP (runtime API and kernel language; docs at rocm.docs.amd.com/projects/HIP)
- HIP kernel language: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html
- Matrix cores on CDNA3/CDNA4: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
- AMD Matrix Instruction Calculator: https://github.com/ROCm/amd_matrix_instruction_calculator
- AMDGPU backend (buffer descriptors, ds and sched builtins, s_waitcnt): https://llvm.org/docs/AMDGPUUsage.html
- CDNA4 ISA: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
- CDNA4 whitepaper: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf

These are links for provenance; the container has no network.
