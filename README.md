# LIFTOFF — Warp-Level Primitives Library

> **Replacing Shared Memory with Register-Space Communication via Warp Shuffle Intrinsics**

[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Compute](https://img.shields.io/badge/Compute-7.0%2B-blue)](https://developer.nvidia.com/cuda-gpus)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20GPU-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com)

---

## What is LIFTOFF?

LIFTOFF is a **zero-dependency, header-only CUDA C++ library** that eliminates shared memory (`__shared__`) as the default intra-warp communication medium. It replaces shared memory entirely with:

- **Warp Shuffle Intrinsics** (`__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, `__shfl_down_sync`)
- **Warp Ballot Intrinsics** (`__ballot_sync`, `__any_sync`, `__all_sync`, `__match_any_sync`)
- **CUDA Cooperative Groups** (`cg::tiled_partition<N>`)

The result: **lower latency, higher occupancy, zero bank conflicts, and extreme composability**.

## Why Not Shared Memory?

| Problem | Shared Memory | LIFTOFF (Shuffles) |
|---|---|---|
| **Bank conflicts** | Up to 32× serialization | Zero by design |
| **Occupancy** | Large allocations reduce warps/SM | Zero shared memory used |
| **Latency** | ~20–30 cycles (L1-equivalent) | ~2 cycles (register-to-register) |
| **Synchronization** | `__syncthreads()` required | Implicit warp-synchronous |

## Project Structure

```
liftoff/
├── core/
│   ├── config.cuh          ← Warp size, SM detection, arch guards
│   ├── types.cuh           ← Lane utilities, numeric limits
│   └── profile.cuh         ← CUDA Event timing, NVTX, error checking
├── primitives/
│   ├── intrinsics.cuh      ← Safe shuffle/ballot wrappers (Module 1)
│   ├── reduce.cuh          ← Warp reduce: sum, max, min, argmax, half2 (Module 2)
│   ├── scan.cuh            ← Warp scan: inclusive, exclusive, generic (Module 3)
│   ├── ballot.cuh          ← Ballot: popcount, leader, compaction, elect (Module 4)
│   ├── broadcast.cuh       ← Broadcast, rotate, reverse, transpose, zip (Module 5)
│   ├── sort.cuh            ← Bitonic sort, odd-even sort, KV pairs (Module 6)
│   └── topk.cuh            ← Top-K, bottom-K, streaming buffer (Module 7)
├── cooperative/
│   └── warptile.cuh        ← WarpTile<N> with sort, scan, reduce (Module 8)
├── kernels/
│   ├── softmax.cuh         ← Warp softmax (zero shared mem)
│   ├── layernorm.cuh       ← Warp LayerNorm
│   ├── topk_sampling.cuh   ← Fused softmax + top-K + RMS Norm
│   ├── attention_reduce.cuh← Attention Q·K^T/√d + online softmax
│   ├── dot_product.cuh     ← Batched dot product
│   ├── gelu_reduce.cuh     ← GELU activation + fused reduce
│   ├── histogram.cuh       ← Warp histogram via ballot
│   └── compositions.cuh    ← Fused recipes: LN+GELU, stream compact
├── bench/
│   ├── benchmark.cuh       ← Timing harness, CSV writer
│   └── benchmark_main.cu   ← 10 benchmarks + occupancy analysis
├── tests/
│   └── correctness.cu      ← 22+ tests: edge cases, WarpTile, argmax
└── liftoff.cuh             ← Single-header include-all
```

## Quick Start (Kaggle)

```bash
# Build correctness tests
nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 \
     -I./liftoff liftoff/tests/correctness.cu -o liftoff_test
./liftoff_test

# Build benchmarks
nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 \
     -I./liftoff liftoff/bench/benchmark_main.cu -o liftoff_bench
./liftoff_bench
```

Or use the Python driver: `python kaggle_driver.py`

## Module Overview

| Module | Header | Key Functions |
|---|---|---|
| **1. Intrinsics** | `intrinsics.cuh` | `shfl_down`, `shfl_xor`, `shfl_idx`, `ballot`, `match_any` |
| **2. Reduce** | `reduce.cuh` | `warp_reduce_sum/max/min/prod`, `warp_reduce_argmax`, `half2` |
| **3. Scan** | `scan.cuh` | `warp_scan_inclusive/exclusive`, generic op scan |
| **4. Ballot** | `ballot.cuh` | `warp_popcount`, `warp_leader_lane`, `warp_my_rank`, `warp_elect_one` |
| **5. Broadcast** | `broadcast.cuh` | `warp_broadcast`, `warp_rotate`, `warp_reverse`, `warp_zip` |
| **6. Sort** | `sort.cuh` | `warp_sort_ascending`, `warp_odd_even_sort`, KV pair sort |
| **7. Top-K** | `topk.cuh` | `warp_topk`, `warp_bottom_k`, `warp_topk_with_indices` |
| **8. WarpTile** | `warptile.cuh` | `WarpTile<N>` with `reduce_sum/max/min`, `sort`, `scan`, `barrier` |
| **9. ML Kernels** | `kernels/*.cuh` | Softmax, LayerNorm, Attention, Dot Product, RMS Norm, GELU, Histogram |

## Target Hardware

| GPU | SM | Status |
|---|---|---|
| Tesla T4 | 7.5 (Turing) | ✅ Primary |
| Tesla P100 | 6.0 (Pascal) | ✅ Supported |
| A100 | 8.0 (Ampere) | ✅ Extended |

## Author

**Maaran** — ML Systems Engineering Research | GIET MTech CSE 2026–2028

*LIFTOFF v1.0 — Antigravity GPU Systems*
