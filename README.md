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
- **Warp Ballot Intrinsics** (`__ballot_sync`, `__any_sync`, `__all_sync`)
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
│   ├── reduce.cuh          ← Warp reduce: sum, max, min, argmax (Module 2)
│   ├── scan.cuh            ← Warp scan: inclusive, exclusive (Module 3)
│   ├── ballot.cuh          ← Ballot: popcount, leader, compaction (Module 4)
│   ├── broadcast.cuh       ← Broadcast, rotate, reverse, zip (Module 5)
│   ├── sort.cuh            ← Bitonic sort in registers (Module 6)
│   └── topk.cuh            ← Top-K selection (Module 7)
├── cooperative/
│   └── warptile.cuh        ← WarpTile<N> via cooperative groups (Module 8)
├── kernels/
│   ├── softmax.cuh         ← Warp softmax (zero shared mem)
│   ├── layernorm.cuh       ← Warp LayerNorm
│   ├── topk_sampling.cuh   ← Fused softmax + top-K + RMS Norm
│   ├── attention_reduce.cuh← Attention score reduction
│   └── dot_product.cuh     ← Batched dot product
├── bench/
│   ├── benchmark.cuh       ← Timing harness, CSV writer
│   └── benchmark_main.cu   ← Main benchmark runner
├── tests/
│   └── correctness.cu      ← CPU reference validation
└── liftoff.cuh             ← Single-header include-all
```

## Quick Start (Kaggle)

```python
# 1. Upload the liftoff/ directory to Kaggle
# 2. Run the driver script:
!python kaggle_driver.py
```

Or build manually:

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

## Module Overview

### Module 1: Intrinsic Wrappers (`intrinsics.cuh`)
Type-safe wrappers over raw CUDA shuffle/ballot intrinsics with 64-bit double support.

### Module 2: Warp Reduce (`reduce.cuh`)
Butterfly XOR pattern reductions — 5 shuffle rounds for 32 threads:
```cuda
float sum = warp_reduce_sum(val);     // All lanes get the sum
float mx  = warp_reduce_max(val);     // All lanes get the max
```

### Module 3: Warp Scan (`scan.cuh`)
Kogge-Stone prefix scan via `__shfl_up_sync`:
```cuda
float prefix = warp_scan_inclusive(val);   // Lane k = sum(input[0..k])
float excl   = warp_scan_exclusive(val);  // Lane k = sum(input[0..k-1])
```

### Module 4: Ballot Utilities (`ballot.cuh`)
Warp-wide predicate operations:
```cuda
int count  = warp_popcount(pred);         // How many lanes are true
int leader = warp_leader_lane(pred);      // Lowest true lane
int rank   = warp_my_rank(pred);          // My rank among true lanes
```

### Module 5: Broadcast & Exchange (`broadcast.cuh`)
Register-to-register data movement:
```cuda
float bc  = warp_broadcast(val, src_lane);  // All lanes get src's value
float rot = warp_rotate_up(val, delta);     // Cyclic rotation
float rev = warp_reverse(val);              // Lane 0↔31, 1↔30, ...
```

### Module 6: Bitonic Sort (`sort.cuh`)
32-element sort entirely in registers — 15 shuffle rounds:
```cuda
warp_sort_ascending(val);                   // Sort 32 values across warp
warp_sort_pairs_ascending(key, val);        // Key-value pair sort
```

### Module 7: Top-K (`topk.cuh`)
Register-only top-K selection for K ≤ 32:
```cuda
warp_topk<float, 8>(val, idx, out_vals, out_idxs);  // Top-8
unsigned mask = warp_topk_mask(val, k);               // Which lanes are top-K
```

### Module 8: WarpTile (`warptile.cuh`)
Sub-warp cooperative group abstraction:
```cuda
WarpTile<8> tile;                           // 8-thread tile (4 tiles per warp)
float sum = tile.reduce_sum(val);           // Reduce within 8-thread group
float bc  = tile.broadcast(val, 0);         // Broadcast within tile
```

## Target Hardware

| GPU | SM | Status |
|---|---|---|
| Tesla T4 | 7.5 (Turing) | ✅ Primary target |
| Tesla P100 | 6.0 (Pascal) | ✅ Supported |
| A100 | 8.0 (Ampere) | ✅ Extended |

## Author

**Maaran** — ML Systems Engineering Research  
GIET MTech CSE 2026–2028

---

*LIFTOFF v1.0 — Antigravity GPU Systems*
