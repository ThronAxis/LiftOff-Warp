# LIFTOFF: Warp-Level Primitives Library
## Complete Project Report

> **Author:** Maaran | ML Systems Engineering Research  
> **Institution:** GIET MTech CSE 2026–2028  
> **Platform:** NVIDIA Tesla T4 (Kaggle), CUDA 12.x, Compute 7.5  
> **Repository:** github.com/ThronAxis/LiftOff-Warp  
> **Date:** June 2026

---

## 1. Abstract

LIFTOFF is a zero-dependency, header-only CUDA C++ library that eliminates shared memory (`__shared__`) as the default intra-warp communication medium. It replaces shared memory entirely with register-space warp shuffle intrinsics (`__shfl_sync`), warp ballot intrinsics (`__ballot_sync`), and CUDA Cooperative Groups (`cg::tiled_partition<N>`).

**Key Results on Tesla T4:**
- **2.24× faster** than shared memory reductions
- **65 shuffle instructions vs 5 shared memory ops** in generated PTX
- **Zero synchronization barriers** on 11/12 kernels
- **622.7 GOPS** peak throughput (GELU fused reduce)
- **25/25 correctness tests** passed

---

## 2. Problem Statement

### 2.1 The Shared Memory Bottleneck

Conventional GPU programming uses `__shared__` memory for intra-block communication. While functional, it has structural limitations:

| Problem | Impact |
|---|---|
| **Bank conflicts** | Up to 32× serialization on 32-bank hardware |
| **Occupancy wall** | Large allocations reduce concurrent warps/SM |
| **Latency** | ~20–30 cycles (L1-equivalent), not register speed |
| **Synchronization** | Requires `__syncthreads()` — full block barrier |

### 2.2 The Warp Shuffle Alternative

CUDA warp shuffle intrinsics (available since CC 3.0) provide register-to-register data exchange within a warp of 32 threads:

| Property | Shared Memory | Warp Shuffle |
|---|---|---|
| Latency | ~20–30 cycles | ~2 cycles |
| Bank conflicts | Up to 32× | Zero by design |
| Shared mem usage | Yes | Zero |
| Synchronization | `__syncthreads()` | Implicit (warp-synchronous) |
| Scope | Block-wide | Warp-wide (32 threads) |

Despite being fundamentally faster, shuffle intrinsics are rarely used as composable building blocks. **LIFTOFF closes this gap.**

---

## 3. Architecture

### 3.1 Layer Stack

```
┌──────────────────────────────────────────────────────┐
│                  APPLICATION LAYER                    │
│  softmax | layernorm | attention | GELU | RMS norm   │
├──────────────────────────────────────────────────────┤
│                  COMPOSITION LAYER                    │
│  fused LN+GELU | stream compact | online softmax    │
├──────────────────────────────────────────────────────┤
│                  PRIMITIVE LAYER                      │
│  reduce | scan | broadcast | sort | topk | ballot    │
├──────────────────────────────────────────────────────┤
│                  INTRINSIC LAYER                      │
│  __shfl_sync | __shfl_xor_sync | __ballot_sync      │
├──────────────────────────────────────────────────────┤
│                  COOPERATIVE GROUPS                    │
│  cg::tiled_partition<N> | WarpTile<N>                │
├──────────────────────────────────────────────────────┤
│                  HARDWARE                             │
│  NVIDIA Warp (32 threads) | Register File | SM       │
└──────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
liftoff/                          (23 source files)
├── core/
│   ├── config.cuh               — Warp size, SM detection, device info
│   ├── types.cuh                — Lane/warp IDs, numeric limits
│   └── profile.cuh              — CudaTimer, NVTX, CUDA_CHECK, occupancy
├── primitives/
│   ├── intrinsics.cuh           — Type-safe shuffle/ballot wrappers
│   ├── reduce.cuh               — sum/max/min/prod/argmax, half2, segmented
│   ├── scan.cuh                 — inclusive/exclusive prefix scan
│   ├── ballot.cuh               — popcount, leader, rank, compact, elect
│   ├── broadcast.cuh            — broadcast, rotate, reverse, zip
│   ├── sort.cuh                 — bitonic sort, odd-even sort, KV pairs
│   └── topk.cuh                 — top-K, bottom-K, streaming buffer
├── cooperative/
│   └── warptile.cuh             — WarpTile<N> sub-warp abstraction
├── kernels/
│   ├── softmax.cuh              — Numerically stable warp softmax
│   ├── layernorm.cuh            — Warp LayerNorm with affine
│   ├── topk_sampling.cuh        — Fused softmax+top-K, RMS Norm
│   ├── attention_reduce.cuh     — Q·K^T/√d + online softmax
│   ├── dot_product.cuh          — Batched warp dot product
│   ├── gelu_reduce.cuh          — GELU activation + fused reduce
│   ├── histogram.cuh            — Warp histogram via ballot
│   └── compositions.cuh         — Fused LN+GELU, stream compact
├── bench/
│   ├── benchmark.cuh            — Timing harness, CSV writer
│   └── benchmark_main.cu        — 10 benchmarks + occupancy
├── tests/
│   └── correctness.cu           — 25 tests with edge cases
└── liftoff.cuh                  — Single-header include-all
```

---

## 4. Module Implementation Details

### Module 1: Intrinsic Wrappers (`intrinsics.cuh`)

Type-safe wrappers over raw CUDA shuffle/ballot intrinsics. Eliminates `unsigned mask` boilerplate and adds 64-bit double support via register splitting.

**Key functions:**
- `shfl_down(val, delta, mask)` — Shift value down by delta lanes
- `shfl_up(val, delta, mask)` — Shift value up by delta lanes
- `shfl_xor(val, lane_mask, mask)` — XOR butterfly exchange
- `shfl_idx(val, src_lane, mask)` — Read from specific lane
- `ballot(predicate, mask)` — 32-bit predicate bitmask
- `match_any(val, mask)` — Lanes with matching values

**Design decision:** Double-precision support uses two 32-bit shuffles (hi/lo word split) since `__shfl_sync` only supports 32-bit values natively.

### Module 2: Warp Reduce (`reduce.cuh`)

Butterfly XOR pattern — 5 shuffle rounds for 32 threads, O(log₂N) complexity.

```
Algorithm: Butterfly XOR Reduction
  Round 1: shfl_xor(val, 16)  → pairs (0,16), (1,17), ...
  Round 2: shfl_xor(val, 8)   → pairs (0,8), (1,9), ...
  Round 3: shfl_xor(val, 4)   → pairs (0,4), (1,5), ...
  Round 4: shfl_xor(val, 2)   → pairs (0,2), (1,3), ...
  Round 5: shfl_xor(val, 1)   → pairs (0,1), (2,3), ...
  Result: All lanes hold the reduced value
```

**Functions:** `warp_reduce_sum`, `warp_reduce_max`, `warp_reduce_min`, `warp_reduce_prod`, `warp_reduce_argmax`, `warp_reduce_argmin`

### Module 3: Warp Scan (`scan.cuh`)

Kogge-Stone prefix scan via `__shfl_up_sync` — 5 rounds for 32 elements.

```
Algorithm: Kogge-Stone Inclusive Scan
  Round 1: val += shfl_up(val, 1)   if lane >= 1
  Round 2: val += shfl_up(val, 2)   if lane >= 2
  Round 3: val += shfl_up(val, 4)   if lane >= 4
  Round 4: val += shfl_up(val, 8)   if lane >= 8
  Round 5: val += shfl_up(val, 16)  if lane >= 16
  Result: Lane k holds sum(input[0..k])
```

**Functions:** `warp_scan_inclusive`, `warp_scan_exclusive`, `warp_scan_with_total`

### Module 4: Ballot Utilities (`ballot.cuh`)

Warp-wide predicate operations using `__ballot_sync` and `__popc`.

**Functions:**
- `warp_popcount(pred)` — Count of true lanes
- `warp_leader_lane(pred)` — Lowest true lane index
- `warp_my_rank(pred)` — Rank among true lanes (via masked popcount)
- `warp_elect_one(pred)` — Single leader election
- `warp_compact_indices(pred)` — Gather indices of active lanes

### Module 5: Broadcast & Exchange (`broadcast.cuh`)

Register-to-register data movement patterns.

**Functions:**
- `warp_broadcast(val, src_lane)` — All lanes get src's value
- `warp_rotate_up(val, delta)` — Cyclic rotation
- `warp_reverse(val)` — Lane 0↔31, 1↔30, ...
- `warp_zip/warp_unzip` — Interleave/deinterleave

### Module 6: Bitonic Sort (`sort.cuh`)

32-element sort entirely in registers — 15 compare-and-swap rounds using shuffle-based partner exchange.

```
Algorithm: Bitonic Sort Network (32 elements)
  For k = 2, 4, 8, 16, 32:
    For stride = k/2, k/4, ..., 1:
      partner = lane XOR stride
      other = shfl_xor(val, stride)
      if (ascending direction): keep min/max based on lane position
  Total: 15 shuffle rounds, zero shared memory
```

**Functions:** `warp_sort_ascending`, `warp_sort_descending`, `warp_sort_pairs_ascending`, `warp_odd_even_sort`

### Module 7: Top-K Selection (`topk.cuh`)

Register-only top-K for K ≤ 32 using sort + prefix select.

**Functions:** `warp_top1`, `warp_topk`, `warp_bottom_k`, `warp_topk_with_indices`, `warp_topk_mask`, `WarpTopKBuffer` (streaming)

### Module 8: WarpTile (`warptile.cuh`)

Sub-warp cooperative group abstraction. Wraps `cg::tiled_partition<N>` for tiles of 2, 4, 8, 16, or 32 threads.

**Methods:** `reduce_sum`, `reduce_max`, `reduce_min`, `scan_inclusive_sum`, `scan_exclusive_sum`, `broadcast`, `sort`, `barrier`

Also provides `block_reduce_sum_hybrid` (shuffle within warps + minimal shared memory for cross-warp) and `grid_reduce_sum`.

### Module 9: ML Application Kernels

| Kernel | Algorithm | Shared Memory Used |
|---|---|---|
| **Softmax** | max → exp-sum → normalize (3 reduce passes) | **Zero** |
| **LayerNorm** | mean → variance → normalize + affine | **Zero** |
| **RMS Norm** | sum-of-squares → rsqrt → scale | **Zero** |
| **Dot Product** | element-wise multiply → warp reduce sum | **Zero** |
| **GELU Reduce** | GELU activation → fused warp sum | **Zero** |
| **Attention** | Q·K^T/√d → online softmax merge | **Zero** |
| **Histogram** | ballot-based bin counting | **Zero** |
| **Fused LN+GELU** | LayerNorm → GELU in single pass | **Zero** |

---

## 5. Build System

Header-only — no Makefile or CMake required. Single compilation command:

```bash
# Correctness tests
nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 \
     -I./liftoff liftoff/tests/correctness.cu -o liftoff_test

# Benchmarks
nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 \
     -I./liftoff liftoff/bench/benchmark_main.cu -o liftoff_bench
```

**Compiler flags:**
- `-O3` — Maximum optimization
- `--use_fast_math` — Fast transcendentals (expf, rsqrtf)
- `-std=c++17` — Lambda support for benchmark harness
- `-arch=sm_75` — Tesla T4 Turing architecture

---

## 6. Testing

### 6.1 Test Suite (25 tests)

| Category | Tests | Status |
|---|---|---|
| **Reduce** | sum, max, min, all-zero edge, all-same edge | 5/5 ✅ |
| **Scan** | inclusive (ones), exclusive (ones), varying input | 3/3 ✅ |
| **Sort** | reverse→sorted, all-same stability | 2/2 ✅ |
| **Ballot** | popcount, leader lane, rank ordering | 3/3 ✅ |
| **Broadcast** | broadcast, reverse, rotate | 3/3 ✅ |
| **Softmax** | sum≈1.0, positivity, monotonicity, uniform edge | 4/4 ✅ |
| **Dot Product** | [1..32]·[1..1] = 528 | 1/1 ✅ |
| **WarpTile\<8\>** | reduce_sum (4 tiles), reduce_max (4 tiles) | 2/2 ✅ |
| **Argmax** | value correctness, index correctness | 2/2 ✅ |
| **Total** | | **25/25 ✅** |

### 6.2 Key Bug Found During Testing

**Shuffle participation rule:** All lanes in the active mask must execute `__shfl_xor_sync`. Wrapping the call in `if (lane_id() == 0)` causes undefined behavior since only 1 of 32 lanes participates. Fix: all lanes execute the shuffle, only lane 0 writes the result.

---

## 7. Benchmark Results

### 7.1 Hardware

```
Device: Tesla T4 (Turing)
SM Count: 40 | SM Version: 7.5
Global Memory: 14.6 GB
Shared Memory/Block: 48 KB
Max Threads/SM: 1024 | Max Warps/SM: 32
Warp Size: 32 | Clock Rate: 1590 MHz
```

### 7.2 Micro-Benchmarks (1M elements)

| Kernel | Median Latency | Throughput |
|---|---|---|
| **warp_reduce_sum (shuffle)** | **38.91 μs** | **26.95 GOPS** |
| **shared_mem_reduce (baseline)** | **92.54 μs** | **11.33 GOPS** |
| warp_scan_inclusive | 38.88 μs | 134.85 GOPS |
| warp_sort_ascending | 55.30 μs | 284.44 GOPS |

**Shuffle reduce is 2.38× faster than shared memory reduce.**

### 7.3 Application Benchmarks

| Kernel | Median Latency | Throughput |
|---|---|---|
| warp_softmax (4096×256) | 40.99 μs | 102.32 GOPS |
| warp_layernorm (1024×768) | 36.86 μs | 106.67 GOPS |
| warp_dot_product (4096×256) | 36.86 μs | 56.89 GOPS |
| warp_attention (512×64) | 180.22 μs | 186.18 GOPS |
| warp_rmsnorm (1024×768) | 35.17 μs | 67.09 GOPS |
| warp_gelu_reduce (4096×256) | 13.47 μs | **622.67 GOPS** |

### 7.4 Scaling Analysis (Shuffle vs Shared Memory)

| Problem Size | Shuffle (μs) | Shared (μs) | Speedup |
|---|---|---|---|
| 64K | 8.19 | 14.24 | **1.74×** |
| 256K | 12.93 | 28.70 | **2.22×** |
| 1M | 34.85 | 78.08 | **2.24×** |
| 4M | 87.10 | 164.48 | **1.89×** |
| 16M | 327.36 | 653.25 | **2.00×** |

**Consistent 1.74×–2.24× speedup across all problem sizes.**

### 7.5 PTX Instruction Analysis

| Metric | Count |
|---|---|
| Shuffle instructions (`shfl.sync`) | **65** |
| Shared memory loads (`ld.shared`) | 3 |
| Shared memory stores (`st.shared`) | 2 |
| **Shuffle-to-shared ratio** | **13:1** |

The 5 shared memory ops come from the intentional baseline kernel and the cross-warp hybrid reduce stage.

### 7.6 Register Pressure

| Register Count | Kernels |
|---|---|
| 10–12 registers | Lightweight primitives (6 kernels) |
| 24–46 registers | ML application kernels (4 kernels) |
| 64 registers | Attention kernel (most complex) |

**Zero shared memory allocation** on all LIFTOFF kernels. Maximum register usage (64) is well within the T4's 65536 registers per SM.

### 7.7 Occupancy

| Kernel | Blocks/SM | Total Warps |
|---|---|---|
| warp_reduce (shuffle) | 4 | 1280 |
| shared_reduce (baseline) | 4 | 1280 |
| warp_sort | 4 | 1280 |

---

## 8. How the Project Was Built

### Phase 1: Requirements Analysis
- Studied the PRD specifying 9 modules and 7 goals
- Studied the Implementation Guide with detailed algorithms
- Identified key algorithms: butterfly XOR reduce, Kogge-Stone scan, bitonic sort

### Phase 2: Core Layer (Day 1)
- `config.cuh` — SM detection, warp size constants, `print_device_info()`
- `types.cuh` — `lane_id()`, `warp_id()`, numeric limits
- `profile.cuh` — CUDA Event timer, `CUDA_CHECK` macro, occupancy queries

### Phase 3: Primitive Layer (Day 1-2)
- Built bottom-up: intrinsics → reduce → scan → ballot → broadcast → sort → topk
- Each module depends only on lower modules (no circular deps)
- Every function is `__device__ __forceinline__` for zero overhead

### Phase 4: Cooperative Groups (Day 2)
- `WarpTile<N>` wrapping `cg::tiled_partition<N>`
- Block-level hybrid reduce (shuffle + minimal shared for cross-warp)

### Phase 5: ML Kernels (Day 2-3)
- Softmax, LayerNorm, RMS Norm, Dot Product, Attention, GELU, Histogram
- Each kernel uses only LIFTOFF primitives — zero `__shared__` declarations
- Added composition recipes (fused LN+GELU, stream compact)

### Phase 6: Testing & Benchmarking (Day 3)
- 25 correctness tests with CPU reference validation
- Edge cases: all-zero, all-same, uniform distribution
- Sub-warp tests: WarpTile<8> with 4 independent tiles
- 10 benchmarks with shared-memory baseline comparison

### Phase 7: Kaggle Validation (Day 3)
- Pushed to GitHub, cloned on Kaggle T4
- Fixed 3 compilation bugs:
  1. Duplicate function definitions (compositions.cuh ↔ attention_reduce.cuh)
  2. `thread_block_tile` default constructor (→ member initializer list)
  3. Shuffle participation bug (all lanes must call `__shfl_xor_sync`)
- All 25 tests passed, all 10 benchmarks completed

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Header-only** | Zero build system complexity, single `#include` integration |
| **Namespace `liftoff`** | Avoid polluting global scope |
| **`__forceinline__`** | Zero function call overhead — primitives inline at call site |
| **`FULL_MASK = 0xFFFFFFFF`** | All 32 lanes active by default, overridable |
| **Butterfly XOR for reduce** | All lanes get result (vs shfl_down which only gives lane 0) |
| **Kogge-Stone for scan** | O(log N) depth, naturally maps to shfl_up |
| **Member initializer list for WarpTile** | `thread_block_tile` has no default constructor in CUDA 12.x |

---

## 10. Lessons Learned

1. **All lanes must participate in shuffles** — `__shfl_xor_sync` with `FULL_MASK` requires all 32 lanes to execute the call, even if only one lane needs the result
2. **CUDA Cooperative Groups** lack default constructors — must use initializer lists
3. **Shared memory's latency penalty is real** — 2× measured, consistent across sizes
4. **Register pressure stays low** — even complex bitonic sort uses only 46 registers
5. **Header-only CUDA libraries work** — `#pragma once` + namespaces + templates = clean

---

## 11. Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| CUDA C++ | 12.x | Core language |
| nvcc | 12.x | Compiler |
| CUDA Cooperative Groups | Built-in | WarpTile abstraction |
| Python | 3.10 | Kaggle orchestration |
| matplotlib | 3.x | Benchmark visualization |
| Kaggle | T4 GPU | Execution platform |
| Git/GitHub | - | Version control |

---

## 12. Conclusion

LIFTOFF demonstrates that warp shuffle intrinsics are a viable, high-performance replacement for shared memory in intra-warp GPU communication. The library achieves:

- **2.24× lower latency** than shared memory reductions
- **13:1 shuffle-to-shared instruction ratio** in generated PTX
- **Zero synchronization barriers** in warp-level primitives
- **Complete ML kernel coverage** (softmax, LayerNorm, attention, GELU, RMS Norm)

The results validate the core thesis: register-space communication via warp shuffles is faster, simpler, and more composable than shared memory for warp-scoped operations.

---

*LIFTOFF v1.0 — Antigravity GPU Systems*  
*Maaran | ML Systems Engineering Research | GIET MTech CSE 2026–2028*
