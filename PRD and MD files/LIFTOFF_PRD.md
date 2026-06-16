# LIFTOFF: Warp-Level Primitives Library
## Product Requirements Document (PRD) v1.0
### Antigravity GPU Systems — Kaggle Cloud Edition

---

> **Codename:** LIFTOFF  
> **Subtitle:** Replacing Shared Memory with Register-Space Communication via Warp Shuffle Intrinsics  
> **Author:** Maaran | ML Systems Engineering Research  
> **Target Platform:** NVIDIA CUDA (Kaggle T4/P100/A100), CUDA 12.x, Compute Capability 7.0+  
> **Document Status:** Active — Engineering Release Candidate  
> **Date:** June 2026

---

## 1. Executive Summary

**LIFTOFF** is a zero-dependency, header-only CUDA C++ library that eliminates shared memory (`__shared__`) as the default intra-warp communication medium and replaces it entirely with register-space shuffle primitives, warp ballot intrinsics, and CUDA Cooperative Groups. The result is a warp-synchronous execution model with lower latency, higher occupancy, zero bank conflicts, and extreme composability for building high-performance GPU kernels.

LIFTOFF targets the next generation of ML inference kernels (attention, softmax, layer norm, top-k, reductions) where shared memory pressure is the primary occupancy bottleneck. It is designed to run fully on Kaggle-hosted GPUs with zero local setup, making it accessible for student researchers and systems engineers building toward production-grade CUDA expertise.

---

## 2. Problem Statement

### 2.1 The Shared Memory Bottleneck

Conventional GPU programming teaches shared memory (`__shared__`) as the canonical intra-block communication channel. While powerful, shared memory has structural limitations that limit performance at scale:

| Problem | Impact |
|---|---|
| **Bank conflicts** | Up to 32× serialization penalty on 32-bank hardware |
| **Occupancy wall** | Large `__shared__` allocations reduce concurrent warps per SM |
| **Latency** | L1-equivalent latency (~20–30 cycles), not zero like registers |
| **Synchronization** | Requires `__syncthreads()` — serialization point across entire block |
| **Fragility** | Static allocation sizes; runtime sizing requires dynamic allocation hacks |

### 2.2 The Warp Shuffle Gap

CUDA has provided warp shuffle intrinsics since compute capability 3.0 (`__shfl_sync`, `__shfl_xor_sync`, `__shfl_up_sync`, `__shfl_down_sync`) and warp vote/ballot functions (`__ballot_sync`, `__any_sync`, `__all_sync`) since 5.0. Despite being:

- **Faster** (register-to-register, ~2 cycles)
- **Zero bank conflict** by design
- **Higher occupancy** (no shared memory consumed)
- **Implicit warp-synchronous** (no `__syncthreads()` needed within a warp)

...these intrinsics are almost universally used only in isolation (e.g., a single warp reduce) rather than as composable building blocks of a full kernel programming model.

**LIFTOFF closes this gap** by providing a principled, layered, tested library of warp-level primitives that developers can compose like LEGO to build entire kernels without touching shared memory.

### 2.3 The Cooperative Groups Underutilization Problem

CUDA Cooperative Groups (introduced in CUDA 9) formalizes the notion of thread groups at arbitrary granularity — sub-warp tiles, warp, multi-warp blocks, grid, multi-GPU. The API is powerful but verbose and rarely used in student or research code. LIFTOFF wraps Cooperative Groups to provide ergonomic, safe, and composable warp-tile abstractions.

---

## 3. Goals and Non-Goals

### 3.1 Goals

- **G1:** Implement a complete set of warp-level primitives: reduce, scan (prefix sum), broadcast, exchange, sort, top-k, histogram — all using shuffle intrinsics only.
- **G2:** Provide a `WarpTile<N>` abstraction via Cooperative Groups for sub-warp and multi-warp compositions.
- **G3:** Implement `warp_ballot` utilities: population count, leader election, predicated compaction, masked execution.
- **G4:** Build end-to-end ML kernels (softmax, layer norm, attention score reduction, top-k sampling) that outperform naive `__shared__` baselines on Kaggle GPUs.
- **G5:** Produce a benchmark suite with Nsight Compute-compatible profiling annotations.
- **G6:** Be fully runnable on Kaggle (T4, P100, A100) with Python + PyCUDA or CUDA C++ notebooks.
- **G7:** Zero external dependencies — stdlib CUDA only.

### 3.2 Non-Goals

- Cross-vendor support (AMD/ROCm, Intel XPU) — NVIDIA CUDA only for this release.
- Multi-GPU support (future milestone).
- Python-native API (kernel source is CUDA C++; Python used only for orchestration).
- Production deployment pipeline (this is a research/education artifact).

---

## 4. Target Users

| Persona | Need | How LIFTOFF Helps |
|---|---|---|
| **ML Systems Engineer (Student)** | Learn GPU kernel internals beyond textbook `atomicAdd` | Full primitives library with annotated source |
| **CUDA Kernel Engineer (Intern/Junior)** | Replace naive shared-memory reductions in inference kernels | Drop-in `warp_reduce<T>()` composable primitives |
| **Research Engineer** | Prototype custom attention / softmax variants | `WarpTile` abstraction + shuffle-based scan |
| **Kaggle ML Practitioner** | Understand GPU under-the-hood without cloud compute budget | Runs on free T4 GPUs |

---

## 5. Technical Architecture

### 5.1 Layer Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                            │
│   softmax_kernel | layernorm_kernel | attention_reduce | top_k      │
├─────────────────────────────────────────────────────────────────────┤
│                       COMPOSITION LAYER                             │
│   WarpPipeline | WarpFusion | ShuffleScan | BallotCompact           │
├─────────────────────────────────────────────────────────────────────┤
│                       PRIMITIVE LAYER                               │
│   warp_reduce | warp_scan | warp_broadcast | warp_sort | warp_topk  │
├─────────────────────────────────────────────────────────────────────┤
│                       INTRINSIC LAYER                               │
│   __shfl_sync | __shfl_xor_sync | __ballot_sync | __match_any_sync  │
├─────────────────────────────────────────────────────────────────────┤
│                    COOPERATIVE GROUPS LAYER                         │
│   cg::tiled_partition<N> | cg::coalesced_threads() | cg::grid_group │
├─────────────────────────────────────────────────────────────────────┤
│                       HARDWARE LAYER                                │
│   NVIDIA Warp (32 threads) | Register File | SM Scheduler           │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Module Breakdown

#### Module 1: `liftoff/intrinsics.cuh` — Raw Shuffle Wrapper Layer
Provides type-safe, mask-aware wrappers over raw CUDA intrinsics. Eliminates the `unsigned mask` boilerplate. Supports all arithmetic types: `float`, `double`, `int`, `uint32_t`, `__half`, `__half2`, `bfloat16`.

```
warp_shfl(val, src_lane, mask)
warp_shfl_down(val, delta, mask)
warp_shfl_up(val, delta, mask)
warp_shfl_xor(val, lane_mask, mask)
warp_ballot(predicate, mask)         → uint32_t
warp_any(predicate, mask)            → bool
warp_all(predicate, mask)            → bool
warp_match_any(val, mask)            → uint32_t
warp_match_all(val, mask, pred*)     → uint32_t
```

#### Module 2: `liftoff/reduce.cuh` — Warp Reduce Primitives
Full set of warp-level reductions with arbitrary binary operators, templated on type and operator.

```
warp_reduce_sum<T>(val)              → T
warp_reduce_max<T>(val)              → T
warp_reduce_min<T>(val)              → T
warp_reduce_prod<T>(val)             → T
warp_reduce_op<T, BinaryOp>(val, op) → T    ← generic
warp_reduce_and(val)                 → uint32_t
warp_reduce_or(val)                  → uint32_t
```

All reductions use the butterfly (XOR shuffle) pattern, O(log₂ 32) = 5 shuffle instructions, no sync barriers.

#### Module 3: `liftoff/scan.cuh` — Warp Prefix Scan (Inclusive & Exclusive)
Intra-warp prefix sum and prefix max using Kogge-Stone scan pattern via `__shfl_up_sync`.

```
warp_scan_inclusive_sum<T>(val)      → T
warp_scan_exclusive_sum<T>(val)      → T
warp_scan_inclusive_max<T>(val)      → T
warp_scan_op<T, BinaryOp>(val, op, inclusive) → T
```

#### Module 4: `liftoff/ballot.cuh` — Warp Ballot & Predicate Utilities
All ballot-based operations with full mask support.

```
warp_popcount(predicate)             → int        (# true lanes)
warp_leader_lane(predicate)          → int        (lowest true lane)
warp_my_rank(predicate)              → int        (rank of this lane among true lanes)
warp_compact_indices(predicate, *out)→ int        (lane indices of true lanes → register array)
warp_predicated_broadcast<T>(val, src_predicate) → T
warp_coalesced_group(predicate)      → coalesced_threads group
```

**Key application:** Implementing predicated execution, dynamic warp compaction for sparse attention, and conditional reductions without branch divergence overhead.

#### Module 5: `liftoff/broadcast.cuh` — Warp Broadcast & Exchange
```
warp_broadcast<T>(val, src_lane)     → T          (single source)
warp_rotate<T>(val, delta)           → T          (cyclic shift)
warp_reverse<T>(val)                 → T          (lane 0↔31, 1↔30...)
warp_transpose_4x8<T>(val)          → T          (4×8 matrix in registers)
warp_butterfly_exchange<T>(val, stage) → T        (one stage of FFT/sort network)
```

#### Module 6: `liftoff/sort.cuh` — Warp-Level Bitonic Sort
Full 32-element bitonic sort entirely in registers using shuffle-based compare-and-swap. No shared memory. Sorts 32 values per warp in O(log²N) shuffle rounds.

```
warp_sort_ascending<T>(val)          → T          (each lane holds sorted position)
warp_sort_descending<T>(val)         → T
warp_sort_pairs<K,V>(key, val)       → (K,V)      (key-value pair sort)
warp_odd_even_sort<T>(val)           → T          (alternate network)
```

#### Module 7: `liftoff/topk.cuh` — Warp Top-K
Register-only top-k selection using sorted shuffle networks. Supports k ≤ 32.

```
warp_top1<T>(val)                    → T          (max, equivalent to reduce_max)
warp_topk<T, K>(val)                 → T[K]       (K largest, K≤warp_size)
warp_bottom_k<T, K>(val)            → T[K]
warp_topk_with_indices<T,K>(val)    → (T[K], int[K])
```

**Primary use case:** Top-k logit sampling in LLM autoregressive decoding without global sort.

#### Module 8: `liftoff/cooperative.cuh` — WarpTile<N> Abstraction
Wraps `cg::tiled_partition<N>` with LIFTOFF ergonomics. Enables sub-warp (4, 8, 16) and multi-warp compositions.

```cpp
template<int TileSize>
struct WarpTile {
    cg::thread_block_tile<TileSize> group;
    
    T reduce_sum(T val);
    T scan_sum(T val);
    T broadcast(T val, int src);
    void sort(T& val);
    void barrier();
    int lane_id();
    int tile_rank();
};

// Multi-warp reduce using block-level cooperative groups
template<int BlockSize>
T block_reduce_via_warps(T val, T* warp_results);  // uses WarpTile internally
```

#### Module 9: `liftoff/ml_kernels.cuh` — Application Layer Kernels
End-to-end ML kernels implemented using LIFTOFF primitives only:

```
warp_softmax<T>(x[], out[], len)     → void       (numerically stable, warp-scoped)
warp_layernorm<T>(x[], gamma, beta, out[], len) → void
warp_dot_product<T>(a[], b[], len)   → T
warp_attention_score_reduce(q, k, v, out, seq_len) → void
warp_rms_norm<T>(x[], weight[], out[], len) → void
warp_gelu_fused_reduce<T>(x[], len) → T
```

---

## 6. Performance Requirements

| Primitive | Baseline (shared mem) | LIFTOFF Target | Metric |
|---|---|---|---|
| Warp reduce (float, 32 elem) | ~25 cycles | ≤ 10 cycles | Clock cycles |
| Warp scan (float, 32 elem) | ~40 cycles | ≤ 16 cycles | Clock cycles |
| Warp sort (32 elem) | N/A (no shared baseline) | ≤ 5μs (T4) | Kernel time |
| Warp top-8 | shared+thrust | 2–3× faster | Throughput |
| Softmax (seq=2048, d=64) | cuDNN baseline | ≥ 85% cuDNN | TFLOPS |
| Occupancy gain vs __shared__ | baseline | +20–40% | Warps/SM |

---

## 7. Kaggle Execution Environment

### 7.1 Notebook Architecture
All experiments run as Kaggle notebooks (Python 3.10, CUDA 12.x, GPU accelerator enabled).

```
Kaggle GPU Options:
  - NVIDIA Tesla T4 (15GB VRAM, SM 7.5, Turing) ← primary target
  - NVIDIA P100 (16GB, SM 6.0, Pascal)           ← fallback
  - NVIDIA A100 (40GB, SM 8.0, Ampere)           ← bonus benchmarks
```

### 7.2 Build System
CUDA C++ kernels are compiled inline using `nvcc` (available on Kaggle) via subprocess or PyCUDA's `SourceModule`. The library is header-only — no Makefile or CMake required.

```python
# Kaggle build strategy
import subprocess
result = subprocess.run(
    ["nvcc", "-O3", "-arch=sm_75", "--use_fast_math",
     "-I./liftoff", "benchmark.cu", "-o", "benchmark"],
    capture_output=True
)
```

### 7.3 Profiling on Kaggle
Since Nsight GUI is unavailable, profiling uses:
- `nvprof` CLI (available on Kaggle)
- CUDA Events for microsecond timing
- `cudaDeviceGetAttribute` for SM/warp metadata
- Custom profiling macros in `liftoff/profile.cuh`

---

## 8. Benchmark Suite Specification

### 8.1 Micro-benchmarks
Each primitive is benchmarked in isolation:
- 1000 warm-up iterations + 10,000 timed iterations
- CUDA Event timing (microsecond precision)
- Occupancy measured via `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
- Results written to CSV for Python plotting

### 8.2 Application Benchmarks
| Benchmark | Input Size | Comparison |
|---|---|---|
| Softmax | [1024, 2048, 4096] × [64, 128, 256] | cuDNN / PyTorch |
| LayerNorm | [512, 1024] × [768, 1024, 2048] | Apex / PyTorch |
| Top-K Sampling | batch=32, vocab=32000, k=[4,8,16] | Thrust sort |
| Attention Score | seq=[128,512], heads=8, d_head=64 | FlashAttention-1 |

### 8.3 Occupancy Comparison
A dedicated benchmark measures warp occupancy (active warps / max warps per SM) for:
- Baseline kernel using `__shared__`
- LIFTOFF equivalent kernel
- Theoretical maximum (hardware limit)

---

## 9. Testing Requirements

### 9.1 Correctness Tests
- All primitives validated against CPU reference implementations
- Numerical tolerance: `float` → 1e-5 relative, `__half` → 1e-2 relative
- Edge cases: all-zero input, all-same input, NaN/Inf handling, single active lane

### 9.2 Hardware Compatibility Tests
- Compute capability 7.0 (Volta) — minimum
- Compute capability 7.5 (Turing / T4) — primary
- Compute capability 8.0 (Ampere / A100) — extended

### 9.3 Divergence Tests
- Intentionally divergent warp (different predicates per lane)
- Partial-mask execution (not all 32 lanes active)
- Sub-warp tile correctness (`WarpTile<8>`, `WarpTile<16>`)

---

## 10. Future Roadmap

| Phase | Feature | Timeline |
|---|---|---|
| **v1.1** | `__half2` vectorized shuffle primitives | +4 weeks |
| **v1.2** | Multi-warp NCCL-style reduce across SM | +8 weeks |
| **v1.3** | Persistent warp kernel pattern (producer-consumer with shuffle handoff) | +12 weeks |
| **v2.0** | Triton-based frontend generating LIFTOFF-equivalent PTX | +6 months |
| **v2.1** | bfloat16 + FP8 native shuffle support (Hopper H100) | +6 months |
| **v2.2** | LIFTOFF for AMD RDNA3 via HIP translation layer | +9 months |
| **v3.0** | JIT compilation with cuBIN caching for deployment | +12 months |

---

## 11. Success Criteria

| Criterion | Definition of Done |
|---|---|
| All 9 modules implemented | Header files present, no compilation errors |
| All micro-benchmarks pass correctness | CPU vs GPU delta within tolerance |
| Occupancy improvement demonstrated | ≥15% more warps/SM vs shared-memory baseline |
| 4 ML kernels implemented | Softmax, LayerNorm, Top-K, Dot Product |
| Full Kaggle notebook runnable | One-click execution, no local setup |
| Benchmark CSV + plots generated | matplotlib output embedded in notebook |

---

## 12. Glossary

| Term | Definition |
|---|---|
| **Warp** | 32 threads that execute in lockstep on a CUDA SM |
| **SIMT** | Single Instruction Multiple Threads — NVIDIA's execution model |
| **Shuffle Intrinsic** | Hardware instruction to exchange registers between lanes without memory |
| **Ballot** | Warp-wide predicate aggregation returning a 32-bit bitmask |
| **Lane** | A single thread's position (0–31) within a warp |
| **Mask** | 32-bit integer indicating which lanes participate in an intrinsic |
| **Occupancy** | Ratio of active warps to maximum possible warps on an SM |
| **Cooperative Groups** | CUDA API for expressing thread synchronization at arbitrary granularity |
| **Register File** | On-chip storage private to each thread, zero-latency reads |
| **SM** | Streaming Multiprocessor — the fundamental compute unit of a CUDA GPU |

---

*LIFTOFF PRD v1.0 — Maaran | ML Systems Engineering | GIET MTech CSE 2026–2028*
