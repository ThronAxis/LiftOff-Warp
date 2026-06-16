// liftoff/primitives/reduce.cuh
// LIFTOFF: Warp-Level Primitives Library
// Module 2: Warp-level reductions using butterfly (XOR shuffle) pattern
// O(log₂ 32) = 5 shuffle instructions, no sync barriers, zero shared memory
// Author: Maaran | ML Systems Engineering Research

#pragma once
#include "intrinsics.cuh"

namespace liftoff {

// ─── GENERIC REDUCE ───────────────────────────────────────────────────────────
// Butterfly (XOR) pattern: O(log2(32)) = 5 shuffle rounds

template<typename T, typename BinaryOp>
__device__ __forceinline__ T warp_reduce(T val, BinaryOp op, unsigned mask = FULL_MASK) {
    // Unrolled 5 stages for warp of 32
    val = op(val, shfl_xor(val, 16, mask));
    val = op(val, shfl_xor(val,  8, mask));
    val = op(val, shfl_xor(val,  4, mask));
    val = op(val, shfl_xor(val,  2, mask));
    val = op(val, shfl_xor(val,  1, mask));
    return val;
    // Result: all lanes hold the reduced value
}

// Specializations for common operators

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val, unsigned mask = FULL_MASK) {
    return warp_reduce(val, [](T a, T b){ return a + b; }, mask);
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val, unsigned mask = FULL_MASK) {
    return warp_reduce(val, [](T a, T b){ return a > b ? a : b; }, mask);
}

template<typename T>
__device__ __forceinline__ T warp_reduce_min(T val, unsigned mask = FULL_MASK) {
    return warp_reduce(val, [](T a, T b){ return a < b ? a : b; }, mask);
}

template<typename T>
__device__ __forceinline__ T warp_reduce_prod(T val, unsigned mask = FULL_MASK) {
    return warp_reduce(val, [](T a, T b){ return a * b; }, mask);
}

// Bitwise reductions for integer types
__device__ __forceinline__ unsigned warp_reduce_and(unsigned val, unsigned mask = FULL_MASK) {
    return warp_reduce(val, [](unsigned a, unsigned b){ return a & b; }, mask);
}

__device__ __forceinline__ unsigned warp_reduce_or(unsigned val, unsigned mask = FULL_MASK) {
    return warp_reduce(val, [](unsigned a, unsigned b){ return a | b; }, mask);
}

// ─── ARGMAX / ARGMIN ──────────────────────────────────────────────────────────
// Returns (max_val, lane_of_max) for argmax operations

template<typename T>
__device__ __forceinline__ void warp_reduce_argmax(T& val, int& idx, unsigned mask = FULL_MASK) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        T   other_val = shfl_down(val, delta, mask);
        int other_idx = shfl_down(idx, delta, mask);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    // Broadcast from lane 0
    val = shfl_idx(val, 0, mask);
    idx = shfl_idx(idx, 0, mask);
}

template<typename T>
__device__ __forceinline__ void warp_reduce_argmin(T& val, int& idx, unsigned mask = FULL_MASK) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        T   other_val = shfl_down(val, delta, mask);
        int other_idx = shfl_down(idx, delta, mask);
        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
    // Broadcast from lane 0
    val = shfl_idx(val, 0, mask);
    idx = shfl_idx(idx, 0, mask);
}

// ─── PARTIAL WARP REDUCE (fewer than 32 active lanes) ─────────────────────────
// Reduces only threads active in given mask

template<typename T, typename BinaryOp>
__device__ __forceinline__ T partial_warp_reduce(T val, BinaryOp op, unsigned active_mask) {
    return warp_reduce(val, op, active_mask);
}

// ─── HALF2 VECTORIZED SUM ─────────────────────────────────────────────────────
#ifdef __CUDA_ARCH__
__device__ __forceinline__ __half2 warp_reduce_sum_half2(__half2 val, unsigned mask = FULL_MASK) {
    val = __hadd2(val, __shfl_xor_sync(mask, val, 16));
    val = __hadd2(val, __shfl_xor_sync(mask, val,  8));
    val = __hadd2(val, __shfl_xor_sync(mask, val,  4));
    val = __hadd2(val, __shfl_xor_sync(mask, val,  2));
    val = __hadd2(val, __shfl_xor_sync(mask, val,  1));
    return val;
}
#endif

// ─── SEGMENTED REDUCE ─────────────────────────────────────────────────────────
// Reduce within segments defined by a head-flag (flag=true starts a new segment)
template<typename T>
__device__ __forceinline__ T warp_segmented_reduce_sum(T val, bool head_flag, unsigned mask = FULL_MASK) {
    unsigned heads = ballot(head_flag, mask);
    
    for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
        T other = shfl_up(val, delta, mask);
        // Only add if no segment boundary between me and the other lane
        unsigned between_mask = ((1u << lane_id()) - 1u) & ~((1u << (lane_id() - delta)) - 1u);
        if (lane_id() >= delta && (heads & between_mask) == 0) {
            val += other;
        }
    }
    return val;
}

} // namespace liftoff
