// liftoff/primitives/scan.cuh
// LIFTOFF: Warp-Level Primitives Library
// Module 3: Warp prefix scan (inclusive & exclusive) using Kogge-Stone pattern
// O(log₂ 32) = 5 stages via __shfl_up_sync, zero shared memory
// Author: Maaran | ML Systems Engineering Research

#pragma once
#include "intrinsics.cuh"

namespace liftoff {

// ─── INCLUSIVE PREFIX SUM (Kogge-Stone, up-sweep with shfl_up) ────────────────
template<typename T>
__device__ __forceinline__ T warp_scan_inclusive(T val, unsigned mask = FULL_MASK) {
    int lid = lane_id();
    T tmp;

    // Stage 1: offset 1
    tmp = shfl_up(val, 1, mask);
    if (lid >= 1) val += tmp;

    // Stage 2: offset 2
    tmp = shfl_up(val, 2, mask);
    if (lid >= 2) val += tmp;

    // Stage 3: offset 4
    tmp = shfl_up(val, 4, mask);
    if (lid >= 4) val += tmp;

    // Stage 4: offset 8
    tmp = shfl_up(val, 8, mask);
    if (lid >= 8) val += tmp;

    // Stage 5: offset 16
    tmp = shfl_up(val, 16, mask);
    if (lid >= 16) val += tmp;

    return val;
    // Lane k holds sum of input[0..k] inclusive
}

// ─── EXCLUSIVE PREFIX SUM ─────────────────────────────────────────────────────
template<typename T>
__device__ __forceinline__ T warp_scan_exclusive(T val, unsigned mask = FULL_MASK) {
    T incl = warp_scan_inclusive(val, mask);
    // Shift right: exclusive[k] = inclusive[k-1], exclusive[0] = 0
    T excl = shfl_up(incl, 1, mask);
    if (lane_id() == 0) excl = static_cast<T>(0);
    return excl;
}

// ─── GENERIC SCAN (arbitrary binary op) ───────────────────────────────────────
template<typename T, typename BinaryOp>
__device__ __forceinline__ T warp_scan_inclusive_op(T val, BinaryOp op, T identity, unsigned mask = FULL_MASK) {
    int lid = lane_id();
    T tmp;

    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        tmp = shfl_up(val, offset, mask);
        if (lid >= offset) val = op(val, tmp);
    }
    return val;
}

// Specialization: inclusive max-scan (running max)
template<typename T>
__device__ __forceinline__ T warp_scan_inclusive_max(T val, unsigned mask = FULL_MASK) {
    return warp_scan_inclusive_op(val, [](T a, T b){ return a > b ? a : b; },
        numeric_limits_device<T>::min_val(), mask);
}

// Specialization: inclusive min-scan (running min)
template<typename T>
__device__ __forceinline__ T warp_scan_inclusive_min(T val, unsigned mask = FULL_MASK) {
    return warp_scan_inclusive_op(val, [](T a, T b){ return a < b ? a : b; },
        numeric_limits_device<T>::max_val(), mask);
}

// ─── WARP PREFIX SUM WITH TOTAL ───────────────────────────────────────────────
// Returns both the exclusive prefix and the warp-total sum
template<typename T>
__device__ __forceinline__ T warp_scan_exclusive_with_total(T val, T& total, unsigned mask = FULL_MASK) {
    T incl = warp_scan_inclusive(val, mask);
    // Warp total = lane 31's inclusive value
    total = shfl_idx(incl, 31, mask);
    // Exclusive: shift right
    T excl = shfl_up(incl, 1, mask);
    if (lane_id() == 0) excl = static_cast<T>(0);
    return excl;
}

// ─── WARP DECOUPLED LOOKBACK (for multi-warp scan composition) ────────────────
// Status flags for decoupled lookback scan across blocks
enum ScanStatus : int {
    SCAN_STATUS_INVALID   = 0,
    SCAN_STATUS_AGGREGATE = 1,
    SCAN_STATUS_PREFIX    = 2
};

} // namespace liftoff
