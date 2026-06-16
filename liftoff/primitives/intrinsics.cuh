// liftoff/primitives/intrinsics.cuh
// LIFTOFF: Warp-Level Primitives Library
// Module 1: Safe, type-aware wrappers over raw CUDA shuffle/ballot intrinsics
// Eliminates unsigned mask boilerplate. Supports float, double, int, uint32_t.
// Author: Maaran | ML Systems Engineering Research

#pragma once
#include "../core/config.cuh"
#include "../core/types.cuh"

namespace liftoff {

// ─── SHUFFLE DOWN ─────────────────────────────────────────────────────────────
// val from lane (lane_id + delta), clamped at warp boundary

template<typename T>
__device__ __forceinline__ T shfl_down(T val, int delta, unsigned mask = FULL_MASK) {
    return __shfl_down_sync(mask, val, delta);
}

// double specialization (64-bit: must split lo/hi)
template<>
__device__ __forceinline__ double shfl_down<double>(double val, int delta, unsigned mask) {
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_down_sync(mask, lo, delta);
    hi = __shfl_down_sync(mask, hi, delta);
    return __hiloint2double(hi, lo);
}

// ─── SHUFFLE UP ───────────────────────────────────────────────────────────────
template<typename T>
__device__ __forceinline__ T shfl_up(T val, int delta, unsigned mask = FULL_MASK) {
    return __shfl_up_sync(mask, val, delta);
}

template<>
__device__ __forceinline__ double shfl_up<double>(double val, int delta, unsigned mask) {
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_up_sync(mask, lo, delta);
    hi = __shfl_up_sync(mask, hi, delta);
    return __hiloint2double(hi, lo);
}

// ─── SHUFFLE XOR ──────────────────────────────────────────────────────────────
template<typename T>
__device__ __forceinline__ T shfl_xor(T val, int lane_mask, unsigned mask = FULL_MASK) {
    return __shfl_xor_sync(mask, val, lane_mask);
}

template<>
__device__ __forceinline__ double shfl_xor<double>(double val, int lane_mask, unsigned mask) {
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_xor_sync(mask, lo, lane_mask);
    hi = __shfl_xor_sync(mask, hi, lane_mask);
    return __hiloint2double(hi, lo);
}

// ─── SHUFFLE IDX (arbitrary source lane) ──────────────────────────────────────
template<typename T>
__device__ __forceinline__ T shfl_idx(T val, int src_lane, unsigned mask = FULL_MASK) {
    return __shfl_sync(mask, val, src_lane);
}

template<>
__device__ __forceinline__ double shfl_idx<double>(double val, int src_lane, unsigned mask) {
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_sync(mask, lo, src_lane);
    hi = __shfl_sync(mask, hi, src_lane);
    return __hiloint2double(hi, lo);
}

// ─── BALLOT ───────────────────────────────────────────────────────────────────
__device__ __forceinline__ unsigned ballot(bool pred, unsigned mask = FULL_MASK) {
    return __ballot_sync(mask, pred);
}

__device__ __forceinline__ bool warp_any(bool pred, unsigned mask = FULL_MASK) {
    return __any_sync(mask, pred);
}

__device__ __forceinline__ bool warp_all(bool pred, unsigned mask = FULL_MASK) {
    return __all_sync(mask, pred);
}

// match_any: which lanes have same value as mine
template<typename T>
__device__ __forceinline__ unsigned match_any(T val, unsigned mask = FULL_MASK) {
    return __match_any_sync(mask, val);
}

// match_all: do all lanes (in mask) have same value?
template<typename T>
__device__ __forceinline__ unsigned match_all(T val, int* pred, unsigned mask = FULL_MASK) {
    return __match_all_sync(mask, val, pred);
}

} // namespace liftoff
