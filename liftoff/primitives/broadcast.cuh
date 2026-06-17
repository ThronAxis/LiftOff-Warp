// liftoff/primitives/broadcast.cuh
// LIFTOFF: Module 5 — Broadcast & Exchange
#pragma once
#include "intrinsics.cuh"

namespace liftoff {

template<typename T>
__device__ __forceinline__ T warp_broadcast(T val, int src_lane, unsigned mask = FULL_MASK) {
    return shfl_idx(val, src_lane, mask);
}

template<typename T>
__device__ __forceinline__ T warp_rotate_up(T val, int delta, unsigned mask = FULL_MASK) {
    int src = (lane_id() - delta + WARP_SIZE) & (WARP_SIZE - 1);
    return shfl_idx(val, src, mask);
}

template<typename T>
__device__ __forceinline__ T warp_rotate_down(T val, int delta, unsigned mask = FULL_MASK) {
    int src = (lane_id() + delta) & (WARP_SIZE - 1);
    return shfl_idx(val, src, mask);
}

template<typename T>
__device__ __forceinline__ T warp_reverse(T val, unsigned mask = FULL_MASK) {
    return shfl_idx(val, WARP_SIZE - 1 - lane_id(), mask);
}

template<typename T>
__device__ __forceinline__ T warp_transpose_4x8(T val, unsigned mask = FULL_MASK) {
    val = shfl_xor(val, 0x01, mask);
    val = shfl_xor(val, 0x08, mask);
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_butterfly_stage(T val, int stage_bit, unsigned mask = FULL_MASK) {
    return shfl_xor(val, stage_bit, mask);
}

template<typename T>
__device__ __forceinline__ T warp_zip(T val, unsigned mask = FULL_MASK) {
    int lid = lane_id();
    int src = (lid < 16) ? lid * 2 : (lid - 16) * 2 + 1;
    return shfl_idx(val, src, mask);
}

template<typename T>
__device__ __forceinline__ T warp_unzip(T val, unsigned mask = FULL_MASK) {
    int lid = lane_id();
    int src = (lid & 1) ? 16 + (lid >> 1) : (lid >> 1);
    return shfl_idx(val, src, mask);
}

// ─── EXCHANGE UTILITIES ──────────────────────────────────────────────────────

// Swap values between two specific lanes
template<typename T>
__device__ __forceinline__ T warp_swap(T val, int partner_lane, unsigned mask = FULL_MASK) {
    return shfl_idx(val, partner_lane, mask);
}

// Exchange with adjacent lane (even↔odd)
template<typename T>
__device__ __forceinline__ T warp_exchange_adjacent(T val, unsigned mask = FULL_MASK) {
    return shfl_xor(val, 1, mask);
}

// Gather: each lane reads from an index specified per-lane
template<typename T>
__device__ __forceinline__ T warp_gather(T val, int src_lane, unsigned mask = FULL_MASK) {
    return shfl_idx(val, src_lane & (WARP_SIZE - 1), mask);
}

// Shift left (discard top, fill bottom with identity)
template<typename T>
__device__ __forceinline__ T warp_shift_left(T val, int delta, T fill = T(0), unsigned mask = FULL_MASK) {
    T shifted = shfl_down(val, delta, mask);
    return (lane_id() + delta < WARP_SIZE) ? shifted : fill;
}

// Shift right (discard bottom, fill top with identity)
template<typename T>
__device__ __forceinline__ T warp_shift_right(T val, int delta, T fill = T(0), unsigned mask = FULL_MASK) {
    T shifted = shfl_up(val, delta, mask);
    return (lane_id() >= delta) ? shifted : fill;
}

} // namespace liftoff

