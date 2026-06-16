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

} // namespace liftoff
