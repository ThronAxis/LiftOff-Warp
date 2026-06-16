// liftoff/primitives/topk.cuh
// LIFTOFF: Module 7 — Warp Top-K selection (register-only)
#pragma once
#include "reduce.cuh"
#include "sort.cuh"
#include "../core/types.cuh"

namespace liftoff {

// Top-1 (equivalent to warp_reduce_max)
template<typename T>
__device__ __forceinline__ T warp_top1(T val, unsigned mask = FULL_MASK) {
    return warp_reduce_max(val, mask);
}

// Top-K (k ≤ 32) using sort + prefix select
template<typename T, int K>
__device__ __forceinline__ void warp_topk(
    T val, int orig_idx,
    T* out_vals, int* out_idxs,
    unsigned mask = FULL_MASK)
{
    static_assert(K <= WARP_SIZE, "K must be <= 32 for warp-level top-k");

    warp_sort_pairs_ascending(val, orig_idx, mask);
    // After ascending sort, lane 31 = max, lane 0 = min

    int lid = lane_id();
    if (lid == 0) {
        #pragma unroll
        for (int k = 0; k < K; k++) {
            int src_lane = WARP_SIZE - K + k;
            out_vals[k] = shfl_idx(val,      src_lane, mask);
            out_idxs[k] = shfl_idx(orig_idx, src_lane, mask);
        }
    }
}

// Top-K mask: returns ballot of lanes in top-K
template<typename T>
__device__ __forceinline__ unsigned warp_topk_mask(T val, int k, unsigned mask = FULL_MASK) {
    T sorted_val = val;
    warp_sort_ascending(sorted_val, mask);
    T threshold = shfl_idx(sorted_val, WARP_SIZE - k, mask);
    return ballot(val >= threshold, mask);
}

// Streaming Top-K buffer for sequences longer than 32
template<typename T, int K>
struct WarpTopKBuffer {
    T   vals[K];
    int idxs[K];
    int count;

    __device__ void init() {
        #pragma unroll
        for (int i = 0; i < K; i++) {
            vals[i] = numeric_limits_device<T>::min_val();
            idxs[i] = -1;
        }
        count = 0;
    }

    __device__ void insert(T new_val, int new_idx) {
        T min_val = vals[0];
        int min_pos = 0;
        #pragma unroll
        for (int i = 1; i < K; i++) {
            if (vals[i] < min_val) {
                min_val = vals[i];
                min_pos = i;
            }
        }
        if (new_val > min_val) {
            vals[min_pos] = new_val;
            idxs[min_pos] = new_idx;
        }
    }

    __device__ void warp_merge(unsigned mask = FULL_MASK) {
        // Tournament tree merge across warp:
        // Each lane has its local top-K. We merge by shuffling and keeping
        // the best K values across all candidates from pairs of lanes.
        for (int delta = 1; delta < WARP_SIZE; delta <<= 1) {
            // Receive partner's top-K values
            #pragma unroll
            for (int i = 0; i < K; i++) {
                T partner_val   = shfl_xor(vals[i], delta, mask);
                int partner_idx = shfl_xor(idxs[i], delta, mask);
                insert(partner_val, partner_idx);
            }
        }
    }
};

// ─── WARP BOTTOM-K (K smallest values) ────────────────────────────────────────
template<typename T, int K>
__device__ __forceinline__ void warp_bottom_k(
    T val, int orig_idx,
    T* out_vals, int* out_idxs,
    unsigned mask = FULL_MASK)
{
    static_assert(K <= WARP_SIZE, "K must be <= 32 for warp-level bottom-k");

    warp_sort_pairs_ascending(val, orig_idx, mask);
    // After ascending sort, lane 0 = min, lane 31 = max
    // Bottom-K = lanes 0..(K-1)

    int lid = lane_id();
    if (lid == 0) {
        #pragma unroll
        for (int k = 0; k < K; k++) {
            out_vals[k] = shfl_idx(val,      k, mask);
            out_idxs[k] = shfl_idx(orig_idx, k, mask);
        }
    }
}

// ─── WARP TOP-K WITH INDICES (all lanes get results) ──────────────────────────
template<typename T, int K>
__device__ __forceinline__ void warp_topk_with_indices(
    T val, int orig_idx,
    T* out_vals, int* out_idxs,
    unsigned mask = FULL_MASK)
{
    static_assert(K <= WARP_SIZE, "K must be <= 32");

    T sorted_key = val;
    int sorted_idx = orig_idx;
    warp_sort_pairs_ascending(sorted_key, sorted_idx, mask);

    // Broadcast K largest (from lanes WARP_SIZE-K .. WARP_SIZE-1) to ALL lanes
    #pragma unroll
    for (int k = 0; k < K; k++) {
        int src_lane = WARP_SIZE - K + k;
        out_vals[k] = shfl_idx(sorted_key, src_lane, mask);
        out_idxs[k] = shfl_idx(sorted_idx, src_lane, mask);
    }
}

} // namespace liftoff
