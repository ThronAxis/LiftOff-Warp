// liftoff/primitives/ballot.cuh
// LIFTOFF: Warp-Level Primitives Library
// Module 4: Warp ballot & predicate utilities
// Population count, leader election, predicated compaction, masked execution
// Author: Maaran | ML Systems Engineering Research

#pragma once
#include "intrinsics.cuh"
#include "reduce.cuh"
#include "../core/types.cuh"

namespace liftoff {

// ─── POPULATION COUNT ─────────────────────────────────────────────────────────
__device__ __forceinline__ int warp_popcount(bool pred, unsigned mask = FULL_MASK) {
    return __popc(ballot(pred, mask));
}

// ─── LEADER LANE ──────────────────────────────────────────────────────────────
// Returns the lowest-numbered active lane among those where pred is true
__device__ __forceinline__ int warp_leader_lane(bool pred, unsigned mask = FULL_MASK) {
    unsigned active = ballot(pred, mask);
    return active ? __ffs(active) - 1 : -1;   // __ffs: find first set bit (1-indexed)
}

// ─── MY RANK AMONG TRUE LANES ─────────────────────────────────────────────────
// If I am lane 3, 7, 15 and pred is true for all: returns 0, 1, 2 respectively
__device__ __forceinline__ int warp_my_rank(bool pred, unsigned mask = FULL_MASK) {
    unsigned active = ballot(pred, mask);
    // Count set bits below my lane
    unsigned below_me = active & ((1u << lane_id()) - 1u);
    return __popc(below_me);
}

// ─── COMPACT: GATHER ACTIVE LANE IDS INTO REGISTER ARRAY ─────────────────────
// Fills out[] with the lane IDs of lanes where pred is true.
// Returns count of active lanes.
// out[] must be length ≥ 32 (allocated by caller on stack)
__device__ __forceinline__ int warp_compact_lane_ids(bool pred, int* out, unsigned mask = FULL_MASK) {
    unsigned active = ballot(pred, mask);
    int count = 0;
    unsigned tmp = active;
    while (tmp) {
        int lane = __ffs(tmp) - 1;  // lowest set bit
        out[count++] = lane;
        tmp &= tmp - 1;             // clear lowest bit
    }
    return count;
}

// ─── PREDICATED BROADCAST ─────────────────────────────────────────────────────
// Broadcast val from the first lane where pred is true
template<typename T>
__device__ __forceinline__ T warp_predicated_broadcast(T val, bool pred, unsigned mask = FULL_MASK) {
    int src = warp_leader_lane(pred, mask);
    if (src < 0) return val;  // no active lane
    return shfl_idx(val, src, mask);
}

// ─── MASKED SUM (sum only over lanes where pred is true) ──────────────────────
template<typename T>
__device__ __forceinline__ T warp_masked_sum(T val, bool pred, unsigned mask = FULL_MASK) {
    // Lanes where pred is false contribute 0
    T contrib = pred ? val : static_cast<T>(0);
    return warp_reduce_sum(contrib, mask);
}

// ─── COALESCED GROUP VIA BALLOT ───────────────────────────────────────────────
// Create a cooperative group of only the active (pred=true) lanes
// Use this for warp-divergent but coalesced execution
__device__ __forceinline__ 
cooperative_groups::coalesced_group warp_coalesced_group(bool pred) {
    namespace cg = cooperative_groups;
    // ballot-based approach for Volta+
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    return cg::coalesced_threads();  // threads currently convergent
}

// ─── LANE EXISTS IN MASK ──────────────────────────────────────────────────────
__device__ __forceinline__ bool lane_active_in_mask(int lane, unsigned mask) {
    return (mask >> lane) & 1u;
}

// ─── BITMASK OPERATIONS ───────────────────────────────────────────────────────
__device__ __forceinline__ unsigned lanes_below(int lane) {
    return (1u << lane) - 1u;
}

__device__ __forceinline__ unsigned lanes_above(int lane) {
    return ~((1u << (lane + 1)) - 1u);
}

__device__ __forceinline__ unsigned lanes_from_to(int lo, int hi) {
    // Inclusive [lo, hi]
    return ((1u << (hi - lo + 1)) - 1u) << lo;
}

// ─── WARP ELECT ONE ───────────────────────────────────────────────────────────
// Elects exactly one lane from those where pred is true (the leader)
__device__ __forceinline__ bool warp_elect_one(bool pred, unsigned mask = FULL_MASK) {
    int leader = warp_leader_lane(pred, mask);
    return (lane_id() == leader);
}

// ─── COUNT LEADING/TRAILING ZEROS IN BALLOT ───────────────────────────────────
__device__ __forceinline__ int warp_count_leading_zeros(bool pred, unsigned mask = FULL_MASK) {
    unsigned active = ballot(pred, mask);
    return __clz(active);
}

__device__ __forceinline__ int warp_count_trailing_zeros(bool pred, unsigned mask = FULL_MASK) {
    unsigned active = ballot(pred, mask);
    return active ? __ffs(active) - 1 : 32;
}

} // namespace liftoff
