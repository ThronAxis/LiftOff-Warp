// liftoff/primitives/sort.cuh
// LIFTOFF: Module 6 — Warp Bitonic Sort (32 elements in registers)
#pragma once
#include "intrinsics.cuh"
#include "broadcast.cuh"

namespace liftoff {

// Bitonic sort ascending — full 32-element network in registers
// O(log²N) = 15 shuffle rounds, zero shared memory
template<typename T>
__device__ __forceinline__ void warp_sort_ascending(T& val, unsigned mask = FULL_MASK) {
    int lid = lane_id();

    #define BITONIC_STEP(stride, dir_bit)                               \
    {                                                                    \
        int partner = lid ^ stride;                                      \
        T   other   = shfl_xor(val, stride, mask);                      \
        bool hi     = (lid & (stride << 1)) != 0;                       \
        bool want_swap = hi ? (val > other) : (val < other);            \
        if (want_swap) val = other;                                      \
    }

    // k=2
    BITONIC_STEP(1, 2)
    // k=4
    BITONIC_STEP(2, 4)
    BITONIC_STEP(1, 4)
    // k=8
    BITONIC_STEP(4, 8)
    BITONIC_STEP(2, 8)
    BITONIC_STEP(1, 8)
    // k=16
    BITONIC_STEP(8, 16)
    BITONIC_STEP(4, 16)
    BITONIC_STEP(2, 16)
    BITONIC_STEP(1, 16)
    // k=32
    BITONIC_STEP(16, 32)
    BITONIC_STEP(8,  32)
    BITONIC_STEP(4,  32)
    BITONIC_STEP(2,  32)
    BITONIC_STEP(1,  32)

    #undef BITONIC_STEP
}

template<typename T>
__device__ __forceinline__ void warp_sort_descending(T& val, unsigned mask = FULL_MASK) {
    warp_sort_ascending(val, mask);
    val = warp_reverse(val, mask);
}

// Key-value pair sort
template<typename K, typename V>
__device__ __forceinline__ void warp_sort_pairs_ascending(K& key, V& val, unsigned mask = FULL_MASK) {
    int lid = lane_id();

    #define BITONIC_PAIR_STEP(stride)                                    \
    {                                                                    \
        K other_key = shfl_xor(key, stride, mask);                      \
        V other_val = shfl_xor(val, stride, mask);                      \
        bool hi = (lid & (stride << 1)) != 0;                           \
        bool want_swap = hi ? (key > other_key) : (key < other_key);    \
        if (want_swap) { key = other_key; val = other_val; }            \
    }

    BITONIC_PAIR_STEP(1)
    BITONIC_PAIR_STEP(2)  BITONIC_PAIR_STEP(1)
    BITONIC_PAIR_STEP(4)  BITONIC_PAIR_STEP(2)  BITONIC_PAIR_STEP(1)
    BITONIC_PAIR_STEP(8)  BITONIC_PAIR_STEP(4)  BITONIC_PAIR_STEP(2)  BITONIC_PAIR_STEP(1)
    BITONIC_PAIR_STEP(16) BITONIC_PAIR_STEP(8)  BITONIC_PAIR_STEP(4)
    BITONIC_PAIR_STEP(2)  BITONIC_PAIR_STEP(1)

    #undef BITONIC_PAIR_STEP
}

} // namespace liftoff
