// liftoff/core/types.cuh
// LIFTOFF: Warp-Level Primitives Library
// Core types: lane utilities, numeric limits for device-side operations
// Author: Maaran | ML Systems Engineering Research

#pragma once
#include "config.cuh"
#include <cuda_fp16.h>

namespace liftoff {

// Lane ID (0–31) for current thread
__device__ __forceinline__ int lane_id() {
    return threadIdx.x & 31;
}

// Warp ID within the block
__device__ __forceinline__ int warp_id() {
    return threadIdx.x >> 5;
}

// Predicate → lane bitmask
__device__ __forceinline__ unsigned pred_to_mask(bool pred) {
    return __ballot_sync(FULL_MASK, pred);
}

// Numeric identity elements for reductions
template<typename T> struct numeric_limits_device {
    __device__ static T max_val();
    __device__ static T min_val();
    __device__ static T zero();
    __device__ static T one();
};

template<> struct numeric_limits_device<float> {
    __device__ static float max_val() { return 3.402823466e+38f; }
    __device__ static float min_val() { return -3.402823466e+38f; }
    __device__ static float zero()    { return 0.0f; }
    __device__ static float one()     { return 1.0f; }
};

template<> struct numeric_limits_device<int> {
    __device__ static int max_val() { return 2147483647; }
    __device__ static int min_val() { return -2147483648; }
    __device__ static int zero()    { return 0; }
    __device__ static int one()     { return 1; }
};

template<> struct numeric_limits_device<double> {
    __device__ static double max_val() { return 1.7976931348623158e+308; }
    __device__ static double min_val() { return -1.7976931348623158e+308; }
    __device__ static double zero()    { return 0.0; }
    __device__ static double one()     { return 1.0; }
};

template<> struct numeric_limits_device<unsigned> {
    __device__ static unsigned max_val() { return 0xFFFFFFFFu; }
    __device__ static unsigned min_val() { return 0u; }
    __device__ static unsigned zero()    { return 0u; }
    __device__ static unsigned one()     { return 1u; }
};

// Key-Value pair for sorted operations
template<typename K, typename V>
struct KeyValuePair {
    K key;
    V value;
    __device__ __forceinline__ bool operator<(const KeyValuePair& other) const {
        return key < other.key;
    }
    __device__ __forceinline__ bool operator>(const KeyValuePair& other) const {
        return key > other.key;
    }
};

} // namespace liftoff
