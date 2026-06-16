// liftoff/kernels/dot_product.cuh
// LIFTOFF: Module 9 — Warp Dot Product (zero shared memory)
#pragma once
#include "../primitives/reduce.cuh"
#include "../core/types.cuh"

namespace liftoff {

template<typename T>
__device__ T warp_dot_product(const T* a, const T* b, int len) {
    int lid = lane_id();
    T acc = static_cast<T>(0);
    for (int i = lid; i < len; i += WARP_SIZE) acc += a[i] * b[i];
    return warp_reduce_sum(acc);
}

// Batched dot product kernel: one warp per (row_a · row_b) pair
template<typename T>
__global__ void warp_dot_product_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T*       __restrict__ out,
    int num_pairs, int vec_len)
{
    int pair = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (pair >= num_pairs) return;

    T result = warp_dot_product(A + pair * vec_len, B + pair * vec_len, vec_len);
    if (lane_id() == 0) out[pair] = result;
}

} // namespace liftoff
