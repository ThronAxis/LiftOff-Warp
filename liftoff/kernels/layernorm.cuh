// liftoff/kernels/layernorm.cuh
// LIFTOFF: Module 9 — Warp LayerNorm (zero shared memory)
#pragma once
#include "../primitives/reduce.cuh"
#include "../core/types.cuh"

namespace liftoff {

template<typename T>
__global__ void warp_layernorm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T*       __restrict__ output,
    int rows, int cols, float epsilon = 1e-5f)
{
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (row >= rows) return;

    int lid = lane_id();
    const T* x   = input  + row * cols;
    T*       out = output + row * cols;

    // Step 1: Compute mean
    T sum = static_cast<T>(0);
    for (int col = lid; col < cols; col += WARP_SIZE) sum += x[col];
    sum = warp_reduce_sum(sum);
    T mean = sum / static_cast<T>(cols);

    // Step 2: Compute variance
    T var_sum = static_cast<T>(0);
    for (int col = lid; col < cols; col += WARP_SIZE) {
        T diff = x[col] - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);
    T inv_std = rsqrtf(var_sum / static_cast<T>(cols) + epsilon);

    // Step 3: Normalize + affine transform
    for (int col = lid; col < cols; col += WARP_SIZE) {
        out[col] = (x[col] - mean) * inv_std * gamma[col] + beta[col];
    }
}

} // namespace liftoff
