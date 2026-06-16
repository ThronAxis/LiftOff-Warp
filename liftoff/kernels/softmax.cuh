// liftoff/kernels/softmax.cuh
// LIFTOFF: Module 9 — Warp Softmax (numerically stable, zero shared memory)
#pragma once
#include "../primitives/reduce.cuh"
#include "../core/types.cuh"

namespace liftoff {

template<typename T>
__global__ void warp_softmax_kernel(
    const T* __restrict__ input,
    T*       __restrict__ output,
    int rows, int cols)
{
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (row >= rows) return;

    int lid = lane_id();
    const T* row_in  = input  + row * cols;
    T*       row_out = output + row * cols;

    // Step 1: Find max (numerical stability)
    T max_val = numeric_limits_device<T>::min_val();
    for (int col = lid; col < cols; col += WARP_SIZE) {
        T v = row_in[col];
        max_val = v > max_val ? v : max_val;
    }
    max_val = warp_reduce_max(max_val);

    // Step 2: Compute sum of exp(x - max)
    T exp_sum = static_cast<T>(0);
    for (int col = lid; col < cols; col += WARP_SIZE) {
        T v = __expf(row_in[col] - max_val);
        row_out[col] = v;
        exp_sum += v;
    }
    exp_sum = warp_reduce_sum(exp_sum);

    // Step 3: Normalize
    T inv_sum = static_cast<T>(1) / exp_sum;
    for (int col = lid; col < cols; col += WARP_SIZE) {
        row_out[col] *= inv_sum;
    }
}

} // namespace liftoff
