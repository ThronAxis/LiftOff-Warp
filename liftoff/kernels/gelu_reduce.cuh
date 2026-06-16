// liftoff/kernels/gelu_reduce.cuh
// LIFTOFF: Module 9 — Warp GELU Fused Reduce (PRD requirement)
// Fused GELU activation + warp-level sum reduction, zero shared memory
#pragma once
#include "../primitives/reduce.cuh"
#include "../core/types.cuh"

namespace liftoff {

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
__device__ __forceinline__ float gelu_approx(float x) {
    const float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/π)
    const float kCoeff = 0.044715f;
    float x3 = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Fused GELU + Reduce Sum: apply GELU to each element, then sum across warp
template<typename T>
__device__ T warp_gelu_fused_reduce(const T* input, int len) {
    int lid = lane_id();
    T acc = static_cast<T>(0);
    for (int i = lid; i < len; i += WARP_SIZE) {
        T v = input[i];
        acc += gelu_approx(v);
    }
    return warp_reduce_sum(acc);
}

// GELU activation kernel (element-wise, one warp per row)
template<typename T>
__global__ void warp_gelu_kernel(
    const T* __restrict__ input,
    T*       __restrict__ output,
    int rows, int cols)
{
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (row >= rows) return;

    int lid = lane_id();
    const T* x   = input  + row * cols;
    T*       out = output + row * cols;

    for (int col = lid; col < cols; col += WARP_SIZE) {
        out[col] = gelu_approx(x[col]);
    }
}

// GELU fused reduce kernel — returns per-row sum of GELU(x)
template<typename T>
__global__ void warp_gelu_fused_reduce_kernel(
    const T* __restrict__ input,
    T*       __restrict__ output,
    int rows, int cols)
{
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (row >= rows) return;

    T result = warp_gelu_fused_reduce(input + row * cols, cols);
    if (lane_id() == 0) output[row] = result;
}

} // namespace liftoff
