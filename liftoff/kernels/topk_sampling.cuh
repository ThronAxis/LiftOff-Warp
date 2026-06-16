// liftoff/kernels/topk_sampling.cuh
// LIFTOFF: Module 9 — Fused Softmax + Top-K (LLM logit sampling)
#pragma once
#include "../primitives/reduce.cuh"
#include "../primitives/topk.cuh"
#include "../core/types.cuh"

namespace liftoff {

template<int K>
__global__ void fused_softmax_topk_kernel(
    const float* logits, float* probs, int* top_indices,
    int vocab_size)
{
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    int lid = lane_id();
    const float* row_logits = logits + row * vocab_size;

    // 1. Max for numerical stability
    float max_v = -1e38f;
    for (int i = lid; i < vocab_size; i += WARP_SIZE)
        max_v = fmaxf(max_v, row_logits[i]);
    max_v = warp_reduce_max(max_v);

    // 2. Exp-sum
    float exp_sum = 0.f;
    for (int i = lid; i < vocab_size; i += WARP_SIZE)
        exp_sum += __expf(row_logits[i] - max_v);
    exp_sum = warp_reduce_sum(exp_sum);

    // 3. Softmax + streaming top-K
    WarpTopKBuffer<float, K> buf;
    buf.init();
    for (int i = lid; i < vocab_size; i += WARP_SIZE) {
        float p = __expf(row_logits[i] - max_v) / exp_sum;
        buf.insert(p, i);
    }
}

// Warp RMS Norm kernel
template<typename T>
__global__ void warp_rmsnorm_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T*       __restrict__ out,
    int rows, int cols, float epsilon = 1e-6f)
{
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (row >= rows) return;
    int lid = lane_id();
    const T* xr = x   + row * cols;
    T*       or_ = out + row * cols;

    T ss = static_cast<T>(0);
    for (int col = lid; col < cols; col += WARP_SIZE) ss += xr[col] * xr[col];
    ss = warp_reduce_sum(ss);
    T inv = rsqrtf(ss / static_cast<T>(cols) + epsilon);

    for (int col = lid; col < cols; col += WARP_SIZE)
        or_[col] = xr[col] * inv * weight[col];
}

} // namespace liftoff
