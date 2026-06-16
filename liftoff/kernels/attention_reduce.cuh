// liftoff/kernels/attention_reduce.cuh
// LIFTOFF: Module 9 — Warp Attention Score Reduction
#pragma once
#include "../primitives/reduce.cuh"
#include "../primitives/intrinsics.cuh"
#include "../core/types.cuh"

namespace liftoff {

// Online softmax for attention (Flash Attention style)
__device__ void warp_online_softmax_update(float& m, float& l, float x) {
    float m_new = fmaxf(m, x);
    l = l * __expf(m - m_new) + __expf(x - m_new);
    m = m_new;
}

__device__ void warp_merge_online_softmax(float& m, float& l) {
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float m2 = shfl_xor(m, delta);
        float l2 = shfl_xor(l, delta);
        float m_new = fmaxf(m, m2);
        l = l * __expf(m - m_new) + l2 * __expf(m2 - m_new);
        m = m_new;
    }
}

// Simplified attention score: Q·K^T / sqrt(d) with warp-level reduction
template<typename T>
__global__ void warp_attention_score_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    T*       __restrict__ scores,
    int seq_len, int d_head)
{
    int q_idx = blockIdx.x;
    int k_idx = blockIdx.y * (blockDim.x / WARP_SIZE) + warp_id();
    if (k_idx >= seq_len) return;

    int lid = lane_id();
    const T* q_vec = Q + q_idx * d_head;
    const T* k_vec = K + k_idx * d_head;

    T acc = static_cast<T>(0);
    for (int d = lid; d < d_head; d += WARP_SIZE) {
        acc += q_vec[d] * k_vec[d];
    }
    acc = warp_reduce_sum(acc);

    if (lid == 0) {
        scores[q_idx * seq_len + k_idx] = acc * rsqrtf(static_cast<float>(d_head));
    }
}

} // namespace liftoff
