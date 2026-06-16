// liftoff/kernels/compositions.cuh
// LIFTOFF: Composition Recipes (MD §16)
// Fused multi-primitive kernels demonstrating composability
#pragma once
#include "../primitives/reduce.cuh"
#include "../primitives/scan.cuh"
#include "../primitives/topk.cuh"
#include "../primitives/intrinsics.cuh"
#include "attention_reduce.cuh"

namespace liftoff {

// ═══════════ Recipe 1: Online Softmax (Flash Attention style) ═══════════
// Uses warp_online_softmax_update() and warp_merge_online_softmax()
// defined in attention_reduce.cuh


// ═══════════ Recipe 2: Warp Histogram via match_any ═══════════
__device__ void warp_histogram_recipe(
    int val, int* hist_out, int n_bins)
{
    for (int bin = 0; bin < n_bins; bin++) {
        unsigned matches = __match_any_sync(FULL_MASK, val == bin ? 1 : 0);
        if (val == bin && lane_id() == __ffs(matches) - 1) {
            hist_out[bin] = __popc(matches);
        }
    }
}

// ═══════════ Recipe 3: Fused LayerNorm + GELU ═══════════
template<typename T>
__global__ void fused_layernorm_gelu_kernel(
    const T* __restrict__ input,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T*       __restrict__ output,
    int rows, int cols, float eps = 1e-5f)
{
    int row = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (row >= rows) return;
    int lid = lane_id();
    const T* x = input + row * cols;
    T* out = output + row * cols;

    // Mean
    T sum = static_cast<T>(0);
    for (int c = lid; c < cols; c += WARP_SIZE) sum += x[c];
    sum = warp_reduce_sum(sum);
    T mean = sum / static_cast<T>(cols);

    // Variance
    T var = static_cast<T>(0);
    for (int c = lid; c < cols; c += WARP_SIZE) {
        T d = x[c] - mean; var += d * d;
    }
    var = warp_reduce_sum(var);
    T inv_std = rsqrtf(var / static_cast<T>(cols) + eps);

    // Normalize + GELU
    const float kS = 0.7978845608f;
    const float kC = 0.044715f;
    for (int c = lid; c < cols; c += WARP_SIZE) {
        T normed = (x[c] - mean) * inv_std * gamma[c] + beta[c];
        float xf = (float)normed;
        float gelu = 0.5f * xf * (1.0f + tanhf(kS * (xf + kC * xf * xf * xf)));
        out[c] = (T)gelu;
    }
}

// ═══════════ Recipe 4: Warp-Parallel Prefix + Scatter ═══════════
// Compaction: given predicate, pack active elements contiguously
template<typename T>
__device__ int warp_stream_compact(
    T val, bool pred, T* out, unsigned mask = FULL_MASK)
{
    int rank = warp_my_rank(pred, mask);
    int total = warp_popcount(pred, mask);
    if (pred) out[rank] = val;
    return total;
}

} // namespace liftoff
