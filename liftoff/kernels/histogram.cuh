// liftoff/kernels/histogram.cuh
// LIFTOFF: Warp-Level Histogram via match_any (PRD G1 requirement)
// Uses __match_any_sync — O(N_BINS) shuffles vs O(N) atomic ops
#pragma once
#include "../primitives/intrinsics.cuh"
#include "../primitives/ballot.cuh"
#include "../core/types.cuh"

namespace liftoff {

// ─── WARP HISTOGRAM (small bin count) ─────────────────────────────────────────
// Each lane holds a value in [0, n_bins). Counts occurrences per bin across warp.
// hist_out must be ≥ n_bins elements (only lane 0 writes).
__device__ __forceinline__ void warp_histogram(
    int val, int* hist_out, int n_bins, unsigned mask = FULL_MASK)
{
    int lid = lane_id();
    for (int bin = 0; bin < n_bins; bin++) {
        bool is_match = (val == bin);
        int count = warp_popcount(is_match, mask);
        // Only lane 0 writes (or the elected leader)
        if (lid == 0) {
            hist_out[bin] = count;
        }
    }
}

// ─── WARP HISTOGRAM KERNEL ────────────────────────────────────────────────────
// One warp per histogram batch. Each warp processes 32 elements.
__global__ void warp_histogram_kernel(
    const int* __restrict__ input,
    int*       __restrict__ output,
    int num_batches, int n_bins)
{
    int batch = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id();
    if (batch >= num_batches) return;

    int val = input[batch * WARP_SIZE + lane_id()];
    int* batch_hist = output + batch * n_bins;

    warp_histogram(val, batch_hist, n_bins);
}

// ─── WARP HISTOGRAM ACCUMULATE (with global atomics for multi-warp) ───────────
// Multiple warps accumulate into a shared histogram using warp-level counts
// + atomicAdd. Minimizes atomics: only 1 atomic per bin per warp.
__global__ void warp_histogram_atomic_kernel(
    const int* __restrict__ input,
    int*       __restrict__ global_hist,
    int N, int n_bins)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (tid < N) ? input[tid] : -1;

    for (int bin = 0; bin < n_bins; bin++) {
        bool match = (val == bin);
        int count = warp_popcount(match);
        // Only the warp leader does the atomic
        if (lane_id() == 0 && count > 0) {
            atomicAdd(&global_hist[bin], count);
        }
    }
}

} // namespace liftoff
