// liftoff/cooperative/warptile.cuh
// LIFTOFF: Module 8 — WarpTile<N> via Cooperative Groups
#pragma once
#include <cooperative_groups.h>
#include "../primitives/reduce.cuh"
#include "../primitives/scan.cuh"
#include "../core/config.cuh"

namespace cg = cooperative_groups;
namespace liftoff {

// Sub-warp tile abstraction (N must be power of 2: 2,4,8,16,32)
template<int TileSize>
struct WarpTile {
    static_assert(TileSize == 2  || TileSize == 4  ||
                  TileSize == 8  || TileSize == 16 || TileSize == 32,
                  "TileSize must be power-of-2 in [2, 32]");

    cg::thread_block_tile<TileSize> group;

    __device__ WarpTile() {
        group = cg::tiled_partition<TileSize>(cg::this_thread_block());
    }

    __device__ int lane()  { return group.thread_rank(); }
    __device__ int size()  { return TileSize; }
    __device__ int tiles_per_warp() { return WARP_SIZE / TileSize; }
    __device__ int tile_rank() { return (threadIdx.x & (WARP_SIZE-1)) / TileSize; }
    __device__ void sync() { group.sync(); }

    template<typename T>
    __device__ T reduce_sum(T val) {
        #pragma unroll
        for (int offset = TileSize / 2; offset >= 1; offset >>= 1) {
            val += group.shfl_down(val, offset);
        }
        return val;
    }

    template<typename T>
    __device__ T reduce_max(T val) {
        #pragma unroll
        for (int offset = TileSize / 2; offset >= 1; offset >>= 1) {
            T other = group.shfl_down(val, offset);
            val = val > other ? val : other;
        }
        return val;
    }

    template<typename T>
    __device__ T scan_inclusive_sum(T val) {
        #pragma unroll
        for (int offset = 1; offset < TileSize; offset <<= 1) {
            T tmp = group.shfl_up(val, offset);
            if (lane() >= offset) val += tmp;
        }
        return val;
    }

    template<typename T>
    __device__ T broadcast(T val, int src_lane) {
        return group.shfl(val, src_lane);
    }

    template<typename T>
    __device__ T shfl_xor(T val, int lane_mask) {
        return group.shfl_xor(val, lane_mask);
    }

    // Sort values within the tile (bitonic network at tile granularity)
    template<typename T>
    __device__ void sort(T& val) {
        int lid = lane();
        // Bitonic sort for TileSize elements
        for (int k = 2; k <= TileSize; k <<= 1) {
            for (int j = k >> 1; j >= 1; j >>= 1) {
                int partner_lane = lid ^ j;
                T other = group.shfl(val, partner_lane);
                bool ascending = ((lid & k) == 0);
                bool want_swap = ascending ? (val > other && lid > partner_lane) ||
                                             (val < other && lid < partner_lane)
                                           : (val < other && lid > partner_lane) ||
                                             (val > other && lid < partner_lane);
                if (want_swap) val = other;
            }
        }
    }

    // Barrier — alias for sync()
    __device__ void barrier() { group.sync(); }

    // Reduce min within tile
    template<typename T>
    __device__ T reduce_min(T val) {
        #pragma unroll
        for (int offset = TileSize / 2; offset >= 1; offset >>= 1) {
            T other = group.shfl_down(val, offset);
            val = val < other ? val : other;
        }
        return val;
    }

    // Exclusive prefix sum within tile
    template<typename T>
    __device__ T scan_exclusive_sum(T val) {
        T incl = scan_inclusive_sum(val);
        T excl = group.shfl_up(incl, 1);
        if (lane() == 0) excl = static_cast<T>(0);
        return excl;
    }
};

// Block reduce using WarpTile hierarchy (hybrid: shuffle + minimal shared)
template<int BlockSize, typename T>
__device__ T block_reduce_sum_hybrid(T val) {
    val = warp_reduce_sum(val);

    static __shared__ T warp_sums[BlockSize / WARP_SIZE];
    int wid = warp_id();
    int lid = lane_id();

    if (lid == 0) warp_sums[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (lid < BlockSize / WARP_SIZE) ? warp_sums[lid] : static_cast<T>(0);
        val = warp_reduce_sum(val);
    }
    return val;
}

// Grid-level cooperative reduce
template<typename T>
__device__ T grid_reduce_sum(T val, T* workspace) {
    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    val = block_reduce_sum_hybrid<256, T>(val);

    if (threadIdx.x == 0) workspace[blockIdx.x] = val;
    grid.sync();

    if (blockIdx.x == 0) {
        T sum = (threadIdx.x < gridDim.x) ? workspace[threadIdx.x] : static_cast<T>(0);
        sum = warp_reduce_sum(sum);
        if (threadIdx.x == 0) workspace[0] = sum;
    }
    grid.sync();
    return workspace[0];
}

} // namespace liftoff
