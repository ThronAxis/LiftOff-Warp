// liftoff/examples/quickstart.cu
// LIFTOFF Quick Start — demonstrates core primitives in a single file
// Build: nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 -I./liftoff examples/quickstart.cu -o quickstart
#include "../liftoff.cuh"
#include <cstdio>

using namespace liftoff;

// ── Example 1: Warp Reduce ──────────────────────────────────────────────────
__global__ void example_reduce() {
    float val = (float)(lane_id() + 1);  // Each lane holds 1..32

    float sum = warp_reduce_sum(val);     // → 528
    float mx  = warp_reduce_max(val);     // → 32
    float mn  = warp_reduce_min(val);     // → 1

    if (lane_id() == 0) {
        printf("  Reduce: sum=%.0f, max=%.0f, min=%.0f\n", sum, mx, mn);
    }
}

// ── Example 2: Prefix Scan ──────────────────────────────────────────────────
__global__ void example_scan() {
    float val = 1.0f;  // Every lane holds 1

    float inclusive = warp_scan_inclusive(val);  // → 1,2,3,...,32
    float exclusive = warp_scan_exclusive(val);  // → 0,1,2,...,31

    if (lane_id() < 5) {
        printf("  Scan lane %d: inclusive=%.0f, exclusive=%.0f\n",
               lane_id(), inclusive, exclusive);
    }
}

// ── Example 3: Bitonic Sort ─────────────────────────────────────────────────
__global__ void example_sort() {
    float val = (float)(32 - lane_id());  // Reverse: 32,31,...,1
    warp_sort_ascending(val);              // → 1,2,...,32

    if (lane_id() < 5 || lane_id() >= 28) {
        printf("  Sort lane %2d: %.0f\n", lane_id(), val);
    }
}

// ── Example 4: WarpTile<8> Sub-Warp ─────────────────────────────────────────
__global__ void example_warptile() {
    WarpTile<8> tile;
    float val = (float)(tile.lane() + 1);  // 1..8 per tile

    float tile_sum = tile.reduce_sum(val);  // 36 per tile
    float tile_max = tile.reduce_max(val);  //  8 per tile

    if (tile.lane() == 0) {
        printf("  Tile %d: sum=%.0f, max=%.0f\n",
               tile.tile_rank(), tile_sum, tile_max);
    }
}

// ── Example 5: Ballot Operations ────────────────────────────────────────────
__global__ void example_ballot() {
    bool even = (lane_id() % 2 == 0);

    int count  = warp_popcount(even);       // → 16
    int leader = warp_leader_lane(even);    // → 0
    int rank   = warp_my_rank(even);        // → 0,1,2,...,15 for even lanes

    if (lane_id() == 0) {
        printf("  Ballot: %d even lanes, leader=%d\n", count, leader);
    }
    if (even && lane_id() < 8) {
        printf("  Ballot lane %d: rank=%d\n", lane_id(), rank);
    }
}

// ── Example 6: Online Softmax Kernel ────────────────────────────────────────
__global__ void example_softmax(const float* input, float* output) {
    float val = input[lane_id()];

    // Step 1: Find max for numerical stability
    float max_val = warp_reduce_max(val);

    // Step 2: Compute exp(x - max) and sum
    float exp_val = __expf(val - max_val);
    float exp_sum = warp_reduce_sum(exp_val);

    // Step 3: Normalize
    output[lane_id()] = exp_val / exp_sum;
}

int main() {
    print_device_info();
    printf("\n═══ LIFTOFF Quick Start Examples ═══════════════════════════\n\n");

    printf("── Example 1: Warp Reduce ──\n");
    example_reduce<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n── Example 2: Prefix Scan ──\n");
    example_scan<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n── Example 3: Bitonic Sort ──\n");
    example_sort<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n── Example 4: WarpTile<8> ──\n");
    example_warptile<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n── Example 5: Ballot ──\n");
    example_ballot<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n── Example 6: Softmax ──\n");
    float h_in[32], h_out[32];
    float *d_in, *d_out;
    for (int i = 0; i < 32; i++) h_in[i] = (float)i * 0.1f;
    cudaMalloc(&d_in, 32 * sizeof(float));
    cudaMalloc(&d_out, 32 * sizeof(float));
    cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice);
    example_softmax<<<1, 32>>>(d_in, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0;
    for (int i = 0; i < 32; i++) sum += h_out[i];
    printf("  Softmax: output[0]=%.4f, output[31]=%.4f, sum=%.4f\n",
           h_out[0], h_out[31], sum);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  All examples complete! See liftoff.cuh for full API.\n");
    printf("═══════════════════════════════════════════════════════════\n");
    return 0;
}
