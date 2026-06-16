// liftoff/bench/benchmark_main.cu
// LIFTOFF: Main benchmark runner — exercises all primitives and ML kernels
// Build: nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 -I./liftoff benchmark_main.cu -o liftoff_bench
#include "../liftoff.cuh"
#include "benchmark.cuh"
#include <cstdlib>
#include <cstring>

using namespace liftoff;

// ── Kernel wrappers for benchmarking ─────────────────────────────────────────

__global__ void bench_warp_reduce(float* data, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? data[tid] : 0.f;
    float result = warp_reduce_sum(val);
    if (lane_id() == 0) out[blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id()] = result;
}

__global__ void bench_warp_scan(float* data, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? data[tid] : 0.f;
    float result = warp_scan_inclusive(val);
    if (tid < N) out[tid] = result;
}

__global__ void bench_warp_sort(float* data, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? data[tid] : 0.f;
    warp_sort_ascending(val);
    if (tid < N) out[tid] = val;
}

__global__ void bench_shared_reduce(float* data, float* out, int N) {
    // Shared-memory baseline for comparison
    extern __shared__ float smem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    smem[lid] = (tid < N) ? data[tid] : 0.f;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        __syncthreads();
    }
    if (lid == 0) out[blockIdx.x] = smem[0];
}

int main() {
    // Print device info
    print_device_info();

    const int N = 1 << 20;   // 1M elements
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Initialize with pattern data
    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = static_cast<float>(i % 100) * 0.01f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  LIFTOFF Benchmark Suite — Warp-Level Primitives\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    BenchResult results[6];

    // Benchmark 1: Warp Reduce Sum
    results[0] = benchmark("warp_reduce_sum (1M floats)",
        [&]{ bench_warp_reduce<<<GRID, BLOCK>>>(d_in, d_out, N); },
        N);

    // Benchmark 2: Shared Memory Reduce (baseline)
    results[1] = benchmark("shared_mem_reduce (1M, BASELINE)",
        [&]{ bench_shared_reduce<<<GRID, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N); },
        N);

    // Benchmark 3: Warp Scan
    results[2] = benchmark("warp_scan_inclusive (1M floats)",
        [&]{ bench_warp_scan<<<GRID, BLOCK>>>(d_in, d_out, N); },
        N * 5);

    // Benchmark 4: Warp Sort
    results[3] = benchmark("warp_sort_ascending (1M floats)",
        [&]{ bench_warp_sort<<<GRID, BLOCK>>>(d_in, d_out, N); },
        N * 15);

    // Benchmark 5: Softmax
    int rows = 4096, cols = 256;
    results[4] = benchmark("warp_softmax (4096x256)",
        [&]{
            warp_softmax_kernel<float><<<rows, BLOCK>>>(d_in, d_out, rows, cols);
        },
        (long long)rows * cols * 4);

    // Benchmark 6: LayerNorm
    int ln_rows = 1024, ln_cols = 768;
    float *d_gamma, *d_beta, *d_ln_in, *d_ln_out;
    CUDA_CHECK(cudaMalloc(&d_gamma,  ln_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta,   ln_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_in,  ln_rows * ln_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ln_out, ln_rows * ln_cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gamma, 0x3F, ln_cols * sizeof(float)));  // ~1.0f
    CUDA_CHECK(cudaMemset(d_beta,  0,    ln_cols * sizeof(float)));

    results[5] = benchmark("warp_layernorm (1024x768)",
        [&]{
            warp_layernorm_kernel<float><<<ln_rows, BLOCK>>>(
                d_ln_in, d_gamma, d_beta, d_ln_out, ln_rows, ln_cols);
        },
        (long long)ln_rows * ln_cols * 5);

    // Occupancy queries
    printf("\n── Occupancy Analysis ──────────────────────────────────────\n");
    LIFTOFF_QUERY_OCCUPANCY(bench_warp_reduce, BLOCK, 0);
    LIFTOFF_QUERY_OCCUPANCY(bench_shared_reduce, BLOCK, BLOCK * sizeof(float));
    LIFTOFF_QUERY_OCCUPANCY(bench_warp_sort, BLOCK, 0);

    // Write CSV
    write_csv("liftoff_bench_results.csv", results, 6);

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_ln_in));
    CUDA_CHECK(cudaFree(d_ln_out));
    delete[] h_in;

    printf("\n✓ LIFTOFF Benchmark complete.\n");
    return 0;
}
