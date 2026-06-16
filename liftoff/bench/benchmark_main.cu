// liftoff/bench/benchmark_main.cu — Full benchmark runner (MD §14 compliant)
// Build: nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 -I./liftoff bench/benchmark_main.cu -o liftoff_bench
#include "../liftoff.cuh"
#include "benchmark.cuh"
#include <cstdlib>
#include <cstring>

using namespace liftoff;

// ── Primitive benchmark kernels ──────────────────────────────────────────────

__global__ void bench_warp_reduce(float* data, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? data[tid] : 0.f;
    float result = warp_reduce_sum(val);
    if (lane_id() == 0) out[blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id()] = result;
}

__global__ void bench_shared_reduce(float* data, float* out, int N) {
    extern __shared__ float smem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    smem[lid] = (tid < N) ? data[tid] : 0.f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lid < s) smem[lid] += smem[lid + s];
        __syncthreads();
    }
    if (lid == 0) out[blockIdx.x] = smem[0];
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

int main() {
    print_device_info();

    const int N = 1 << 20;
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    float* h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = static_cast<float>(i % 100) * 0.01f;
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    const int BLK = 256, GRD = (N + BLK - 1) / BLK;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  LIFTOFF Benchmark Suite — Full PRD Coverage\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    printf("── Micro-Benchmarks ────────────────────────────────────────\n\n");

    const int NUM_BENCH = 10;
    BenchResult results[NUM_BENCH];

    results[0] = benchmark("warp_reduce_sum (1M floats)",
        [&]{ bench_warp_reduce<<<GRD,BLK>>>(d_in,d_out,N); }, N);

    results[1] = benchmark("shared_mem_reduce (BASELINE)",
        [&]{ bench_shared_reduce<<<GRD,BLK,BLK*sizeof(float)>>>(d_in,d_out,N); }, N);

    results[2] = benchmark("warp_scan_inclusive (1M)",
        [&]{ bench_warp_scan<<<GRD,BLK>>>(d_in,d_out,N); }, N * 5);

    results[3] = benchmark("warp_sort_ascending (1M)",
        [&]{ bench_warp_sort<<<GRD,BLK>>>(d_in,d_out,N); }, N * 15);

    // ── Application benchmarks (MD §14) ──────────────────────────────────────

    printf("\n── Application Benchmarks ──────────────────────────────────\n\n");

    // Softmax: rows=4096, cols=256
    results[4] = benchmark("warp_softmax (4096x256)",
        [&]{ warp_softmax_kernel<float><<<4096,BLK>>>(d_in,d_out,4096,256); },
        4096LL * 256 * 4);

    // LayerNorm: rows=1024, cols=768
    float *d_g, *d_b, *d_li, *d_lo;
    int lr=1024, lc=768;
    CUDA_CHECK(cudaMalloc(&d_g,  lc*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b,  lc*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_li, lr*lc*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_lo, lr*lc*sizeof(float)));
    // Init gamma=1, beta=0
    float* h_g = new float[lc];
    for(int i=0;i<lc;i++) h_g[i]=1.0f;
    CUDA_CHECK(cudaMemcpy(d_g,h_g,lc*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_b,0,lc*sizeof(float)));

    results[5] = benchmark("warp_layernorm (1024x768)",
        [&]{ warp_layernorm_kernel<float><<<lr,BLK>>>(d_li,d_g,d_b,d_lo,lr,lc); },
        (long long)lr*lc*5);

    // Dot Product: 4096 pairs, vec_len=256
    int dp_n=4096, dp_len=256;
    float *d_a, *d_b2, *d_dp;
    CUDA_CHECK(cudaMalloc(&d_a, dp_n*dp_len*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,dp_n*dp_len*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dp,dp_n*sizeof(float)));

    results[6] = benchmark("warp_dot_product (4096x256)",
        [&]{ warp_dot_product_kernel<float><<<(dp_n+7)/8,BLK>>>(d_a,d_b2,d_dp,dp_n,dp_len); },
        (long long)dp_n*dp_len*2);

    // Attention Score: seq=512, d_head=64
    int seq=512, dh=64;
    float *d_Q, *d_K, *d_sc;
    CUDA_CHECK(cudaMalloc(&d_Q, seq*dh*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, seq*dh*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sc,seq*seq*sizeof(float)));
    dim3 attn_grid(seq, (seq+7)/8);

    results[7] = benchmark("warp_attention (512x64)",
        [&]{ warp_attention_score_kernel<float><<<attn_grid,BLK>>>(d_Q,d_K,d_sc,seq,dh); },
        (long long)seq*seq*dh*2);

    // RMS Norm: rows=1024, cols=768
    float *d_w, *d_ro;
    CUDA_CHECK(cudaMalloc(&d_w,  lc*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ro, lr*lc*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w,h_g,lc*sizeof(float),cudaMemcpyHostToDevice));

    results[8] = benchmark("warp_rmsnorm (1024x768)",
        [&]{ warp_rmsnorm_kernel<float><<<lr,BLK>>>(d_li,d_w,d_ro,lr,lc); },
        (long long)lr*lc*3);

    // GELU fused reduce: rows=4096, cols=256
    results[9] = benchmark("warp_gelu_reduce (4096x256)",
        [&]{ warp_gelu_fused_reduce_kernel<float><<<4096,BLK>>>(d_in,d_out,4096,256); },
        4096LL*256*8);

    // ── Occupancy Analysis ───────────────────────────────────────────────────

    printf("\n── Occupancy Analysis ──────────────────────────────────────\n");
    LIFTOFF_QUERY_OCCUPANCY(bench_warp_reduce, BLK, 0);
    LIFTOFF_QUERY_OCCUPANCY(bench_shared_reduce, BLK, BLK*sizeof(float));
    LIFTOFF_QUERY_OCCUPANCY(bench_warp_sort, BLK, 0);

    // Speedup summary
    printf("\n── LIFTOFF vs Shared Memory ────────────────────────────────\n");
    if (results[1].median_us > 0 && results[0].median_us > 0) {
        float speedup = results[1].median_us / results[0].median_us;
        printf("  Reduce speedup: %.2fx (shuffle vs shared)\n", speedup);
    }

    write_csv("liftoff_bench_results.csv", results, NUM_BENCH);

    // Cleanup
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_g));  CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_li)); CUDA_CHECK(cudaFree(d_lo));
    CUDA_CHECK(cudaFree(d_a));  CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_dp)); CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));  CUDA_CHECK(cudaFree(d_sc));
    CUDA_CHECK(cudaFree(d_w));  CUDA_CHECK(cudaFree(d_ro));
    delete[] h_in; delete[] h_g;

    printf("\n✓ LIFTOFF Benchmark complete (10 benchmarks).\n");
    return 0;
}
