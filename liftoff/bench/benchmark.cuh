// liftoff/bench/benchmark.cuh
// LIFTOFF: Benchmark harness — timing, statistics, CSV output
#pragma once
#include "../core/profile.cuh"
#include <cstdio>
#include <functional>

namespace liftoff {

struct BenchResult {
    const char* name;
    float median_us;
    float min_us;
    float max_us;
    long long ops;
    float throughput_gops;
};

// Benchmark runner with warmup and timed iterations
template<typename KernelFn>
BenchResult benchmark(
    const char* name,
    KernelFn kernel_fn,
    long long ops_per_call,
    int warmup_iters = 100,
    int timed_iters  = 1000)
{
    // Warmup
    for (int i = 0; i < warmup_iters; i++) kernel_fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    float* timings = new float[timed_iters];
    CudaTimer timer;

    for (int i = 0; i < timed_iters; i++) {
        timer.start();
        kernel_fn();
        timer.stop();
        timings[i] = timer.us();
    }

    // Statistics
    float sum = 0, mn = 1e18f, mx = 0;
    for (int i = 0; i < timed_iters; i++) {
        sum += timings[i];
        mn = timings[i] < mn ? timings[i] : mn;
        mx = timings[i] > mx ? timings[i] : mx;
    }
    float median = timings[timed_iters / 2];

    BenchResult r;
    r.name           = name;
    r.median_us      = median;
    r.min_us         = mn;
    r.max_us         = mx;
    r.ops            = ops_per_call;
    r.throughput_gops = (ops_per_call / 1e9f) / (median / 1e6f);

    printf("%-40s | median %8.2f us | min %8.2f us | %.2f GOPS\n",
           name, median, mn, r.throughput_gops);

    delete[] timings;
    return r;
}

// CSV writer for benchmark results
inline void write_csv(const char* path, BenchResult* results, int n) {
    FILE* f = fopen(path, "w");
    fprintf(f, "name,median_us,min_us,max_us,gops\n");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s,%.4f,%.4f,%.4f,%.4f\n",
            results[i].name, results[i].median_us,
            results[i].min_us, results[i].max_us,
            results[i].throughput_gops);
    }
    fclose(f);
    printf("Results written to %s\n", path);
}

} // namespace liftoff
