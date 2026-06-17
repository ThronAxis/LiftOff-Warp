// liftoff/bench/benchmark.cuh
// LIFTOFF: Benchmark harness — timing, statistics, CSV output
// Proper sort-based median, stddev, bandwidth, speedup tracking
#pragma once
#include "../core/profile.cuh"
#include <cstdio>
#include <cmath>
#include <functional>
#include <algorithm>

namespace liftoff {

struct BenchResult {
    const char* name;
    float median_us;
    float min_us;
    float max_us;
    float mean_us;
    float stddev_us;
    long long ops;
    float throughput_gops;
};

// Simple insertion sort for timing array (small N, no dependencies)
inline void sort_timings(float* arr, int n) {
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) { arr[j+1] = arr[j]; j--; }
        arr[j+1] = key;
    }
}

// Benchmark runner with warmup and timed iterations
template<typename KernelFn>
BenchResult benchmark(
    const char* name,
    KernelFn kernel_fn,
    long long ops_per_call,
    int warmup_iters = 200,
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

    // Sort for proper median (not just midpoint sample)
    sort_timings(timings, timed_iters);

    // Statistics
    float mn = timings[0];
    float mx = timings[timed_iters - 1];
    float median = (timed_iters & 1)
        ? timings[timed_iters / 2]
        : 0.5f * (timings[timed_iters/2 - 1] + timings[timed_iters/2]);

    float sum = 0;
    for (int i = 0; i < timed_iters; i++) sum += timings[i];
    float mean = sum / timed_iters;

    float var = 0;
    for (int i = 0; i < timed_iters; i++) {
        float d = timings[i] - mean;
        var += d * d;
    }
    float stddev = sqrtf(var / timed_iters);

    BenchResult r;
    r.name           = name;
    r.median_us      = median;
    r.min_us         = mn;
    r.max_us         = mx;
    r.mean_us        = mean;
    r.stddev_us      = stddev;
    r.ops            = ops_per_call;
    r.throughput_gops = (ops_per_call / 1e9f) / (median / 1e6f);

    printf("%-40s | median %8.2f us | min %8.2f us | std %6.2f us | %.2f GOPS\n",
           name, median, mn, stddev, r.throughput_gops);

    delete[] timings;
    return r;
}

// CSV writer for benchmark results
inline void write_csv(const char* path, BenchResult* results, int n) {
    FILE* f = fopen(path, "w");
    fprintf(f, "name,median_us,min_us,max_us,mean_us,stddev_us,gops\n");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            results[i].name, results[i].median_us,
            results[i].min_us, results[i].max_us,
            results[i].mean_us, results[i].stddev_us,
            results[i].throughput_gops);
    }
    fclose(f);
    printf("Results written to %s\n", path);
}

// Print speedup comparison between two results
inline void print_speedup(const BenchResult& baseline, const BenchResult& optimized) {
    float speedup = baseline.median_us / optimized.median_us;
    float latency_saved = baseline.median_us - optimized.median_us;
    printf("  ⚡ %s vs %s: %.2fx speedup (%.1f us saved)\n",
           optimized.name, baseline.name, speedup, latency_saved);
}

} // namespace liftoff

