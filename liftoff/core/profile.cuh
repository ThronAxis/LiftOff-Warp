// liftoff/core/profile.cuh
// LIFTOFF: Warp-Level Primitives Library
// Profiling macros: CUDA Event timing, NVTX annotations, error checking, occupancy queries
// Author: Maaran | ML Systems Engineering Research

#pragma once
#include <cuda_runtime.h>
#include <cstdio>

namespace liftoff {

// ─── CUDA EVENT TIMER ─────────────────────────────────────────────────────────
struct CudaTimer {
    cudaEvent_t start_, stop_;
    float ms_ = 0.f;

    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { cudaEventRecord(start_); }
    void stop()  {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        cudaEventElapsedTime(&ms_, start_, stop_);
    }
    float ms()   { return ms_; }
    float us()   { return ms_ * 1000.f; }
};

// ─── NVTX RANGE (for Nsight Systems profiler) ─────────────────────────────────
#ifdef LIFTOFF_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#define LIFTOFF_RANGE_PUSH(name) nvtxRangePushA(name)
#define LIFTOFF_RANGE_POP()      nvtxRangePop()
#else
#define LIFTOFF_RANGE_PUSH(name)
#define LIFTOFF_RANGE_POP()
#endif

// ─── CUDA ERROR CHECK ─────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(1);                                                       \
        }                                                                  \
    } while(0)

// ─── OCCUPANCY QUERY MACRO ────────────────────────────────────────────────────
#define LIFTOFF_QUERY_OCCUPANCY(kernel, block_size, dynamic_smem)          \
    do {                                                                   \
        int max_blocks = 0;                                                \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(                     \
            &max_blocks, kernel, block_size, dynamic_smem);                \
        int device; cudaGetDevice(&device);                                \
        int sm_count;                                                      \
        cudaDeviceGetAttribute(&sm_count,                                  \
            cudaDevAttrMultiProcessorCount, device);                       \
        printf("Occupancy [%s]: %d blocks/SM × %d SMs = %d total warps\n",\
            #kernel, max_blocks, sm_count,                                 \
            max_blocks * sm_count * (block_size / 32));                    \
    } while(0)

// ─── SCOPED TIMER (RAII) ──────────────────────────────────────────────────────
struct ScopedTimer {
    const char* name_;
    CudaTimer timer_;

    ScopedTimer(const char* name) : name_(name) {
        timer_.start();
    }
    ~ScopedTimer() {
        timer_.stop();
        printf("[LIFTOFF] %s: %.3f ms (%.1f μs)\n", name_, timer_.ms(), timer_.us());
    }
};

#define LIFTOFF_TIME_SCOPE(name) liftoff::ScopedTimer _liftoff_timer_##__LINE__(name)

} // namespace liftoff
