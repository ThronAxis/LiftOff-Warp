// liftoff/core/config.cuh
// LIFTOFF: Warp-Level Primitives Library
// Core configuration: warp size constants, SM detection, architecture guards
// Author: Maaran | ML Systems Engineering Research
// Target: NVIDIA CUDA 12.x, Compute Capability 7.0+

#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace liftoff {

static constexpr int WARP_SIZE = 32;
static constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

// Compile-time SM guard
#if __CUDA_ARCH__ >= 700
    #define LIFTOFF_VOLTA_PLUS 1
#endif
#if __CUDA_ARCH__ >= 750
    #define LIFTOFF_TURING_PLUS 1
#endif
#if __CUDA_ARCH__ >= 800
    #define LIFTOFF_AMPERE_PLUS 1
#endif

// Runtime SM detection
inline int get_sm_count() {
    int sm_count;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    return sm_count;
}

inline int get_sm_version() {
    int major, minor, device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    return major * 10 + minor;
}

// Max warps per SM query
inline int get_max_warps_per_sm() {
    int max_threads;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device);
    return max_threads / WARP_SIZE;
}

// Shared memory per SM query
inline int get_shared_mem_per_sm() {
    int smem;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    return smem;
}

// Print device info
inline void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  LIFTOFF Device: %s\n", prop.name);
    printf("  SM Count: %d | SM Version: %d.%d\n", prop.multiProcessorCount, prop.major, prop.minor);
    printf("  Global Memory: %.1f GB\n", prop.totalGlobalMem / 1073741824.0);
    printf("  Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max Threads/SM: %d | Max Warps/SM: %d\n",
           prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerMultiProcessor / WARP_SIZE);
    printf("  Warp Size: %d | Clock Rate: %.0f MHz\n", prop.warpSize, prop.clockRate / 1000.0);
    printf("════════════════════════════════════════════════════════════\n");
}

} // namespace liftoff
