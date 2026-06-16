// liftoff/tests/correctness.cu
// LIFTOFF: Correctness tests — CPU reference vs GPU warp primitives
// Build: nvcc -O3 -arch=sm_75 -std=c++17 -I./liftoff correctness.cu -o liftoff_test
#include "../liftoff.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace liftoff;

// ── Test status tracking ─────────────────────────────────────────────────────
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                      \
    do {                                                             \
        if (!(cond)) {                                               \
            printf("  ✗ FAIL: %s\n", msg);                          \
            tests_failed++;                                          \
        } else {                                                     \
            printf("  ✓ PASS: %s\n", msg);                          \
            tests_passed++;                                          \
        }                                                            \
    } while(0)

// ── Reduce Test Kernel ───────────────────────────────────────────────────────
__global__ void test_reduce_kernel(float* input, float* out_sum,
                                    float* out_max, float* out_min) {
    float val = input[threadIdx.x];
    float s = warp_reduce_sum(val);
    float mx = warp_reduce_max(val);
    float mn = warp_reduce_min(val);
    if (lane_id() == 0) {
        *out_sum = s;
        *out_max = mx;
        *out_min = mn;
    }
}

void test_warp_reduce() {
    printf("\n── Test: Warp Reduce ──────────────────────────────────────\n");
    float h_in[32], h_sum, h_max, h_min;
    float *d_in, *d_sum, *d_max, *d_min;

    float cpu_sum = 0, cpu_max = -1e38f, cpu_min = 1e38f;
    for (int i = 0; i < 32; i++) {
        h_in[i] = (float)(i + 1);
        cpu_sum += h_in[i];
        cpu_max = fmaxf(cpu_max, h_in[i]);
        cpu_min = fminf(cpu_min, h_in[i]);
    }

    CUDA_CHECK(cudaMalloc(&d_in,  32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice));

    test_reduce_kernel<<<1, 32>>>(d_in, d_sum, d_max, d_min);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost));

    TEST_ASSERT(fabsf(h_sum - cpu_sum) < 1e-3f, "reduce_sum correctness");
    TEST_ASSERT(fabsf(h_max - cpu_max) < 1e-5f, "reduce_max correctness");
    TEST_ASSERT(fabsf(h_min - cpu_min) < 1e-5f, "reduce_min correctness");

    cudaFree(d_in); cudaFree(d_sum); cudaFree(d_max); cudaFree(d_min);
}

// ── Scan Test Kernel ─────────────────────────────────────────────────────────
__global__ void test_scan_kernel(float* input, float* out_incl, float* out_excl) {
    float val = input[threadIdx.x];
    out_incl[threadIdx.x] = warp_scan_inclusive(val);
    out_excl[threadIdx.x] = warp_scan_exclusive(val);
}

void test_warp_scan() {
    printf("\n── Test: Warp Scan ────────────────────────────────────────\n");
    float h_in[32], h_incl[32], h_excl[32];
    float *d_in, *d_incl, *d_excl;

    for (int i = 0; i < 32; i++) h_in[i] = 1.0f;

    CUDA_CHECK(cudaMalloc(&d_in,   32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_incl, 32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_excl, 32 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice));

    test_scan_kernel<<<1, 32>>>(d_in, d_incl, d_excl);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_incl, d_incl, 32 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_excl, d_excl, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    bool incl_ok = true, excl_ok = true;
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_incl[i] - (float)(i + 1)) > 1e-5f) incl_ok = false;
        if (fabsf(h_excl[i] - (float)(i))     > 1e-5f) excl_ok = false;
    }
    TEST_ASSERT(incl_ok, "scan_inclusive (all-ones → [1,2,...,32])");
    TEST_ASSERT(excl_ok, "scan_exclusive (all-ones → [0,1,...,31])");

    cudaFree(d_in); cudaFree(d_incl); cudaFree(d_excl);
}

// ── Sort Test Kernel ─────────────────────────────────────────────────────────
__global__ void test_sort_kernel(float* data) {
    float val = data[threadIdx.x];
    warp_sort_ascending(val);
    data[threadIdx.x] = val;
}

void test_warp_sort() {
    printf("\n── Test: Warp Sort ────────────────────────────────────────\n");
    float h_data[32];
    float *d_data;

    // Reverse-sorted input
    for (int i = 0; i < 32; i++) h_data[i] = (float)(32 - i);

    CUDA_CHECK(cudaMalloc(&d_data, 32 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, 32 * sizeof(float), cudaMemcpyHostToDevice));

    test_sort_kernel<<<1, 32>>>(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    bool sorted = true;
    for (int i = 1; i < 32; i++) {
        if (h_data[i] < h_data[i-1]) { sorted = false; break; }
    }
    TEST_ASSERT(sorted, "sort_ascending (reverse → sorted)");

    cudaFree(d_data);
}

// ── Ballot Test Kernel ───────────────────────────────────────────────────────
__global__ void test_ballot_kernel(int* out_popcount, int* out_leader, int* out_rank) {
    bool pred = (lane_id() % 2 == 0);  // even lanes
    if (lane_id() == 0) {
        *out_popcount = warp_popcount(pred);
        *out_leader   = warp_leader_lane(pred);
    }
    *out_rank = warp_my_rank(pred);
}

void test_warp_ballot() {
    printf("\n── Test: Warp Ballot ──────────────────────────────────────\n");
    int *d_pop, *d_lead, *d_rank;
    int h_pop, h_lead, h_rank[32];

    CUDA_CHECK(cudaMalloc(&d_pop,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lead, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rank, 32 * sizeof(int)));

    test_ballot_kernel<<<1, 32>>>(d_pop, d_lead, d_rank);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_pop,  d_pop,  sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_lead, d_lead, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rank,  d_rank, 32 * sizeof(int), cudaMemcpyDeviceToHost));

    TEST_ASSERT(h_pop == 16, "popcount (even lanes → 16)");
    TEST_ASSERT(h_lead == 0, "leader_lane (even pred → lane 0)");

    cudaFree(d_pop); cudaFree(d_lead); cudaFree(d_rank);
}

// ── Broadcast Test Kernel ────────────────────────────────────────────────────
__global__ void test_broadcast_kernel(float* out) {
    float val = (float)lane_id();
    float bc = warp_broadcast(val, 7);
    out[lane_id()] = bc;
}

void test_warp_broadcast() {
    printf("\n── Test: Warp Broadcast ────────────────────────────────────\n");
    float *d_out, h_out[32];
    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(float)));

    test_broadcast_kernel<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < 32; i++) {
        if (fabsf(h_out[i] - 7.0f) > 1e-5f) { ok = false; break; }
    }
    TEST_ASSERT(ok, "broadcast from lane 7 → all lanes get 7.0");

    cudaFree(d_out);
}

// ── Softmax Test ─────────────────────────────────────────────────────────────
void test_softmax() {
    printf("\n── Test: Warp Softmax ─────────────────────────────────────\n");
    int cols = 32;
    float h_in[32], h_out[32];
    float *d_in, *d_out;

    for (int i = 0; i < 32; i++) h_in[i] = (float)i * 0.1f;

    CUDA_CHECK(cudaMalloc(&d_in,  32 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, 32 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice));

    warp_softmax_kernel<float><<<1, 32>>>(d_in, d_out, 1, cols);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: sum ≈ 1.0, all values > 0
    float sum = 0;
    bool positive = true;
    for (int i = 0; i < 32; i++) {
        sum += h_out[i];
        if (h_out[i] <= 0) positive = false;
    }
    TEST_ASSERT(fabsf(sum - 1.0f) < 1e-4f, "softmax sum ≈ 1.0");
    TEST_ASSERT(positive, "softmax all values > 0");
    TEST_ASSERT(h_out[31] > h_out[0], "softmax monotonicity (larger input → larger prob)");

    cudaFree(d_in); cudaFree(d_out);
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main() {
    print_device_info();

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  LIFTOFF Correctness Test Suite\n");
    printf("════════════════════════════════════════════════════════════\n");

    test_warp_reduce();
    test_warp_scan();
    test_warp_sort();
    test_warp_ballot();
    test_warp_broadcast();
    test_softmax();

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  Results: %d PASSED, %d FAILED\n", tests_passed, tests_failed);
    printf("════════════════════════════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
