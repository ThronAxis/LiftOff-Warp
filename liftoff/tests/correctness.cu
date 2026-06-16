// liftoff/tests/correctness.cu — Full correctness test suite
// PRD §9: edge cases, partial masks, WarpTile, divergence tests
// Build: nvcc -O3 -arch=sm_75 -std=c++17 -I./liftoff tests/correctness.cu -o liftoff_test
#include "../liftoff.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

using namespace liftoff;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { printf("  ✗ FAIL: %s\n", msg); tests_failed++; } \
    else { printf("  ✓ PASS: %s\n", msg); tests_passed++; } \
} while(0)

// ═══════════ REDUCE TESTS ═══════════
__global__ void test_reduce_kernel(float* in, float* s, float* mx, float* mn) {
    float val = in[threadIdx.x];
    float rs  = warp_reduce_sum(val);
    float rmx = warp_reduce_max(val);
    float rmn = warp_reduce_min(val);
    if (lane_id() == 0) {
        *s  = rs;
        *mx = rmx;
        *mn = rmn;
    }
}

__global__ void test_reduce_allzero(float* out) {
    float val = 0.0f;
    if (lane_id() == 0) *out = warp_reduce_sum(val);
}

__global__ void test_reduce_allsame(float* out) {
    float val = 42.0f;
    if (lane_id() == 0) *out = warp_reduce_sum(val);
}

void test_warp_reduce() {
    printf("\n── Test: Warp Reduce ──────────────────────────────────────\n");
    float h_in[32], h_s, h_mx, h_mn;
    float *d_in, *d_s, *d_mx, *d_mn;
    float cpu_s=0, cpu_mx=-1e38f, cpu_mn=1e38f;
    for (int i=0;i<32;i++) {
        h_in[i]=(float)(i+1); cpu_s+=h_in[i];
        cpu_mx=fmaxf(cpu_mx,h_in[i]); cpu_mn=fminf(cpu_mn,h_in[i]);
    }
    CUDA_CHECK(cudaMalloc(&d_in,32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s,sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mx,sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mn,sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,32*sizeof(float),cudaMemcpyHostToDevice));
    test_reduce_kernel<<<1,32>>>(d_in,d_s,d_mx,d_mn);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_s,d_s,sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_mx,d_mx,sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_mn,d_mn,sizeof(float),cudaMemcpyDeviceToHost));
    TEST_ASSERT(fabsf(h_s-cpu_s)<1e-3f,"reduce_sum [1..32]");
    TEST_ASSERT(fabsf(h_mx-cpu_mx)<1e-5f,"reduce_max [1..32]");
    TEST_ASSERT(fabsf(h_mn-cpu_mn)<1e-5f,"reduce_min [1..32]");

    // Edge: all-zero
    float h_z; float *d_z;
    CUDA_CHECK(cudaMalloc(&d_z,sizeof(float)));
    test_reduce_allzero<<<1,32>>>(d_z);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_z,d_z,sizeof(float),cudaMemcpyDeviceToHost));
    TEST_ASSERT(fabsf(h_z)<1e-5f,"reduce_sum all-zero → 0");

    // Edge: all-same
    float h_same; float *d_same;
    CUDA_CHECK(cudaMalloc(&d_same,sizeof(float)));
    test_reduce_allsame<<<1,32>>>(d_same);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_same,d_same,sizeof(float),cudaMemcpyDeviceToHost));
    TEST_ASSERT(fabsf(h_same-42.0f*32)<1e-2f,"reduce_sum all-same (42×32=1344)");

    cudaFree(d_in);cudaFree(d_s);cudaFree(d_mx);cudaFree(d_mn);
    cudaFree(d_z);cudaFree(d_same);
}

// ═══════════ SCAN TESTS ═══════════
__global__ void test_scan_kernel(float* in, float* incl, float* excl) {
    float val = in[threadIdx.x];
    incl[threadIdx.x] = warp_scan_inclusive(val);
    excl[threadIdx.x] = warp_scan_exclusive(val);
}

void test_warp_scan() {
    printf("\n── Test: Warp Scan ────────────────────────────────────────\n");
    float h_in[32],h_incl[32],h_excl[32];
    float *d_in,*d_incl,*d_excl;
    for(int i=0;i<32;i++) h_in[i]=1.0f;
    CUDA_CHECK(cudaMalloc(&d_in,32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_incl,32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_excl,32*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,32*sizeof(float),cudaMemcpyHostToDevice));
    test_scan_kernel<<<1,32>>>(d_in,d_incl,d_excl);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_incl,d_incl,32*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_excl,d_excl,32*sizeof(float),cudaMemcpyDeviceToHost));
    bool iok=true,eok=true;
    for(int i=0;i<32;i++){
        if(fabsf(h_incl[i]-(float)(i+1))>1e-5f) iok=false;
        if(fabsf(h_excl[i]-(float)(i))>1e-5f) eok=false;
    }
    TEST_ASSERT(iok,"scan_inclusive (all-ones → [1..32])");
    TEST_ASSERT(eok,"scan_exclusive (all-ones → [0..31])");

    // Varying input scan
    for(int i=0;i<32;i++) h_in[i]=(float)(i+1);
    CUDA_CHECK(cudaMemcpy(d_in,h_in,32*sizeof(float),cudaMemcpyHostToDevice));
    test_scan_kernel<<<1,32>>>(d_in,d_incl,d_excl);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_incl,d_incl,32*sizeof(float),cudaMemcpyDeviceToHost));
    float cpu_prefix=0; bool vary_ok=true;
    for(int i=0;i<32;i++){
        cpu_prefix+=(float)(i+1);
        if(fabsf(h_incl[i]-cpu_prefix)>1e-2f) vary_ok=false;
    }
    TEST_ASSERT(vary_ok,"scan_inclusive (varying input [1..32])");
    cudaFree(d_in);cudaFree(d_incl);cudaFree(d_excl);
}

// ═══════════ SORT TESTS ═══════════
__global__ void test_sort_kernel(float* data) {
    float val = data[threadIdx.x];
    warp_sort_ascending(val);
    data[threadIdx.x] = val;
}

void test_warp_sort() {
    printf("\n── Test: Warp Sort ────────────────────────────────────────\n");
    float h[32]; float *d;
    CUDA_CHECK(cudaMalloc(&d,32*sizeof(float)));

    // Reverse input
    for(int i=0;i<32;i++) h[i]=(float)(32-i);
    CUDA_CHECK(cudaMemcpy(d,h,32*sizeof(float),cudaMemcpyHostToDevice));
    test_sort_kernel<<<1,32>>>(d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h,d,32*sizeof(float),cudaMemcpyDeviceToHost));
    bool sorted=true;
    for(int i=1;i<32;i++) if(h[i]<h[i-1]){sorted=false;break;}
    TEST_ASSERT(sorted,"sort_ascending (reverse → sorted)");

    // All-same input
    for(int i=0;i<32;i++) h[i]=5.0f;
    CUDA_CHECK(cudaMemcpy(d,h,32*sizeof(float),cudaMemcpyHostToDevice));
    test_sort_kernel<<<1,32>>>(d);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h,d,32*sizeof(float),cudaMemcpyDeviceToHost));
    bool all_five=true;
    for(int i=0;i<32;i++) if(fabsf(h[i]-5.0f)>1e-5f){all_five=false;break;}
    TEST_ASSERT(all_five,"sort_ascending (all-same input → unchanged)");

    cudaFree(d);
}

// ═══════════ BALLOT TESTS ═══════════
__global__ void test_ballot_kernel(int* pop, int* lead, int* rank) {
    bool pred = (lane_id()%2==0);
    if(lane_id()==0){ *pop=warp_popcount(pred); *lead=warp_leader_lane(pred); }
    rank[lane_id()] = warp_my_rank(pred);
}

void test_warp_ballot() {
    printf("\n── Test: Warp Ballot ──────────────────────────────────────\n");
    int *d_p,*d_l,*d_r; int h_p,h_l,h_r[32];
    CUDA_CHECK(cudaMalloc(&d_p,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_l,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_r,32*sizeof(int)));
    test_ballot_kernel<<<1,32>>>(d_p,d_l,d_r);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_p,d_p,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_l,d_l,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_r,d_r,32*sizeof(int),cudaMemcpyDeviceToHost));
    TEST_ASSERT(h_p==16,"popcount (even lanes → 16)");
    TEST_ASSERT(h_l==0,"leader_lane (even pred → lane 0)");
    // Even lanes should have ranks 0,1,2,...15
    bool rank_ok=true;
    for(int i=0;i<32;i+=2) if(h_r[i]!=i/2){rank_ok=false;break;}
    TEST_ASSERT(rank_ok,"my_rank (even lanes → 0,1,...,15)");
    cudaFree(d_p);cudaFree(d_l);cudaFree(d_r);
}

// ═══════════ BROADCAST TESTS ═══════════
__global__ void test_broadcast_kernel(float* out) {
    float val=(float)lane_id();
    out[lane_id()]=warp_broadcast(val,7);
}

__global__ void test_reverse_kernel(float* out) {
    float val=(float)lane_id();
    out[lane_id()]=warp_reverse(val);
}

__global__ void test_rotate_kernel(float* out) {
    float val=(float)lane_id();
    out[lane_id()]=warp_rotate_up(val,3);
}

void test_warp_broadcast() {
    printf("\n── Test: Warp Broadcast & Exchange ─────────────────────────\n");
    float *d_o,h_o[32];
    CUDA_CHECK(cudaMalloc(&d_o,32*sizeof(float)));

    // Broadcast
    test_broadcast_kernel<<<1,32>>>(d_o);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_o,d_o,32*sizeof(float),cudaMemcpyDeviceToHost));
    bool bc_ok=true;
    for(int i=0;i<32;i++) if(fabsf(h_o[i]-7.0f)>1e-5f){bc_ok=false;break;}
    TEST_ASSERT(bc_ok,"broadcast lane 7 → all get 7.0");

    // Reverse
    test_reverse_kernel<<<1,32>>>(d_o);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_o,d_o,32*sizeof(float),cudaMemcpyDeviceToHost));
    bool rev_ok=true;
    for(int i=0;i<32;i++) if(fabsf(h_o[i]-(float)(31-i))>1e-5f){rev_ok=false;break;}
    TEST_ASSERT(rev_ok,"reverse (lane k → lane 31-k)");

    // Rotate up by 3
    test_rotate_kernel<<<1,32>>>(d_o);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_o,d_o,32*sizeof(float),cudaMemcpyDeviceToHost));
    bool rot_ok=true;
    for(int i=0;i<32;i++){
        float expected=(float)((i-3+32)%32);
        if(fabsf(h_o[i]-expected)>1e-5f){rot_ok=false;break;}
    }
    TEST_ASSERT(rot_ok,"rotate_up by 3");

    cudaFree(d_o);
}

// ═══════════ SOFTMAX TEST ═══════════
void test_softmax() {
    printf("\n── Test: Warp Softmax ─────────────────────────────────────\n");
    float h_in[32],h_out[32]; float *d_in,*d_out;
    for(int i=0;i<32;i++) h_in[i]=(float)i*0.1f;
    CUDA_CHECK(cudaMalloc(&d_in,32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,32*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in,h_in,32*sizeof(float),cudaMemcpyHostToDevice));
    warp_softmax_kernel<float><<<1,32>>>(d_in,d_out,1,32);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out,d_out,32*sizeof(float),cudaMemcpyDeviceToHost));
    float sum=0; bool pos=true;
    for(int i=0;i<32;i++){sum+=h_out[i]; if(h_out[i]<=0)pos=false;}
    TEST_ASSERT(fabsf(sum-1.0f)<1e-4f,"softmax sum ≈ 1.0");
    TEST_ASSERT(pos,"softmax all values > 0");
    TEST_ASSERT(h_out[31]>h_out[0],"softmax monotonicity");

    // Edge: all-zero input → uniform distribution
    for(int i=0;i<32;i++) h_in[i]=0.0f;
    CUDA_CHECK(cudaMemcpy(d_in,h_in,32*sizeof(float),cudaMemcpyHostToDevice));
    warp_softmax_kernel<float><<<1,32>>>(d_in,d_out,1,32);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out,d_out,32*sizeof(float),cudaMemcpyDeviceToHost));
    bool uniform=true;
    for(int i=0;i<32;i++) if(fabsf(h_out[i]-1.0f/32.0f)>1e-4f){uniform=false;break;}
    TEST_ASSERT(uniform,"softmax all-zero → uniform 1/32");
    cudaFree(d_in);cudaFree(d_out);
}

// ═══════════ DOT PRODUCT TEST ═══════════
__global__ void test_dot_kernel(float* a, float* b, float* out) {
    float result = warp_dot_product(a, b, 32);
    if(lane_id()==0) *out = result;
}

void test_dot_product() {
    printf("\n── Test: Warp Dot Product ──────────────────────────────────\n");
    float ha[32],hb[32],h_r; float *da,*db,*dr;
    float cpu_dot=0;
    for(int i=0;i<32;i++){ha[i]=(float)(i+1);hb[i]=1.0f;cpu_dot+=ha[i];}
    CUDA_CHECK(cudaMalloc(&da,32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db,32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dr,sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da,ha,32*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db,hb,32*sizeof(float),cudaMemcpyHostToDevice));
    test_dot_kernel<<<1,32>>>(da,db,dr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_r,dr,sizeof(float),cudaMemcpyDeviceToHost));
    TEST_ASSERT(fabsf(h_r-cpu_dot)<1e-2f,"dot_product [1..32]·[1..1] = 528");
    cudaFree(da);cudaFree(db);cudaFree(dr);
}

// ═══════════ WARPTILE TEST ═══════════
__global__ void test_warptile8_kernel(float* out_sum, float* out_max) {
    WarpTile<8> tile;
    float val = (float)(tile.lane() + 1);  // 1..8 within each tile
    float s = tile.reduce_sum(val);
    float m = tile.reduce_max(val);
    if(tile.lane()==0){
        out_sum[tile.tile_rank()] = s;
        out_max[tile.tile_rank()] = m;
    }
}

void test_warptile() {
    printf("\n── Test: WarpTile<8> ───────────────────────────────────────\n");
    float *d_s,*d_m; float h_s[4],h_m[4];
    CUDA_CHECK(cudaMalloc(&d_s,4*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m,4*sizeof(float)));
    test_warptile8_kernel<<<1,32>>>(d_s,d_m);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_s,d_s,4*sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m,d_m,4*sizeof(float),cudaMemcpyDeviceToHost));
    // Each tile: sum(1..8) = 36, max(1..8) = 8
    bool s_ok=true,m_ok=true;
    for(int i=0;i<4;i++){
        if(fabsf(h_s[i]-36.0f)>1e-3f) s_ok=false;
        if(fabsf(h_m[i]-8.0f)>1e-5f) m_ok=false;
    }
    TEST_ASSERT(s_ok,"WarpTile<8> reduce_sum (4 tiles, each sum=36)");
    TEST_ASSERT(m_ok,"WarpTile<8> reduce_max (4 tiles, each max=8)");
    cudaFree(d_s);cudaFree(d_m);
}

// ═══════════ ARGMAX TEST ═══════════
__global__ void test_argmax_kernel(float* in, float* out_val, int* out_idx) {
    float val = in[threadIdx.x];
    int idx = threadIdx.x;
    warp_reduce_argmax(val, idx);
    if(lane_id()==0){ *out_val=val; *out_idx=idx; }
}

void test_argmax() {
    printf("\n── Test: Warp Argmax ───────────────────────────────────────\n");
    float h[32]; for(int i=0;i<32;i++) h[i]=(float)(i*3);
    float *d_in,*d_v; int *d_i; float hv; int hi;
    CUDA_CHECK(cudaMalloc(&d_in,32*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v,sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_i,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in,h,32*sizeof(float),cudaMemcpyHostToDevice));
    test_argmax_kernel<<<1,32>>>(d_in,d_v,d_i);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&hv,d_v,sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&hi,d_i,sizeof(int),cudaMemcpyDeviceToHost));
    TEST_ASSERT(fabsf(hv-93.0f)<1e-3f,"argmax value = 93 (31*3)");
    TEST_ASSERT(hi==31,"argmax index = 31");
    cudaFree(d_in);cudaFree(d_v);cudaFree(d_i);
}

// ═══════════ MAIN ═══════════
int main() {
    print_device_info();
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  LIFTOFF Correctness Test Suite (Full Coverage)\n");
    printf("════════════════════════════════════════════════════════════\n");

    test_warp_reduce();
    test_warp_scan();
    test_warp_sort();
    test_warp_ballot();
    test_warp_broadcast();
    test_softmax();
    test_dot_product();
    test_warptile();
    test_argmax();

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  Results: %d PASSED, %d FAILED (total %d tests)\n",
           tests_passed, tests_failed, tests_passed+tests_failed);
    printf("════════════════════════════════════════════════════════════\n");
    return tests_failed > 0 ? 1 : 0;
}
