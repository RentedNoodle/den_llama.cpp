// Gate-1 harness for the Stage-1 NVFP4 FA-VEC K-dot kernel.
//
// WHAT IT CHECKS
//   vec_dot_fattn_vec_KQ_nvfp4<D, nthreads> (fattn-common.cuh) against the CPU numeric
//   blueprint ggml_vec_dot_nvfp4_q8_0_generic (ggml-cpu/quants.c:330-363).
//   For D in {64, 128, 256}: one random block_nvfp4 K row and one random q8_0 Q row are
//   generated on the host; the CPU oracle produces the reference float; a single warp runs
//   the device VEC hook (with the same per-lane Q_q8/Q_ds register layout that
//   fattn-vec.cuh builds, then warp_reduce_sum<nthreads_KQ>) and the two are compared.
//   PASS/FAIL and max-abs-diff are printed per D.
//
// THIS FILE IS NOT PART OF ANY BUILD. It is not matched by any CMake glob/list in
// ggml/src/ggml-cuda/CMakeLists.txt (the vec instance that ships is
// fattn-vec-instance-nvfp4-nvfp4.cu). It is an authoring deliverable only.
//
// BUILD + RUN (Windows; lib naming fixed per nvfp4-stage1-review.md §4.2):
//   nvcc -std=c++17 -arch=sm_120 -O2 `
//        -I "I:/llama-fa/ggml/src" -I "I:/llama-fa/ggml/src/ggml-cuda" -I "I:/llama-fa/ggml/include" `
//        "I:/llama-fa/ggml/src/ggml-cuda/template-instances/gate1-nvfp4-vec-dot.cu" `
//        "I:/llama-fa/build-fa/ggml/src/ggml-cpu.lib" `
//        "I:/llama-fa/build-fa/ggml/src/ggml-base.lib" `
//        -lcudart -o gate1-nvfp4-vec-dot.exe
//   ./gate1-nvfp4-vec-dot.exe        # requires a CUDA GPU
//   Link note: the oracle has C linkage (declared inside extern "C" in ggml-cpu/quants.h) and
//   the exported name is ggml_vec_dot_nvfp4_q8_0_generic. If the CPU build renames it via
//   ggml-cpu/arch-fallback.h, adjust the extern declaration / link symbol accordingly.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <initializer_list>

#include "../fattn-common.cuh"

extern "C" void ggml_vec_dot_nvfp4_q8_0_generic(
        int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

// Assemble the per-lane Q register arrays exactly as fattn-vec.cuh does
// (Q_i32[r] = shared int r*nthreads + lane; Q_ds[r] = shared fs2[(r*nthreads + lane)/QI8_1]),
// then invoke the VEC hook and reduce the way the FA caller does.
template <int D>
static __global__ void gate1_kernel(
        const char * __restrict__ K, const int * __restrict__ Q_q8_shared,
        const float2 * __restrict__ Q_ds_shared, float * __restrict__ out) {
    constexpr int nthreads = (D/4 < 32 ? D/4 : 32);
    constexpr int nslot    = (1 > int(D/(sizeof(int)*nthreads)) ? 1 : int(D/(sizeof(int)*nthreads)));

    const int lane = (nthreads == WARP_SIZE) ? int(threadIdx.x) : int(threadIdx.x) % nthreads;

    int    Q_q8[nslot];
    float2 Q_ds[nslot];
#pragma unroll
    for (int r = 0; r < nslot; ++r) {
        const int i = r*nthreads + lane;
        Q_q8[r] = Q_q8_shared[i];
        Q_ds[r] = Q_ds_shared[i/QI8_1];
    }

    float sum = vec_dot_fattn_vec_KQ_nvfp4<D, nthreads>(K, nullptr, Q_q8, Q_ds);
    sum = warp_reduce_sum<nthreads>(sum);

    if (threadIdx.x == 0) {
        out[0] = sum;
    }
}

static uint32_t g_rng = 0x12345678u;
static uint32_t rng_u32() {
    g_rng = g_rng*1664525u + 1013904223u;
    return g_rng;
}

// 2^e as IEEE fp16 bit pattern (e in [-14, 15]); used to give the q8_0 blocks an exact d.
static uint16_t half_2pow(int e) {
    return (uint16_t)((e + 15) << 10);
}

static bool run_one(int D, float * io_maxdiff) {
    const int n_blk_nv = D / QK_NVFP4;   // block_nvfp4 per row
    const int n_blk_q8 = D / QK8_0;      // block_q8_0 per row

    uint8_t        * K_host  = (uint8_t *)        malloc((size_t) n_blk_nv * sizeof(block_nvfp4));
    block_q8_0     * Q_host  = (block_q8_0 *)     malloc((size_t) n_blk_q8 * sizeof(block_q8_0));
    int            * qi_host = (int *)            malloc((size_t)(D/sizeof(int)) * sizeof(int));
    float2         * qd_host = (float2 *)         malloc((size_t)(D/QI8_1) * sizeof(float2));

    // Random NVFP4 blocks: random nibbles, UE4M3 scale bytes in a safe normal range (exp==7).
    for (int b = 0; b < n_blk_nv; ++b) {
        block_nvfp4 blk;
        for (int s = 0; s < QK_NVFP4/QK_NVFP4_SUB; ++s) {
            blk.d[s] = (uint8_t)(0x38 + (rng_u32() % 7));
        }
        for (int i = 0; i < QK_NVFP4/2; ++i) {
            blk.qs[i] = (uint8_t)(rng_u32() & 0xFF);
        }
        memcpy(K_host + (size_t) b*sizeof(block_nvfp4), &blk, sizeof(blk));
    }

    // Random q8_0 blocks with an exact fp16 scale d = 2^-4 = 0.0625.
    const uint16_t q8_d_bits = half_2pow(-4);
    for (int b = 0; b < n_blk_q8; ++b) {
        memcpy(&Q_host[b].d, &q8_d_bits, sizeof(q8_d_bits));
        for (int i = 0; i < QK8_0; ++i) {
            Q_host[b].qs[i] = (int8_t) rng_u32();
        }
    }

    // Flatten Q into the q8_1-style ints + float2{d, sum} that the FA VEC hook consumes.
    // d == 2^-4 is exact, so the float form below is bit-identical to Q_host[b].d.
    for (int b = 0; b < n_blk_q8; ++b) {
        memcpy(qi_host + b*(QK8_0/sizeof(int)), Q_host[b].qs, QK8_0);
        qd_host[b] = make_float2(0.0625f, 0.0f);
    }

    // CPU oracle.
    float ref = 0.0f;
    ggml_vec_dot_nvfp4_q8_0_generic(D, &ref, 0, K_host, 0, Q_host, 0, 1);

    // GPU hook.
    char   * K_d   = nullptr;
    int    * qi_d  = nullptr;
    float2 * qd_d  = nullptr;
    float  * out_d = nullptr;
    cudaMalloc((void **) &K_d,   (size_t) n_blk_nv * sizeof(block_nvfp4));
    cudaMalloc((void **) &qi_d,  (size_t)(D/sizeof(int)) * sizeof(int));
    cudaMalloc((void **) &qd_d,  (size_t)(D/QI8_1) * sizeof(float2));
    cudaMalloc((void **) &out_d, sizeof(float));
    cudaMemcpy(K_d,  K_host,  (size_t) n_blk_nv * sizeof(block_nvfp4), cudaMemcpyHostToDevice);
    cudaMemcpy(qi_d, qi_host, (size_t)(D/sizeof(int)) * sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(qd_d, qd_host, (size_t)(D/QI8_1) * sizeof(float2),      cudaMemcpyHostToDevice);

    float got = 0.0f;
    switch (D) {
        case  64: gate1_kernel< 64><<<1, 32>>>(K_d, qi_d, qd_d, out_d); break;
        case 128: gate1_kernel<128><<<1, 32>>>(K_d, qi_d, qd_d, out_d); break;
        case 256: gate1_kernel<256><<<1, 32>>>(K_d, qi_d, qd_d, out_d); break;
        default: fprintf(stderr, "bad D %d\n", D); return false;
    }
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA error at D=%d\n", D);
        return false;
    }
    cudaMemcpy(&got, out_d, sizeof(float), cudaMemcpyDeviceToHost);

    const float diff = fabsf(got - ref);
    const float tol  = 1e-3f + 1e-4f*fabsf(ref);
    const bool  pass = diff <= tol;

    if (io_maxdiff != nullptr) {
        *io_maxdiff = fmaxf(*io_maxdiff, diff);
    }

    printf("D=%3d  ref=% .8f  vec=% .8f  diff=%.3e  tol=%.3e  %s\n",
           D, ref, got, diff, tol, pass ? "PASS" : "FAIL");

    cudaFree(K_d); cudaFree(qi_d); cudaFree(qd_d); cudaFree(out_d);
    free(K_host); free(Q_host); free(qi_host); free(qd_host);
    return pass;
}

extern "C" void ggml_cpu_init(void);
int main() {
    ggml_cpu_init();
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        fprintf(stderr, "no CUDA device\n");
        return 2;
    }

    float maxdiff = 0.0f;
    bool ok = true;
    for (int D : {64, 128, 256}) {
        ok = run_one(D, &maxdiff) && ok;
    }

    printf("Gate-1 NVFP4 VEC vs ggml_vec_dot_nvfp4_q8_0_generic: %s (maxdiff=%.3e)\n",
           ok ? "PASS" : "FAIL", maxdiff);
    return ok ? 0 : 1;
}
