// den_cubin_gemv_test.cu — Isolated GEMV test for the standalone NVFP4 OMMA cubin.
//
// Loads nvfp4_omma.cubin via cuModuleLoadData, launches nvfp4_gemv_kernel on
// synthetic NULLGLASS 160B tiles, and compares to a software reference that
// replicates the kernel EXACTLY (E2M1 activation quant, UE4M3 per-16 scales,
// tile norm @152:155). Reports cosine + relative error.
//
// Build:  nvcc -arch=sm_120a -o den_cubin_gemv_test den_cubin_gemv_test.cu -lcuda -lcudart
// Run:    ./den_cubin_gemv_test [path/to/nvfp4_omma.cubin]

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>

#define TILE_BYTES 160

// ── Host fp16 helpers ────────────────────────────────────────────────────
static float h2f(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0) return (sign ? -1.0f : 1.0f) * ldexpf(mant / 1024.0f, -14);
    if (exp == 0x1F) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    return (sign ? -1.0f : 1.0f) * ldexpf(1.0f + mant / 1024.0f, exp - 15);
}
static uint16_t f2h(float v) {
    if (isnan(v)) return 0x7E00;
    if (isinf(v)) return v > 0 ? 0x7C00 : 0xFC00;
    uint32_t sign = 0; if (v < 0) { sign = 0x8000; v = -v; }
    int e; float m = frexpf(v, &e);         // v = m * 2^e, m in [0.5, 1)
    int exp16 = e + 14;
    if (exp16 <= 0) {                       // subnormal
        float s = ldexpf(v, 24);
        uint32_t mbits = (uint32_t)s;       // 10 mantissa bits
        return (uint16_t)(sign | mbits);
    }
    if (exp16 >= 0x1F) return (uint16_t)(sign | 0x7C00);
    uint32_t mant16 = (uint32_t)(m * 2048.0f) - 1024;   // 10 bits
    return (uint16_t)(sign | (exp16 << 10) | (mant16 & 0x3FF));
}

// ── E2M1 helpers (MUST match nvfp4_omma.cu exactly) ─────────────────────
static const float e2m1_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
static uint8_t f32_to_e2m1(float v) {
    if (v == 0.0f) return 0x0;
    uint32_t bits; memcpy(&bits, &v, 4);
    uint8_t sign = (bits >> 28) & 0x8;
    float av = fabsf(v);
    if (av < 0.25f) return sign | 0x0;
    if (av < 0.75f) return sign | 0x1;
    if (av < 1.25f) return sign | 0x2;
    if (av < 1.75f) return sign | 0x3;
    if (av < 2.5f)  return sign | 0x4;
    if (av < 3.5f)  return sign | 0x5;
    if (av < 5.0f)  return sign | 0x6;
    return sign | 0x7;
}

// ── UE4M3 helpers (from den_omma_shared.cuh — proven decode) ────────────
static const uint8_t ue4m3_code_to_byte[16] = {
    0x00, 0x18, 0x20, 0x24, 0x28, 0x2A, 0x2C, 0x2E,
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F
};
static float ue4m3_byte_to_f32(uint8_t b) {
    if (b >= 0x7F) return 0.0f;
    int e = (b >> 3) & 0x0F, m = b & 0x07;
    if (e == 0) return ldexpf((float)m / 8.0f, -7);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}
static uint8_t quant_f32_ue4m3(float v) {
    if (v <= 0.03125f) return 0;
    if (v <= 0.09375f) return 1;
    if (v <= 0.15625f) return 2;
    if (v <= 0.21875f) return 3;
    if (v <= 0.28125f) return 4;
    if (v <= 0.34375f) return 5;
    if (v <= 0.40625f) return 6;
    if (v <= 0.71875f) return 7;
    if (v <= 1.0625f)  return 8;
    if (v <= 1.1875f)  return 9;
    if (v <= 1.3125f)  return 10;
    if (v <= 1.4375f)  return 11;
    if (v <= 1.5625f)  return 12;
    if (v <= 1.6875f)  return 13;
    if (v <= 1.8125f)  return 14;
    return 15;
}

// ── Pack one weight row into a 160B NULLGLASS tile ──────────────────────
// Layout: [0:16] d4[4] UE4M3 scales (byte j -> K[16j:16j+16) of sub-block),
//         [16:144] E2M1 nibbles (element e at byte 16+e/2, nibble (e%2)*4),
//         [148] dispatch 0x10, [152:156] tile norm.
// If use_scales is false, all scale bytes = 0x38 (1.0) and w is quantized /1.0.
static void pack_tile(uint8_t* tile, const float* w, int K, float norm, bool use_scales) {
    memset(tile, 0, TILE_BYTES);
    for (int sub = 0; sub < 4; sub++) {
        uint32_t d4 = 0;
        for (int j = 0; j < 4; j++) {
            float maxa = 0.0f;
            for (int i = 0; i < 16; i++) {
                int k = sub * 64 + j * 16 + i;
                if (k < K) maxa = fmaxf(maxa, fabsf(w[k]));
            }
            uint8_t byte = 0x38;
            if (use_scales) {
                uint8_t code = quant_f32_ue4m3(fmaxf(0.0625f, maxa / 6.0f));
                byte = ue4m3_code_to_byte[code];
            }
            d4 |= (uint32_t)byte << (j * 8);
            float sval = ue4m3_byte_to_f32(byte);
            for (int i = 0; i < 16; i++) {
                int k = sub * 64 + j * 16 + i;
                if (k >= K) continue;
                uint8_t nib = f32_to_e2m1(sval == 0.0f ? 0.0f : w[k] / sval);
                int idx = 16 + k / 2;
                tile[idx] |= (uint8_t)(nib << ((k % 2) * 4));
            }
        }
        *(uint32_t*)(tile + sub * 4) = d4;
    }
    tile[148] = 0x10;
    memcpy(tile + 152, &norm, 4);
}

// ── Software reference (replicates kernel math exactly) ─────────────────
static double ref_gemv_row(const uint8_t* tile, const uint16_t* x, int K, float norm) {
    double acc = 0.0;
    for (int sub = 0; sub < 4; sub++) {
        uint32_t d4 = *(const uint32_t*)(tile + sub * 4);
        for (int j = 0; j < 4; j++) {
            uint8_t byte = (d4 >> (j * 8)) & 0xFF;
            float sval = ue4m3_byte_to_f32(byte);
            for (int i = 0; i < 16; i++) {
                int k = sub * 64 + j * 16 + i;
                if (k >= K) continue;
                uint8_t nib = (tile[16 + k / 2] >> ((k % 2) * 4)) & 0xF;
                float wq = e2m1_lut[nib];
                float xf = h2f(x[k]);
                float xq = e2m1_lut[f32_to_e2m1(xf)];
                acc += (double)wq * (double)sval * (double)xq;
            }
        }
    }
    return acc * (double)norm;
}

// Report cos + relative error for a kernel output vs the software reference.
static void report(const char* label, const std::vector<float>& h_y,
                   const std::vector<float>& h_y_ref, int N, int& global_pass) {
    double num = 0, den_a = 0, den_b = 0, err = 0, ref_abs = 0;
    double max_err = 0; int worst = -1;
    for (int n = 0; n < N; n++) {
        num += (double)h_y[n] * h_y_ref[n];
        den_a += (double)h_y[n] * h_y[n];
        den_b += (double)h_y_ref[n] * h_y_ref[n];
        err += fabs((double)h_y[n] - h_y_ref[n]);
        ref_abs += fabs(h_y_ref[n]);
        double e = fabs((double)h_y[n] - h_y_ref[n]);
        if (e > max_err) { max_err = e; worst = n; }
    }
    double cos = (den_a * den_b > 0) ? num / sqrt(den_a * den_b) : 0.0;
    double rel = (ref_abs > 1e-12) ? err / ref_abs : 0.0;

    printf("  [%s] cos = %.6f  rel_err = %.4f%%\n", label, cos, rel * 100.0);
    printf("         y[0]=%.4f ref[0]=%.4f | y[1]=%.4f ref[1]=%.4f%s\n",
           h_y[0], h_y_ref[0], h_y[1], h_y_ref[1],
           (cos > 0.99) ? "  PASS" : "  FAIL");
    if (worst >= 0)
        printf("         worst: row %d err %.4f (y=%.4f ref=%.4f)\n",
               worst, max_err, h_y[worst], h_y_ref[worst]);
    if (!(cos > 0.99)) global_pass = 0;
}

int main(int argc, char** argv) {
    const char* cubin_path = argc > 1 ? argv[1] : "build_wsl/bin/nvfp4_omma.cubin";

    // Driver init: primary context must be current before cuModuleLoadData.
    cuInit(0);
    CUdevice dev; cuDeviceGet(&dev, 0);
    CUcontext ctx; cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // Load cubin
    FILE* f = fopen(cubin_path, "rb");
    if (!f) { fprintf(stderr, "cannot open cubin %s\n", cubin_path); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* img = (uint8_t*)malloc(sz);
    if (fread(img, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read fail\n"); return 1; }
    fclose(f);

    CUmodule mod; CUfunction fn, fn_batch;
    CUresult cr = cuModuleLoadData(&mod, img);
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "cuModuleLoadData failed %d\n", cr); return 1; }
    cr = cuModuleGetFunction(&fn, mod, "nvfp4_gemv_kernel");
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "cuModuleGetFunction failed %d\n", cr); return 1; }
    cr = cuModuleGetFunction(&fn_batch, mod, "nvfp4_gemv_batch_kernel");
    if (cr != CUDA_SUCCESS) { fprintf(stderr, "batch kernel not found (ok)\n"); fn_batch = nullptr; }
    printf("cubin loaded: %s (%ld bytes)\n", cubin_path, sz);

    // ── Test cases ──
    struct { int N, K; bool scales; } cases[] = {
        { 64, 256,  true },   // realistic per-16 scales, tpr=1
        { 64, 256,  false },  // scale=1.0 (isolates fragment layout)
        { 32, 640,  true },   // tpr=3, K not multiple of 256
    };
    int ncase = 3;
    int global_pass = 1;

    for (int ci = 0; ci < ncase; ci++) {
        int N = cases[ci].N, K = cases[ci].K;
        bool use_scales = cases[ci].scales;
        int tpr = (K + 255) / 256;

        printf("\n=== CASE %d: N=%d K=%d tpr=%d %s ===\n",
               ci + 1, N, K, tpr, use_scales ? "per-16 scales" : "scale=1.0");

        // Generate weights + activations
        std::vector<float> h_w(N * K), h_x(K), h_y_ref(N);
        srand(1234 + ci);
        for (int i = 0; i < N * K; i++) h_w[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        for (int i = 0; i < K; i++)     h_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

        // Pack tiles
        std::vector<uint8_t> h_tiles((size_t)N * tpr * TILE_BYTES);
        float norm = 1.0f;
        for (int n = 0; n < N; n++)
            for (int t = 0; t < tpr; t++)
                pack_tile(&h_tiles[(size_t)(n * tpr + t) * TILE_BYTES],
                          &h_w[(size_t)n * K + t * 256], K - t * 256, norm, use_scales);

        // fp16 activations
        std::vector<uint16_t> h_x16(K);
        for (int k = 0; k < K; k++) h_x16[k] = f2h(h_x[k]);

        // Reference: tile t reads activations at K[t*256 + local] (kernel does the same)
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int t = 0; t < tpr; t++)
                acc += ref_gemv_row(&h_tiles[(size_t)(n * tpr + t) * TILE_BYTES],
                                    &h_x16[t * 256], K - t * 256, norm);
            h_y_ref[n] = (float)acc;
        }

        // Upload once
        uint8_t* d_tiles; uint16_t* d_x; float* d_y;
        cudaMalloc(&d_tiles, h_tiles.size());
        cudaMalloc(&d_x, K * 2);
        cudaMalloc(&d_y, N * 4);
        cudaMemcpy(d_tiles, h_tiles.data(), h_tiles.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x16.data(), K * 2, cudaMemcpyHostToDevice);

        // Run primary kernel
        void* args[] = { &d_tiles, &d_x, &d_y, &N, &K, &tpr };
        cudaMemset(d_y, 0, N * 4);
        cr = cuLaunchKernel(fn, (unsigned)N, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr);
        if (cr != CUDA_SUCCESS) {
            const char* es; cuGetErrorString(cr, &es);
            fprintf(stderr, "  [gemv] launch failed: %s\n", es);
            global_pass = 0;
        } else {
            cudaDeviceSynchronize();
            std::vector<float> h_y(N);
            cudaMemcpy(h_y.data(), d_y, N * 4, cudaMemcpyDeviceToHost);
            report("nvfp4_gemv_kernel", h_y, h_y_ref, N, global_pass);
        }

        // Run batch kernel (row = gridDim.x * blockDim.y + threadIdx.y)
        if (fn_batch) {
            cudaMemset(d_y, 0, N * 4);
            unsigned bdy = 32;
            unsigned gx = (N + bdy - 1) / bdy;
            cr = cuLaunchKernel(fn_batch, gx, 1, 1, 32, bdy, 1, 0, nullptr, args, nullptr);
            if (cr != CUDA_SUCCESS) {
                const char* es; cuGetErrorString(cr, &es);
                fprintf(stderr, "  [batch] launch failed: %s\n", es);
                global_pass = 0;
            } else {
                cudaDeviceSynchronize();
                std::vector<float> h_y(N);
                cudaMemcpy(h_y.data(), d_y, N * 4, cudaMemcpyDeviceToHost);
                report("nvfp4_gemv_batch_kernel", h_y, h_y_ref, N, global_pass);
            }
        }

        cudaFree(d_tiles); cudaFree(d_x); cudaFree(d_y);
    }

    printf("\n=== RESULT: %s ===\n", global_pass ? "ALL PASS" : "FAILURES PRESENT");
    return global_pass ? 0 : 1;
}
