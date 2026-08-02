// den_omma_parity.cu
// Decisive MMQ(software)-vs-OMMA GEMV parity test for a single NVFP4 tensor.
//
// Reads a .bin produced by den_extract_nvfp4_tensor.py:
//   header: int N, int K, int nblocks, int block_bytes, float global_scale
//   data:   nblocks * 160 NULLGLASS NVFP4 blocks
//
// Runs:
//   1. CPU software dequant GEMV  (the "MMQ" reference, matches convert.cu dequant)
//   2. GPU OMMA.SF.16864 GEMV      (den_mxf4nvf4_gemv_launch, production kernel)
// then reports cosine, max-abs-err, ratio, and the first 8 outputs of each.
//
// Compile:
//   nvcc -arch=compute_120a -code=sm_120a \
//        -I<den_final>/ggml/include -I<den_final>/ggml/src \
//        -I<den_final>/ggml/src/ggml-cuda \
//        den_omma_parity.cu den_omma_parity_omma.cu -o den_omma_parity
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

extern "C" void run_omma_gemv(const float* w, const float* act, float* dst,
                              int N, int K, cudaStream_t stream,
                              const float* tile_norms, int n_norms);
extern "C" void den_omma_parity_scale_init(void);

// E2M1 decode table — MUST match kvalues_mxfp4 in ggml-common.h (DOUBLED table)
static const float KVAL[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};

// UE4M3 unsigned decode — MUST match ggml_cuda_ue4m3_to_fp32 in common.cuh
static float ue4m3_to_f32(uint8_t code) {
    if (code >= 0x7F) return 0.0f;
    int exp = (code >> 3) & 0x0F;
    int mant = code & 0x07;
    if (exp == 0) return ldexpf((float)mant / 8.0f, -7);
    return ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
}

// The GGUF "kvalues convention" (verified cos 0.99996 vs HF in verify_nvfp4.py):
//   value = ue4m3_to_f32(scale) * 0.5f  *  KVAL_doubled[nib]
// The *0.5 on the scale and the doubled E2M1 cancel to standard E2M1.
static float scale_gguf(uint8_t code) {
    return ue4m3_to_f32(code) * 0.5f;
}

// Software reference: per-row dot of dequantized weight with activation.
// block[k] for element k in [0,256): scale=block[k/16], nib=block[16+k/2]>>((k%2)*4)
// then * tile_norm (float at bytes 144-147).
static void software_gemv(const uint8_t* w, const float* x, float* y, int N, int K) {
    const int kt_per_row = K / 256;
    for (int row = 0; row < N; ++row) {
        double sum = 0.0;
        const uint8_t* base = w + (size_t)row * kt_per_row * 160;
        for (int kt = 0; kt < kt_per_row; ++kt) {
            const uint8_t* blk = base + (size_t)kt * 160;
            float tile_norm;
            memcpy(&tile_norm, blk + 152, 4);
            const int k0 = kt * 256;
            for (int k = 0; k < 256; ++k) {
                float scale = scale_gguf(blk[k / 16]);
                uint8_t nib = (uint8_t)((blk[16 + k/2] >> ((k % 2) * 4)) & 0xF);
                float wval = KVAL[nib] * scale * tile_norm;
                sum += (double)wval * x[k0 + k];
            }
        }
        y[row] = (float)sum;
    }
}

static double cos_sim(const float* a, const float* b, int n) {
    double dot=0, na=0, nb=0;
    for (int i=0;i<n;++i){ dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return dot / (sqrt(na)*sqrt(nb));
}

static void stats(const float* a, const float* b, int n, const char* label) {
    double max_abs = 0, sum_abs_a = 0, sum_abs_b = 0;
    double best_ratio = 0;
    for (int i=0;i<n;++i){ double d=fabs((double)a[i]-b[i]); if(d>max_abs)max_abs=d; sum_abs_a+=fabs(a[i]); sum_abs_b+=fabs(b[i]); }
    double ratio = sum_abs_b > 0 ? sum_abs_a/sum_abs_b : 0.0;
    printf("%-28s cos=%.8f  max_abs_err=%.6g  sum|a|/sum|b|=%.6f  (n=%d)\n", label, cos_sim(a,b,n), max_abs, ratio, n);
    (void)best_ratio;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/root/den_final/den_parity/attn_qkv.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    int N, K, nblocks, block_bytes;
    float global_scale;
    if (fread(&N,4,1,f)!=1 || fread(&K,4,1,f)!=1 || fread(&nblocks,4,1,f)!=1 ||
        fread(&block_bytes,4,1,f)!=1 || fread(&global_scale,4,1,f)!=1) {
        fprintf(stderr, "bad header\n"); return 1;
    }
    printf("bin: N=%d K=%d nblocks=%d block_bytes=%d global_scale=%.9g\n", N, K, nblocks, block_bytes, global_scale);

    std::vector<uint8_t> w(nblocks * 160);
    if (fread(w.data(), 1, w.size(), f) != w.size()) { fprintf(stderr, "bad data\n"); return 1; }
    fclose(f);

    const int kt_per_row = K / 256;

    // Populate this TU's g_ue4m3_full_decode constant table (production
    // initializes it in ggml-cuda.cu's TU; ours is zeroed otherwise).
    den_omma_parity_scale_init();

    // Deterministic activation
    std::vector<float> x(K);
    uint32_t seed = 12345;
    for (int i = 0; i < K; ++i) {
        seed = seed * 1664525u + 1013904223u;
        float u = (float)((seed >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f;
        x[i] = u * 0.5f;
    }

    // Software reference
    std::vector<float> y_sw(N);
    software_gemv(w.data(), x.data(), y_sw.data(), N, K);
    printf("SW  y[0..7] =");
    for (int i=0;i<8&&i<N;++i) printf(" %.6g", y_sw[i]);
    printf("\n");

    // GPU buffers
    uint8_t* w_dev; float* x_dev; float* y_dev; float* tn_dev;
    CK(cudaMalloc(&w_dev, w.size()));
    CK(cudaMalloc(&x_dev, K*sizeof(float)));
    CK(cudaMalloc(&y_dev, N*sizeof(float)));
    CK(cudaMalloc(&tn_dev, sizeof(float)));
    CK(cudaMemcpy(w_dev, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x_dev, x.data(), K*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tn_dev, &global_scale, sizeof(float), cudaMemcpyHostToDevice));

    // OMMA run with tile_norms = {global_scale} (n=1 -> broadcast, matches production)
    std::vector<float> y_omma(N);
    run_omma_gemv((const float*)w_dev, x_dev, y_dev, N, K, 0, tn_dev, 1);
    CK(cudaMemcpy(y_omma.data(), y_dev, N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("OMMA y[0..7] =");
    for (int i=0;i<8&&i<N;++i) printf(" %.6g", y_omma[i]);
    printf("\n");

    // OMMA run without tile_norms (nullptr) to check tile-norm placement
    std::vector<float> y_omma_non(N);
    run_omma_gemv((const float*)w_dev, x_dev, y_dev, N, K, 0, nullptr, 0);
    CK(cudaMemcpy(y_omma_non.data(), y_dev, N*sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n=== RESULTS ===\n");
    stats(y_sw.data(), y_sw.data(), N, "SW vs SW (sanity)");
    stats(y_omma.data(), y_sw.data(), N, "OMMA(tn)  vs SW");
    stats(y_omma_non.data(), y_sw.data(), N, "OMMA(none)vs SW");

    // Ratio per element on a few rows
    printf("\n=== ROW-BY-ROW RATIO (omma/sw) first 16 rows ===\n");
    for (int r = 0; r < 16 && r < N; ++r) {
        double a = y_omma[r], b = y_sw[r];
        printf("row %4d: sw=% .6g  omma=% .6g  ratio=%.6f\n", r, b, a, b != 0 ? a/b : 0.0);
    }

    // Peek at scattered indices to detect permutation patterns
    printf("\n=== SCATTER CHECK (sw vs omma) ===\n");
    const int idxs[12] = {0,1,2,3,7,15,31,63,127,255,511,1023};
    for (int ii = 0; ii < 12 && idxs[ii] < N; ++ii) {
        int i = idxs[ii];
        printf("idx %4d: sw=% .6g  omma=% .6g\n", i, y_sw[i], y_omma[i]);
    }

    // ── ONE-HOT PERMUTATION PROBE ──
    // Feed x = e_k0 (single 1.0 at position k0). Software output[row] = W[row,k0].
    // If OMMA's lane mapping is permuted, OMMA[row] will equal W[row, perm(k0)].
    if (argc > 2 && strcmp(argv[2], "probe") == 0) {
        printf("\n=== ONE-HOT PERMUTATION PROBE ===\n");
        const int probe_pos[16] = {0,1,2,3,4,5,6,7,8,15,16,31,32,33,64,65};
        std::vector<float> x1(K, 0.0f);
        std::vector<float> y_sw1(N), y_o1(N);
        for (int p = 0; p < 16; ++p) {
            int k0 = probe_pos[p];
            if (k0 >= K) continue;
            std::fill(x1.begin(), x1.end(), 0.0f);
            x1[k0] = 1.0f;
            software_gemv(w.data(), x1.data(), y_sw1.data(), N, K);
            CK(cudaMemcpy(x_dev, x1.data(), K*sizeof(float), cudaMemcpyHostToDevice));
            run_omma_gemv((const float*)w_dev, x_dev, y_dev, N, K, 0, tn_dev, 1);
            CK(cudaMemcpy(y_o1.data(), y_dev, N*sizeof(float), cudaMemcpyDeviceToHost));
            // Compare first 4 rows: sw = W[row, k0]; omma = ?
            printf("k0=%3d: row0 sw=% .6g omma=% .6g | row1 sw=% .6g omma=% .6g | "
                   "row2 sw=% .6g omma=% .6g | row3 sw=% .6g omma=% .6g\n",
                   k0, y_sw1[0], y_o1[0], y_sw1[1], y_o1[1], y_sw1[2], y_o1[2], y_sw1[3], y_o1[3]);
        }
    }

    CK(cudaFree(w_dev)); CK(cudaFree(x_dev)); CK(cudaFree(y_dev)); CK(cudaFree(tn_dev));
    return 0;
}
