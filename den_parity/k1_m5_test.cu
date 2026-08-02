// k1_m5_test.cu — test the M>1 k1_dense path on real attn_qkv data
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
void den_k1_dense_dispatch(const void*, const float*, float*, int M, int N, int K, cudaStream_t, const float*, int n_norms, bool, float);

static const float KVAL[16] = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12};
static float ue4m3_to_f32(uint8_t code) {
    if (code >= 0x7F) return 0.0f;
    int exp = (code >> 3) & 0x0F; int mant = code & 0x07;
    if (exp == 0) return ldexpf((float)mant / 8.0f, -7);
    return ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
}
static float scale_gguf(uint8_t code) { return ue4m3_to_f32(code) * 0.5f; }
// SW reference for row r of token m: reads norm at 152:155
static double sw_row(const uint8_t* w, const float* x, int row, int K) {
    const int kt_per_row = K / 256;
    double sum = 0.0;
    const uint8_t* base = w + (size_t)row * kt_per_row * 160;
    for (int kt = 0; kt < kt_per_row; ++kt) {
        const uint8_t* blk = base + (size_t)kt * 160;
        float tile_norm; memcpy(&tile_norm, blk + 152, 4);
        for (int k = 0; k < 256; ++k) {
            float scale = scale_gguf(blk[k / 16]);
            uint8_t nib = (uint8_t)((blk[16 + k/2] >> ((k % 2) * 4)) & 0xF);
            sum += (double)(KVAL[nib] * scale * tile_norm) * x[kt*256 + k];
        }
    }
    return sum;
}
static double cos_sim(const float* a, const float* b, int n) {
    double dot=0, na=0, nb=0;
    for (int i=0;i<n;++i){ dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return dot / (sqrt(na)*sqrt(nb));
}
int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/root/den_final/den_parity/attn_qkv_rt.bin";
    FILE* f = fopen(path, "rb");
    int N, K, nblocks, block_bytes; float global_scale;
    fread(&N,4,1,f); fread(&K,4,1,f); fread(&nblocks,4,1,f); fread(&block_bytes,4,1,f); fread(&global_scale,4,1,f);
    std::vector<uint8_t> w(nblocks * 160);
    fread(w.data(), 1, w.size(), f); fclose(f);
    printf("bin: N=%d K=%d global=%.9g\n", N, K, global_scale);

    const int M = 5;
    std::vector<float> x(M * K), y_sw(M * N), y_gpu(M * N);
    srand(777);
    for (int m = 0; m < M; ++m) for (int i = 0; i < K; ++i) x[m*K+i] = ((float)rand()/RAND_MAX - 0.5f);
    for (int m = 0; m < M; ++m) for (int r = 0; r < N; ++r) y_sw[m*N+r] = (float)sw_row(w.data(), &x[m*K], r, K);

    uint8_t* w_dev; float* x_dev; float* y_dev; float* tn_dev;
    CK(cudaMalloc(&w_dev, w.size()));
    CK(cudaMalloc(&x_dev, M*K*sizeof(float)));
    CK(cudaMalloc(&y_dev, M*N*sizeof(float)));
    CK(cudaMalloc(&tn_dev, sizeof(float)));
    CK(cudaMemcpy(w_dev, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x_dev, x.data(), M*K*sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(tn_dev, &global_scale, sizeof(float), cudaMemcpyHostToDevice));

    den_k1_dense_dispatch(w_dev, x_dev, y_dev, M, N, K, 0, tn_dev, 1, false, 1e-6f);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(y_gpu.data(), y_dev, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    for (int m = 0; m < M; ++m) {
        double cos = cos_sim(&y_gpu[m*N], &y_sw[m*N], N);
        // per-row ratio stats
        double sum_ratio = 0; int nbad = 0; double maxerr = 0; int worst = -1;
        for (int r = 0; r < N; ++r) {
            double e = fabs(y_gpu[m*N+r] - y_sw[m*N+r]);
            if (e > maxerr) { maxerr = e; worst = r; }
            double ratio = fabs(y_sw[m*N+r]) > 1e-6 ? y_gpu[m*N+r]/y_sw[m*N+r] : 0;
            if (ratio < 0.5 || ratio > 1.5) nbad++;
        }
        printf("M=%d cos=%.6f nbad_rows(ratio out 0.5-1.5)=%d/%d maxerr=%.6g worst_row=%d (sw=%.6g gpu=%.6g)\n",
               m, cos, nbad, N, maxerr, worst, y_sw[m*N+worst], y_gpu[m*N+worst]);
        printf("   y_sw[0..7]= "); for (int r=0;r<8;r++) printf("%.5g ", y_sw[m*N+r]); printf("\n");
        printf("   y_gp[0..7]= "); for (int r=0;r<8;r++) printf("%.5g ", y_gpu[m*N+r]); printf("\n");
        printf("   y_sw[8..15]="); for (int r=8;r<16;r++) printf("%.5g ", y_sw[m*N+r]); printf("\n");
        printf("   y_gp[8..15]="); for (int r=8;r<16;r++) printf("%.5g ", y_gpu[m*N+r]); printf("\n");
    }
    return 0;
}
