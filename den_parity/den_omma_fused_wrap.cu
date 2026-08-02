#include "ggml-cuda/den_mxf4nvf4_gemv.cuh"
extern "C" void run_omma_gemv_fused(const float* w, const float* act, float* dst, int N, int K, cudaStream_t s, const float* tn, int n, float eps) {
    den_mxf4nvf4_gemv_launch(w, act, dst, N, K, s, tn, n, /*fused_rmsnorm*/true, eps);
}
