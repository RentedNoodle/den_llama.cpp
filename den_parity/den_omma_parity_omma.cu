// den_omma_parity_omma.cu
// Isolated TU for the OMMA inline-asm kernel (Rule 7.9: no OMMA PTX
// co-located with normal CUDA code). Exposes a host-callable wrapper.
#include "ggml-cuda/den_mxf4nvf4_gemv.cuh"

// g_ue4m3_full_decode is static __constant__ per-TU (declared in
// den_omma_shared.cuh). In production ggml-cuda.cu's TU populates ITS copy;
// this TU's copy is zero-initialized unless we fill it here.
// EXPERIMENT: populate the FULL 256-entry UE4M3 decode (all valid bytes
// 0x00..0x7E), not just the 16 canonical code_to_byte values. Real weight
// scale bytes span the full UE4M3 range (e.g. 0x64 = 48.0), so the 16-entry
// version makes den_scale_product() return 0 for them.
extern "C" void den_omma_parity_scale_init(void) {
#ifdef DEN_USE_CONSTANT_SCALE_TABLE
    float decode[256] = {0};
    for (int b = 0; b < 256; b++) {
        if (b >= 0x7F) { decode[b] = 0.0f; continue; }
        int exp = (b >> 3) & 0x0F;
        int mant = b & 0x07;
        if (exp == 0) decode[b] = ldexpf((float)mant / 8.0f, -7);
        else          decode[b] = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
    cudaMemcpyToSymbol(g_ue4m3_full_decode, decode, sizeof(decode));
    float rb[256];
    cudaMemcpyFromSymbol(rb, g_ue4m3_full_decode, sizeof(rb));
    fprintf(stderr, "[parity] FULL scale table init: rb[0x64]=%.6g rb[0x2A]=%.6g rb[0x38]=%.6g rb[0x3E]=%.6g\n",
            rb[0x64], rb[0x2A], rb[0x38], rb[0x3E]);
#endif
}

extern "C" void run_omma_gemv(
    const float * weights, const float * act, float * dst,
    int N, int K, cudaStream_t stream,
    const float * tile_norms, int n_norms)
{
    // weights points to 160B NULLGLASS blocks (host buffer, caller uploads).
    // Tile norm applied via tile_norms registry (n_norms==1 -> broadcast).
    den_mxf4nvf4_gemv_launch(weights, act, dst, N, K, stream, tile_norms, n_norms, /*fused_rmsnorm*/ false, 1e-6f);
}
