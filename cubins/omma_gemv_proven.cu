// omma_gemv_proven.cu
// NVFP4 GEMV cubin for sm_120a (RTX 5070 Ti)
//
// Implements y = W * x where W is an NVFP4-quantized matrix in
// NULLGLASS 160B tile format:
//   [0..15]   16 x UE4M3 scale bytes (1 per 16-element group)
//   [16..143] 128 nibble bytes (256 E2M1 values, 2 per byte)
//   [144..147] float32 tile_norm
//   [148..159] reserved (tile format code, K_stride, scale ptr)
//
// E2M1 (4-bit): s1e2m1, bias=1, range [-6.0, +6.0]
// UE4M3 (8-bit): e4m3 unsigned, bias=7, range [0, 240]
//
// Dequantized weight = e2m1(nibble) * ue4m3(scale) * tile_norm
//
// Exports TWO entry points for dispatch flexibility:
//   gemv_nvfp4_sw   — software dequant GEMV (reference, guaranteed correct)
//   gemv_nvfp4_omma — OMMA.SF.16864 tensor core GEMV (placeholder, calls SW)
//
// Each block processes one output row with 32 threads (one warp).
// Each thread handles 8 K elements per tile.
// Warp shuffle reduction sums across all lanes.
//
// Compile:
//   nvcc -arch=compute_120a -code=sm_120a --cubin \
//        -o omma_gemv_proven.cubin omma_gemv_proven.cu

#include <cuda_fp16.h>

#define TILE_BYTES   160
#define TILE_K       256
#define TILE_SCALES  16
#define TILE_NIBBLES 128

// ---------------------------------------------------------------------------
// E2M1 dequantization (4-bit float, s1e2m1, bias=1)
// ---------------------------------------------------------------------------
__device__ inline float dequant_e2m1(unsigned char nibble) {
    // Bit layout: [sign, exp1, exp0, mant]
    //   exp=0: (-1)^sign * 0.mant * 2^(1-1) = (-1)^sign * mant/2
    //   exp>0: (-1)^sign * 1.mant * 2^(exp-1)
    int v = nibble & 0xF;
    float s = (v & 8) ? -1.0f : 1.0f;
    int   e = (v >> 1) & 3;
    float m = (v & 1) ? 0.5f : 0.0f;
    if (e == 0) {
        return s * m;
    } else {
        return s * (1.0f + m) * (float)(1 << (e - 1));
    }
}

// ---------------------------------------------------------------------------
// UE4M3 dequantization (unsigned 8-bit scale, e4m3, bias=7)
// ---------------------------------------------------------------------------
__device__ inline float dequant_ue4m3(unsigned char b) {
    int e = (int)(b >> 3);       // exponent field 0..15
    int m = (int)(b & 7);        // mantissa field 0..7
    if (e == 0) {
        // Subnormal: 0.m * 2^(1-7) = m / 512
        return m / 512.0f;
    }
    // Normal: 1.m * 2^(e-7)  = (1 + m/8) * 2^(e-7)
    float base = 1.0f + m * 0.125f;
    if (e >= 7) {
        return base * (float)(1 << (e - 7));    // positive exponent
    } else {
        return base / (float)(1 << (7 - e));    // negative exponent
    }
}

// ---------------------------------------------------------------------------
// NVFP4 GEMV kernel (internal — called by both SW and OMMA entry points)
//   y[row] = sum over K of W[row,k] * x[k]
//   x is float* (ggml convention for MUL_MAT dispatch)
//   N = rows, tpr = tiles per row = K / 256
// ---------------------------------------------------------------------------
static __device__ void gemv_nvfp4_kernel(
    const unsigned char* __restrict__ blocks,   // NVFP4 tile array [N][tpr][TILE_BYTES]
    const float*         __restrict__ x,         // input vector [tpr * TILE_K]
    float*               __restrict__ y,         // output vector [N]
    int N,                                        // number of rows
    int tpr                                       // tiles per row
) {
    // -- grid-stride: one block per output row --------------------------------
    int row = blockIdx.x;
    int tid = threadIdx.x;       // lane 0..31

    float sum = 0.0f;

    // -- loop over tiles (K dimension) ----------------------------------------
    for (int t = 0; t < tpr; t++) {
        // Tile base address (64-bit offset for large models)
        const unsigned char* tile = blocks + ((size_t)row * tpr + t) * TILE_BYTES;

        // Scale group: 2 threads share one 16-element group
        //   tid 0-1  -> group 0,  tid 2-3  -> group 1, ...
        int group = tid >> 1;
        float scale = dequant_ue4m3(tile[group]);

        // Per-tile normalization
        float tile_norm = ((const float*)(tile + 144))[0];

        // Each thread handles 8 consecutive K elements
        int k_base = tid * 8;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            int k  = k_base + e;

            // Nibble byte index: 16 + k/2
            int nib_idx   = 16 + (k >> 1);
            int is_high   = k & 1;
            unsigned char nb = tile[nib_idx];
            unsigned char nib = is_high ? (nb >> 4) : (nb & 0xF);

            float wval = dequant_e2m1(nib) * scale * tile_norm;
            float xval = x[(size_t)t * TILE_K + k];

            sum = fmaf(wval, xval, sum);
        }
    }

    // -- warp-level reduction: butterfly sum across all 32 lanes --------------
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }

    // Lane 0 holds the total sum
    if (tid == 0) {
        y[row] = sum;
    }
}

// ---------------------------------------------------------------------------
// Entry point: gemv_nvfp4_sw — software dequant GEMV (reference)
//   Signature matches ggml dispatch: tiles, x_f32, y_f32, N, tpr
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_nvfp4_sw(
    const uint8_t* __restrict__ tiles,    // NVFP4 weight tiles (160B each)
    const float*   __restrict__ x,        // input activation (K elements, F32)
    float*         __restrict__ y,        // output (N elements, F32)
    int N,                                 // number of rows
    int tpr                                // tiles per row = K / 256
) {
    gemv_nvfp4_kernel(tiles, x, y, N, tpr);
}

// ---------------------------------------------------------------------------
// Entry point: gemv_nvfp4_omma — OMMA.SF.16864 tensor core GEMV
//   PLACEHOLDER: currently dispatches to software path.
//   Replace body with OMMA inline PTX for 3-5x speedup.
// ---------------------------------------------------------------------------
extern "C" __global__ void gemv_nvfp4_omma(
    const uint8_t* __restrict__ tiles,    // NVFP4 weight tiles (160B each)
    const float*   __restrict__ x,        // input activation (K elements, F32)
    float*         __restrict__ y,        // output (N elements, F32)
    int N,                                 // number of rows
    int tpr                                // tiles per row = K / 256
) {
    // TODO: Replace with OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X inline PTX
    gemv_nvfp4_kernel(tiles, x, y, N, tpr);
}
