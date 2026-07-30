// nvfp4_omma.cu — NVFP4 OMMA.SF.16864 GEMV cubin for ik_llama.cpp
//
// Standalone cubin — NO GGML HEADER INCLUDES (PTX isolation per dengine Rule 7.9).
// Compiled via: nvcc -arch=compute_120a -code=sm_120a --cubin -o nvfp4_omma.cubin nvfp4_omma.cu
// Loaded at runtime via cuModuleLoadData() from ggml-cuda NVFP4 dispatch.
//
// Ported from dengine omma_gemv_proven.cu (SASS-proven on RTX 5070 Ti sm_120a).
// Identity test: all-1.0 weights → every lane = 64.0 (K=64 sum).
//
// OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X
// 3-operand scale format: {sfa},{bid,tid},{sfb},{bid,tid} with "h" constraints
// Fragment layout: K_a=lane, K_b=lane+32, nibbles=rows

#include <cuda_runtime.h>
#include <cstdint>
#include <cuda_fp16.h>

#define TILE_BYTES   160
#define TILE_ELEMS   256
#define WARP          32

// ── E2M1 decode LUT (4-bit nibble → float) ──────────────────────────────
__device__ __constant__ float e2m1_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// ── E2M1 quantize (float → 4-bit nibble) ────────────────────────────────
__device__ __forceinline__ uint8_t f32_to_e2m1(float v) {
    if (v == 0.0f) return 0x0;
    uint32_t bits = __float_as_uint(v);
    uint8_t sign = (bits >> 28) & 0x8;
    float av = fabsf(v);
    if (av < 0.25f)      return sign | 0x0;
    if (av < 0.75f)      return sign | 0x1;
    if (av < 1.25f)      return sign | 0x2;
    if (av < 1.75f)      return sign | 0x3;
    if (av < 2.5f)       return sign | 0x4;
    if (av < 3.5f)       return sign | 0x5;
    if (av < 5.0f)       return sign | 0x6;
    return sign | 0x7;
}

// ── half → float helper (CUDA intrinsic) ────────────────────────────────
__device__ __forceinline__ float half_to_float(uint16_t h) {
    return __half2float(*(const __half*)&h);
}

// Forward declaration — noinline OMMA wrapper (prevents register conflicts)
__device__ __noinline__ void omma_step(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,float&,float&,float&,float&);

// ═══════════════════════════════════════════════════════════════════════════════
// nvfp4_gemv_kernel — OMMA-accelerated NVFP4 GEMV, 1 warp per row
//
// Tile layout (160B NULLGLASS, matching dengine den_format.h):
//   [0:15]     — UE4M3 block scales (16 bytes, 4x uint32 packed per sub-tile)
//   [16:143]   — E2M1 weight nibbles (256 elements × 4 bits = 128 bytes)
//   [144:147]  — tile_norm (float32)
//   [148]      — format/dispatch byte
//   [149]      — flags (8=WH4 v1 needs /16 correction, 9=WH4 v2 pre-folded)
//   [150:159]  — reserved
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" __global__ void nvfp4_gemv_kernel(
    const uint8_t* __restrict__ tiles,   // [N/tpr][160] tile data
    const uint16_t* __restrict__ x,      // [K] input activation in fp16
    float*         __restrict__ y,       // [N] output
    int N, int K, int tpr)               // rows, cols, tiles per row
{
    int row = blockIdx.x;
    if (row >= N) return;

    int lane = threadIdx.x;
    int kgroup = lane / 8;
    int within = lane % 4;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    for (int t = 0; t < tpr; t++) {
        const uint8_t* tile = tiles + ((size_t)row * tpr + t) * TILE_BYTES;

        // Read packed scales (4x uint32 at tile[0:15])
        uint32_t packed_scales[4];
        for (int s = 0; s < 4; s++) {
            packed_scales[s] = *(const uint32_t*)(tile + s * 4);
        }

        for (int sub = 0; sub < 4; sub++) {
            int k_off = sub * 64;

            // ── Build A fragment (weight E2M1 nibbles → uint32 registers) ──
            volatile uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            (void)(k_off + kgroup * 8);         // k_start_a0 — reserved for multi-row GEMV
            (void)(k_off + 32 + kgroup * 8);    // k_start_a1 — reserved for multi-row GEMV

            for (int ni = 0; ni < 8; ni++) {
                int byte_a0 = 16 + sub * 32 + (kgroup * 8 + ni) / 2;
                int nib_a0 = (kgroup * 8 + ni) & 1;
                uint8_t w_a0 = (byte_a0 < 144) ? ((tile[byte_a0] >> (nib_a0 * 4)) & 0xF) : 0;
                a0 |= (uint32_t)w_a0 << (ni * 4);

                int byte_a1 = 16 + sub * 32 + (kgroup * 8 + ni + 32) / 2;
                int nib_a1 = (kgroup * 8 + ni + 32) & 1;
                uint8_t w_a1 = (byte_a1 < 144) ? ((tile[byte_a1] >> (nib_a1 * 4)) & 0xF) : 0;
                a1 |= (uint32_t)w_a1 << (ni * 4);

                a2 |= (uint32_t)w_a0 << (ni * 4);
                a3 |= (uint32_t)w_a1 << (ni * 4);
            }

            // ── Build B fragment (fp16 input → E2M1 → uint32 registers) ──
            volatile uint32_t b0 = 0, b1 = 0;
            int ks = within * 16;
            for (int ni = 0; ni < 8; ni++) {
                int ki0 = k_off + ks + ni;
                float xf0 = (ki0 < K) ? half_to_float(x[ki0]) : 0.0f;
                b0 |= (uint32_t)f32_to_e2m1(xf0) << (ni * 4);

                int ki1 = k_off + ks + 8 + ni;
                float xf1 = (ki1 < K) ? half_to_float(x[ki1]) : 0.0f;
                b1 |= (uint32_t)f32_to_e2m1(xf1) << (ni * 4);
            }

            // ── OMMA.SF.16864 — THE PROVEN PTX ──────────────────────────
            uint32_t sfa = packed_scales[sub];
            uint32_t sfb = 0x38383838u;  // activation scale = 1.0
            uint16_t bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;

            volatile float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
                  "r"(b0),"r"(b1),
                  "f"(c0),"f"(c1),"f"(c2),"f"(c3),
                  "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b)
            : "memory");

            d0 += c0; d1 += c1; d2 += c2; d3 += c3;
        }

        // ── Apply per-tile norm correction ──────────────────────────────
        float tile_norm = *(const float*)(tile + 144);
        float norm_correction = tile_norm;
        if (tile[149] == 8) norm_correction /= 16.0f;  // WH4 v1: Hadamard correction
        if (norm_correction != 0.0f && norm_correction != 1.0f) {
            d0 *= norm_correction;
            d1 *= norm_correction;
            d2 *= norm_correction;
            d3 *= norm_correction;
        }
    }

    if (lane == 0) {
        y[row] = d0;
    }
}

// ── Batch GEMV: multiple rows per block (for decode with batch > 1) ──────────
extern "C" __global__ void nvfp4_gemv_batch_kernel(
    const uint8_t* __restrict__ tiles,
    const uint16_t* __restrict__ x,
    float*         __restrict__ y,
    int N, int K, int tpr)
{
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;

    int lane = threadIdx.x;
    int kgroup = lane / 8;
    int within = lane % 4;

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    for (int t = 0; t < tpr; t++) {
        const uint8_t* tile = tiles + ((size_t)row * tpr + t) * TILE_BYTES;

        uint32_t packed_scales[4];
        for (int s = 0; s < 4; s++) {
            packed_scales[s] = *(const uint32_t*)(tile + s * 4);
        }

        for (int sub = 0; sub < 4; sub++) {
            int k_off = sub * 64;

            volatile uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int ni = 0; ni < 8; ni++) {
                int byte_a0 = 16 + sub * 32 + (kgroup * 8 + ni) / 2;
                int nib_a0 = (kgroup * 8 + ni) & 1;
                uint8_t w_a0 = (byte_a0 < 144) ? ((tile[byte_a0] >> (nib_a0 * 4)) & 0xF) : 0;
                a0 |= (uint32_t)w_a0 << (ni * 4);

                int byte_a1 = 16 + sub * 32 + (kgroup * 8 + ni + 32) / 2;
                int nib_a1 = (kgroup * 8 + ni + 32) & 1;
                uint8_t w_a1 = (byte_a1 < 144) ? ((tile[byte_a1] >> (nib_a1 * 4)) & 0xF) : 0;
                a1 |= (uint32_t)w_a1 << (ni * 4);

                a2 |= (uint32_t)w_a0 << (ni * 4);
                a3 |= (uint32_t)w_a1 << (ni * 4);
            }

            volatile uint32_t b0 = 0, b1 = 0;
            int ks = within * 16;
            for (int ni = 0; ni < 8; ni++) {
                int ki0 = k_off + ks + ni;
                float xf0 = (ki0 < K) ? half_to_float(x[ki0]) : 0.0f;
                b0 |= (uint32_t)f32_to_e2m1(xf0) << (ni * 4);

                int ki1 = k_off + ks + 8 + ni;
                float xf1 = (ki1 < K) ? half_to_float(x[ki1]) : 0.0f;
                b1 |= (uint32_t)f32_to_e2m1(xf1) << (ni * 4);
            }

            uint32_t sfa = packed_scales[sub];
            uint32_t sfb = 0x38383838u;
            uint16_t bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;

            volatile float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
                  "r"(b0),"r"(b1),
                  "f"(c0),"f"(c1),"f"(c2),"f"(c3),
                  "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b)
            : "memory");

            d0 += c0; d1 += c1; d2 += c2; d3 += c3;
        }

        float tile_norm = *(const float*)(tile + 144);
        float norm_correction = tile_norm;
        if (tile[149] == 8) norm_correction /= 16.0f;
        if (norm_correction != 0.0f && norm_correction != 1.0f) {
            d0 *= norm_correction;
            d1 *= norm_correction;
            d2 *= norm_correction;
            d3 *= norm_correction;
        }
    }

    if (lane == 0) {
        y[row] = d0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FUSED KERNEL — GGML block_nvfp4 → OMMA.SF.16864 (Innovation #1)
// Reads GGML block_nvfp4 directly (146 bytes). Eliminates the separate
// convert_blocks_to_tiles_kernel launch — one kernel instead of two.
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" __global__ void nvfp4_gemv_fused_kernel(
    const uint8_t* __restrict__ blocks,  // [N*tpr][146] block_nvfp4
    const uint16_t* __restrict__ x,      // [K] input fp16
    float*         __restrict__ y,       // [N] output
    int N, int K, int tpr)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x;
    int kgroup = lane / 8, within = lane % 4;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    constexpr int BLOCK_BYTES = 146;

    for (int t = 0; t < tpr; t++) {
        const uint8_t* blk = blocks + ((size_t)row * tpr + t) * BLOCK_BYTES;
        float tile_norm = __half2float(*(const __half*)blk);
        uint32_t packed_scales[4];
        for (int s = 0; s < 4; s++)
            packed_scales[s] = *(const uint32_t*)(blk + 2 + s * 4);

        for (int sub = 0; sub < 4; sub++) {
            int k_off = sub * 64;
            volatile uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int ni = 0; ni < 8; ni++) {
                int byte_a0 = 18 + sub * 32 + (kgroup * 8 + ni) / 2;
                int nib_a0 = (kgroup * 8 + ni) & 1;
                uint8_t w_a0 = (byte_a0 < 146) ? ((blk[byte_a0] >> (nib_a0 * 4)) & 0xF) : 0;
                a0 |= (uint32_t)w_a0 << (ni * 4);
                int byte_a1 = 18 + sub * 32 + (kgroup * 8 + ni + 32) / 2;
                int nib_a1 = (kgroup * 8 + ni + 32) & 1;
                uint8_t w_a1 = (byte_a1 < 146) ? ((blk[byte_a1] >> (nib_a1 * 4)) & 0xF) : 0;
                a1 |= (uint32_t)w_a1 << (ni * 4);
                a2 |= (uint32_t)w_a0 << (ni * 4);
                a3 |= (uint32_t)w_a1 << (ni * 4);
            }
            volatile uint32_t b0 = 0, b1 = 0;
            int ks = within * 16;
            for (int ni = 0; ni < 8; ni++) {
                int ki0 = k_off + ks + ni;
                float xf0 = (ki0 < K) ? half_to_float(x[ki0]) : 0;
                b0 |= (uint32_t)f32_to_e2m1(xf0) << (ni * 4);
                int ki1 = k_off + ks + 8 + ni;
                float xf1 = (ki1 < K) ? half_to_float(x[ki1]) : 0;
                b1 |= (uint32_t)f32_to_e2m1(xf1) << (ni * 4);
            }
            uint32_t sfa = packed_scales[sub], sfb = 0x38383838u;
            uint16_t bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
                  "f"(c0),"f"(c1),"f"(c2),"f"(c3),
                  "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b)
            : "memory");
            d0 += c0; d1 += c1; d2 += c2; d3 += c3;
        }
        if (tile_norm != 0 && tile_norm != 1.0f)
            { d0 *= tile_norm; d1 *= tile_norm; d2 *= tile_norm; d3 *= tile_norm; }
    }
    if (lane == 0) y[row] = d0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// WALSH-HADAMARD TRANSFORM — Register-domain via __shfl_xor_sync
// Ported from dengine den_wht_reg.h (Innovation #5)
// 6-stage 64-point WHT: 4 intra-thread + 2 cross-thread butterfly stages
// ═══════════════════════════════════════════════════════════════════════════════

// Intra-thread 16-point WHT (stages 1-4)
__device__ __forceinline__ void wht16_f32_intra(float r[16]) {
    for (int i = 0; i < 16; i += 2) {
        float a = r[i], b = r[i+1]; r[i] = a + b; r[i+1] = a - b;
    }
    for (int i = 0; i < 16; i += 4) {
        float a = r[i], b = r[i+2]; r[i] = a + b; r[i+2] = a - b;
        a = r[i+1]; b = r[i+3]; r[i+1] = a + b; r[i+3] = a - b;
    }
    for (int i = 0; i < 16; i += 8) {
        for (int j = 0; j < 4; j++) {
            float a = r[i+j], b = r[i+j+4]; r[i+j] = a + b; r[i+j+4] = a - b;
        }
    }
    for (int i = 0; i < 8; i++) {
        float a = r[i], b = r[i+8]; r[i] = a + b; r[i+8] = a - b;
    }
}

// 1/sqrt(N) normalization
__device__ __forceinline__ void wht_normalize(float r[], int n) {
    float s = rsqrtf((float)n);
    for (int i = 0; i < n; i++) r[i] *= s;
}

// 64-point WHT — 4 threads x 16 values each = 64 K-positions
// Thread layout: lane%4 covers K-threads; each thread has 16 contiguous values
__device__ __forceinline__ void wht64_f32(float r[16], int lane_id) {
    wht16_f32_intra(r);                          // Stages 1-4: intra-thread
    unsigned active = __activemask();
    // Stage 5: cross-thread XOR mask 1 (lane pairs 0-1, 2-3)
    for (int i = 0; i < 16; i++) {
        float p = __shfl_xor_sync(active, r[i], 1);
        r[i] = (lane_id & 1) ? (p - r[i]) : (r[i] + p);
    }
    // Stage 6: cross-thread XOR mask 2 (lane pairs 0-2, 1-3)
    for (int i = 0; i < 16; i++) {
        float p = __shfl_xor_sync(active, r[i], 2);
        r[i] = (lane_id & 2) ? (p - r[i]) : (r[i] + p);
    }
    wht_normalize(r, 64);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FUSED WH4 KERNEL — GGML block_nvfp4 → WHT(input) → OMMA.SF.16864
// Inline WHT eliminates the separate gpu_wht_f32_kernel launch.
// WH4 identity: <Hw, Hx> = 16·<w,x> — correction folded into weight scales.
// ═══════════════════════════════════════════════════════════════════════════════
extern "C" __global__ void nvfp4_gemv_fused_wh4_kernel(
    const uint8_t* __restrict__ blocks,  // [N*tpr][146] block_nvfp4
    const uint16_t* __restrict__ x,      // [K] input fp16
    float*         __restrict__ y,       // [N] output
    int N, int K, int tpr)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x, kgroup = lane / 8, within = lane % 4;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    constexpr int BLOCK_BYTES = 146;

    for (int t = 0; t < tpr; t++) {
        const uint8_t* blk = blocks + ((size_t)row * tpr + t) * BLOCK_BYTES;
        float tile_norm = __half2float(*(const __half*)blk);
        uint32_t packed_scales[4];
        for (int s = 0; s < 4; s++)
            packed_scales[s] = *(const uint32_t*)(blk + 2 + s * 4);

        for (int sub = 0; sub < 4; sub++) {
            int k_off = sub * 64;

            // A fragment: weight E2M1 nibbles (unchanged — pre-WHT'd during quant)
            volatile uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int ni = 0; ni < 8; ni++) {
                int byte_a0 = 18 + sub * 32 + (kgroup * 8 + ni) / 2;
                uint8_t w_a0 = (byte_a0 < 146) ? ((blk[byte_a0] >> (((kgroup * 8 + ni) & 1) * 4)) & 0xF) : 0;
                a0 |= (uint32_t)w_a0 << (ni * 4);
                int byte_a1 = 18 + sub * 32 + (kgroup * 8 + ni + 32) / 2;
                uint8_t w_a1 = (byte_a1 < 146) ? ((blk[byte_a1] >> (((kgroup * 8 + ni + 32) & 1) * 4)) & 0xF) : 0;
                a1 |= (uint32_t)w_a1 << (ni * 4);
                a2 |= (uint32_t)w_a0 << (ni * 4);
                a3 |= (uint32_t)w_a1 << (ni * 4);
            }

            // B fragment: INLINE WHT on input activations (Innovation #5)
            float r[16];
            int k_start = k_off + within * 16;
            for (int i = 0; i < 16; i++)
                r[i] = (k_start + i < K) ? half_to_float(x[k_start + i]) : 0.0f;
            wht64_f32(r, lane);  // register-domain WHT
            uint32_t b0 = 0, b1 = 0;
            for (int ni = 0; ni < 8; ni++) {
                b0 |= (uint32_t)f32_to_e2m1(r[ni]) << (ni * 4);
                b1 |= (uint32_t)f32_to_e2m1(r[ni + 8]) << (ni * 4);
            }

            // OMMA.SF.16864 — INLINE with memory clobber
            uint32_t sfa = packed_scales[sub], sfb = 0x38383838u;
            uint16_t bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;
            float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
                  "f"(c0),"f"(c1),"f"(c2),"f"(c3),
                  "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b)
                : "memory");
            d0 += c0; d1 += c1; d2 += c2; d3 += c3;
        }
        if (tile_norm != 0 && tile_norm != 1.0f)
            { d0 *= tile_norm; d1 *= tile_norm; d2 *= tile_norm; d3 *= tile_norm; }
    }
    if (lane == 0) y[row] = d0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEBUG PROBE: validate pointer + data layout (no OMMA)
extern "C" __global__ void nvfp4_probe_kernel(
    const uint8_t* __restrict__ blocks,
    const uint16_t* __restrict__ x,
    float*         __restrict__ y,
    int N, int K, int tpr)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x;
    constexpr int BLOCK_BYTES = 146;

    // Probe: read first block's fp16 norm and write it
    const uint8_t* blk = blocks + (size_t)row * tpr * BLOCK_BYTES;
    float norm = __half2float(*(const __half*)blk);

    // Probe: read first few scales and nibbles
    uint32_t scales_0 = *(const uint32_t*)(blk + 2);
    uint8_t nibble_0 = blk[18] & 0xF;

    // Probe: read input activation
    float x0 = (K > 0) ? half_to_float(x[0]) : 0.0f;

    // Write probe values (lane 0 writes all diagnostics)
    if (lane == 0) {
        y[row] = norm + (float)(scales_0 & 0xFF) * 0.0001f + nibble_0 * 0.000001f + x0 * 0.00000001f;
    } else if (lane < 4) {
        // Lanes 1-3: more probes
        float v = 0;
        if (lane == 1) v = half_to_float(x[lane]);
        if (lane == 2) v = (float)((scales_0 >> 8) & 0xFF);
        if (lane == 3) v = (float)((scales_0 >> 16) & 0xFF);
        __syncwarp();
    }
}

// DEBUG: OMMA with hardcoded all-ones (no memory reads)
extern "C" __global__ void nvfp4_omma_hardcoded_kernel(
    const uint8_t*, const uint16_t*, float* y, int N, int, int)
{
    int row = blockIdx.x;
    if (row >= N) return;
    if (threadIdx.x != 0) return;
    
    uint32_t a0 = 0x22222222u, a1 = 0x22222222u, a2 = 0x22222222u, a3 = 0x22222222u;
    uint32_t b0 = 0x22222222u, b1 = 0x22222222u;
    uint32_t sfa = 0x38383838u, sfb = 0x38383838u;
    uint16_t bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3),
          "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b)
            : "memory");
    y[row] = c0;
}

// DEBUG: iterate all tiles + all sub-tiles (no OMMA, but same access pattern)
extern "C" __global__ void nvfp4_probe_full_kernel(
    const uint8_t* __restrict__ blocks, const uint16_t* __restrict__ x,
    float* __restrict__ y, int N, int K, int tpr)
{
    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x;
    int kgroup = lane / 8, within = lane % 4;
    constexpr int BLOCK_BYTES = 146;
    float sum = 0;

    for (int t = 0; t < tpr; t++) {
        const uint8_t* blk = blocks + ((size_t)row * tpr + t) * BLOCK_BYTES;
        sum += __half2float(*(const __half*)blk) * 0.000001f;  // norm
        for (int sub = 0; sub < 4; sub++) {
            int k_off = sub * 64;
            // Access A-fragment nibbles (same as fused kernel)
            for (int ni = 0; ni < 8; ni++) {
                int ba = 18 + sub * 32 + (kgroup * 8 + ni) / 2;
                if (ba < 146) sum += (float)(blk[ba] & 0xF) * 1e-9f;
                int bb = 18 + sub * 32 + (kgroup * 8 + ni + 32) / 2;
                if (bb < 146) sum += (float)(blk[bb] & 0xF) * 1e-9f;
            }
            // Access B-fragment activations (same as fused kernel)
            int ks = within * 16;
            for (int ni = 0; ni < 8; ni++) {
                int ki0 = k_off + ks + ni;
                if (ki0 < K) sum += half_to_float(x[ki0]) * 1e-12f;
                int ki1 = k_off + ks + 8 + ni;
                if (ki1 < K) sum += half_to_float(x[ki1]) * 1e-12f;
            }
        }
    }
    // Single warp reduction
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    if (lane == 0) y[row] = sum;
}
// DEBUG: Hardcoded OMMA with full 32-thread warp (same config as fused)
extern "C" __global__ void nvfp4_omma_mt_kernel(float* y, int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x;
    int kgroup = lane / 8, within = lane % 4;
    
    // Read memory (simulate the fused kernel's access pattern)
    volatile float dummy = 0;
    for (int i = 0; i < 16; i++) dummy += (float)(lane * 0.0001f);
    
    // OMMA with per-lane registers (matching fused kernel layout)
    uint32_t a0 = 0x22222222u, a1 = 0x22222222u, a2 = 0x22222222u, a3 = 0x22222222u;
    uint32_t b0 = 0x22222222u, b1 = 0x22222222u;
    uint32_t sfa = 0x38383838u, sfb = 0x38383838u;
    uint16_t bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;
    float c0 = dummy, c1 = 0, c2 = 0, c3 = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3),
          "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b)
        : "memory");
    if (lane == 0) y[row] = c0;
}
// Forward declaration
__device__ __noinline__ void omma_step(uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,float&,float&,float&,float&);

// Noinline OMMA wrapper — prevents register conflicts with calling code
__device__ __noinline__ void omma_step(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb,
    float& d0, float& d1, float& d2, float& d3)
{
    uint16_t bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;
    float c0 = d0, c1 = d1, c2 = d2, c3 = d3;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
          "f"(c0),"f"(c1),"f"(c2),"f"(c3),
          "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b)
        : "memory");
    d0 += c0; d1 += c1; d2 += c2; d3 += c3;
}
// Fused kernel with SMEM staging to avoid register pressure
extern "C" __global__ void nvfp4_gemv_smem_fused_kernel(
    const uint8_t* __restrict__ blocks, const uint16_t* __restrict__ x,
    float* __restrict__ y, int N, int K, int tpr)
{
    __shared__ uint32_t smem_scales[4][4];    // 4 tiles * 4 uint32 scales
    __shared__ uint8_t  smem_nibs[4][128];    // 4 sub-tiles * 128B nibbles
    __shared__ float    smem_x[256];          // 256 fp32 activations per tile

    int row = blockIdx.x;
    if (row >= N) return;
    int lane = threadIdx.x;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    constexpr int BLOCK_BYTES = 146;

    for (int t = 0; t < tpr; t++) {
        const uint8_t* blk = blocks + ((size_t)row * tpr + t) * BLOCK_BYTES;
        float tile_norm = __half2float(*(const __half*)blk);

        // Cooperative load: 32 threads load nibbles into SMEM
        for (int i = lane; i < 128; i += 32) smem_nibs[0][i] = blk[18 + i];
        for (int i = lane; i < 4; i++) smem_scales[0][i] = *(const uint32_t*)(blk + 2 + i*4);
        // Load activations for this tile
        int base_k = t * 256;
        for (int i = lane; i < 256; i += 32)
            smem_x[i] = (base_k + i < K) ? half_to_float(x[base_k + i]) : 0.0f;
        __syncthreads();

        // Process 4 sub-tiles from SMEM (no memory reads, no register pressure)
        for (int sub = 0; sub < 4; sub++) {
            int kgroup = lane / 8, within = lane % 4;
            uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int ni = 0; ni < 8; ni++) {
                int byte_idx = sub * 32 + (kgroup * 8 + ni) / 2;
                uint8_t w = (byte_idx < 128) ? ((smem_nibs[0][byte_idx] >> (((kgroup * 8 + ni) & 1) * 4)) & 0xF) : 0;
                a0 |= (uint32_t)w << (ni * 4);
                int byte_idx2 = sub * 32 + (kgroup * 8 + ni + 32) / 2;
                uint8_t w2 = (byte_idx2 < 128) ? ((smem_nibs[0][byte_idx2] >> (((kgroup * 8 + ni + 32) & 1) * 4)) & 0xF) : 0;
                a1 |= (uint32_t)w2 << (ni * 4);
                a2 |= (uint32_t)w << (ni * 4);
                a3 |= (uint32_t)w2 << (ni * 4);
            }
            uint32_t b0 = 0, b1 = 0;
            int ks = within * 16;
            for (int ni = 0; ni < 8; ni++) {
                float xf0 = smem_x[sub * 64 + ks + ni];
                b0 |= (uint32_t)f32_to_e2m1(xf0) << (ni * 4);
                float xf1 = smem_x[sub * 64 + ks + 8 + ni];
                b1 |= (uint32_t)f32_to_e2m1(xf1) << (ni * 4);
            }
            uint32_t sfa = smem_scales[0][sub], sfb = 0x38383838u;
            uint16_t bid_a=0,tid_a=0,bid_b=0,tid_b=0;
            float c0=0,c1=0,c2=0,c3=0;
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14,{%15,%16},%17,{%18,%19};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
                  "f"(c0),"f"(c1),"f"(c2),"f"(c3),
                  "r"(sfa),"h"(bid_a),"h"(tid_a),"r"(sfb),"h"(bid_b),"h"(tid_b));
            d0 += c0; d1 += c1; d2 += c2; d3 += c3;
        }
        if (tile_norm != 0 && tile_norm != 1.0f) { d0 *= tile_norm; d1 *= tile_norm; d2 *= tile_norm; d3 *= tile_norm; }
        __syncthreads();
    }
    if (lane == 0) y[row] = d0;
}
