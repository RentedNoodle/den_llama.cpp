//
// AVX-512-accelerated MoE expert FFN computation
//
// Copyright (C) 2026 Project Den
// MIT license
// SPDX-License-Identifier: MIT
//
// Replaces the generic ggml matmul path for CPU-side expert weight
// multiplication when --n-cpu-moe places expert tensors on CPU.
// For a 35B MoE model with 7168-embd/20480-ff, each expert matmul
// does 7168*20480 = ~147M FMAs.  With AVX-512, this drops from
// ~70 ms to ~9 ms per token.
//
// Supported quantization types:
//   GGML_TYPE_F16    — direct FP16->FP32 FMA (block size 1)
//   GGML_TYPE_Q8_0   — int8 x scale (block size 32)
//   GGML_TYPE_Q4_K   — 4-bit with scale+min (block size 256)
//   GGML_TYPE_Q4_K_R4 — same as Q4_K at runtime
//   GGML_TYPE_IQ2_S  — 2.5625 bpw grid+sign+scale (block size 256)
//

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

#include "ggml-impl.h"   // GGML_FP16_TO_FP32, type traits
#define GGML_COMMON_DECL_C
#include "ggml-common.h" // block_q8_0, block_q4_K, block_iq2_s (types)
#undef GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h" // iq2s_grid, iq2s_avg_norm (data tables)
#undef GGML_COMMON_IMPL_C

// =====================================================================
// Utility: horizontal sum of a 512-bit vector of 16 floats
// =====================================================================
#if defined(__AVX512F__)
static inline float hsum_512(__m512 v) {
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 sum = _mm256_add_ps(lo, hi);
    __m128 hi4 = _mm256_extractf128_ps(sum, 1);
    __m128 lo4 = _mm256_castps256_ps128(sum);
    __m128 sum4 = _mm_add_ps(lo4, hi4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

#endif

// =====================================================================
// FP16 expert matmul — direct FP16->FP32 conversion + FMA
// W: [n_ff, n_embd] in FP16 (ggml_half)
// =====================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__)
// GCC 15+ changed _mm512_cvtph_ps arg from __m512i to __m256i.
// Use __GNUC__ guard until the AVX-512 expert path is updated for new API.
#if defined(__AVX512F__) && defined(__GNUC__) && __GNUC__ < 15
static void expert_mul_mat_f16(
    const float * restrict x,
    const void * restrict w,
    float * restrict y,
    int n_embd,
    int n_ff)
{
    const ggml_half * w16 = (const ggml_half *)w;

    // Process 4 output rows at a time to amortize x-load cost over rows
    for (int i = 0; i < n_ff; i += 4) {
        int n_rows = (i + 4 <= n_ff) ? 4 : (n_ff - i);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (int j = 0; j < n_embd; j += 16) {
            __m512 xv = _mm512_loadu_ps(x + j);

            // Row 0
            __m512i w0 = _mm512_loadu_si512(w16 + (i+0)*n_embd + j);
            acc0 = _mm512_fmadd_ps(_mm512_cvtph_ps(w0), xv, acc0);

            if (n_rows > 1) {
                __m512i w1 = _mm512_loadu_si512(w16 + (i+1)*n_embd + j);
                acc1 = _mm512_fmadd_ps(_mm512_cvtph_ps(w1), xv, acc1);
            }
            if (n_rows > 2) {
                __m512i w2 = _mm512_loadu_si512(w16 + (i+2)*n_embd + j);
                acc2 = _mm512_fmadd_ps(_mm512_cvtph_ps(w2), xv, acc2);
            }
            if (n_rows > 3) {
                __m512i w3 = _mm512_loadu_si512(w16 + (i+3)*n_embd + j);
                acc3 = _mm512_fmadd_ps(_mm512_cvtph_ps(w3), xv, acc3);
            }
        }

        y[i+0] = hsum_512(acc0);
        if (n_rows > 1) y[i+1] = hsum_512(acc1);
        if (n_rows > 2) y[i+2] = hsum_512(acc2);
        if (n_rows > 3) y[i+3] = hsum_512(acc3);
    }
}
#endif
#endif // __AVX512F__ && __GNUC__ < 15

// =====================================================================
// Q8_0 expert matmul
// Block: { ggml_half d; int8_t qs[32]; } — 32 weights per block
// Compute: for each block, y += d * sum(qs[j] * x[j])
// =====================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__)
static void expert_mul_mat_q8_0(
    const float * restrict x,
    const void * restrict w,
    float * restrict y,
    int n_embd,
    int n_ff)
{
    const int blk_per_row = n_embd / 32;
    const size_t row_stride = blk_per_row * sizeof(block_q8_0);

    for (int i = 0; i < n_ff; i += 4) {
        int n_rows = (i + 4 <= n_ff) ? 4 : (n_ff - i);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        const block_q8_0 * r0 = (const block_q8_0 *)((const char *)w + (i+0)*row_stride);
        const block_q8_0 * r1 = (const block_q8_0 *)((const char *)w + (i+1)*row_stride);
        const block_q8_0 * r2 = (const block_q8_0 *)((const char *)w + (i+2)*row_stride);
        const block_q8_0 * r3 = (const block_q8_0 *)((const char *)w + (i+3)*row_stride);

        for (int b = 0; b < blk_per_row; ++b) {
            // Load x[32*b .. 32*b+31]
            __m512 x_lo = _mm512_loadu_ps(x + 32*b + 0);
            __m512 x_hi = _mm512_loadu_ps(x + 32*b + 16);

            // Dequantize 32 int8 -> 32 float and FMA per row
            #define PROC_Q8_ROW(rp, acc) do { \
                float d = GGML_FP16_TO_FP32(rp[b].d); \
                __m256i i8 = _mm256_loadu_si256((const __m256i *)rp[b].qs); \
                __m128i i8_lo = _mm256_castsi256_si128(i8); \
                __m128i i8_hi = _mm256_extracti128_si256(i8, 1); \
                __m512 f_lo = _mm512_mul_ps( \
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(i8_lo)), \
                    _mm512_set1_ps(d)); \
                __m512 f_hi = _mm512_mul_ps( \
                    _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(i8_hi)), \
                    _mm512_set1_ps(d)); \
                acc = _mm512_fmadd_ps(f_lo, x_lo, acc); \
                acc = _mm512_fmadd_ps(f_hi, x_hi, acc); \
            } while (0)

            PROC_Q8_ROW(r0, acc0);
            if (n_rows > 1) PROC_Q8_ROW(r1, acc1);
            if (n_rows > 2) PROC_Q8_ROW(r2, acc2);
            if (n_rows > 3) PROC_Q8_ROW(r3, acc3);
        }

        y[i+0] = hsum_512(acc0);
        if (n_rows > 1) y[i+1] = hsum_512(acc1);
        if (n_rows > 2) y[i+2] = hsum_512(acc2);
        if (n_rows > 3) y[i+3] = hsum_512(acc3);
    }
}
#endif

// =====================================================================
// Scale/min decoder for Q4_K block (get_scale_min_k4 replacement)
// Decodes the packed 12-byte scales[] into 8 scale and 8 min values,
// each 0..63 (6-bit).
//
// Encoding (from ggml-quants.c):
//   For j = 0..3:  sc = scales[j] & 63,  min = scales[j+4] & 63
//   For j = 4..7:  sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//                   min = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
// =====================================================================
static inline void decode_q4k_scales(const uint8_t scales[12],
                                     uint8_t sc[8], uint8_t mn[8]) {
    for (int j = 0; j < 4; ++j) {
        sc[j] = scales[j] & 63;
        mn[j] = scales[j + 4] & 63;
    }
    for (int j = 4; j < 8; ++j) {
        sc[j] = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        mn[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

// =====================================================================
// Q4_K / Q4_K_M expert matmul
// Block: { ggml_half d, ggml_half dmin, uint8_t scales[12], qs[128] }
// 256 weights per block, 8 sub-blocks of 32 weights each.
//
// Sub-block value: d * sc * quant - min * m
//   where sc,m are 6-bit decoded from scales[]
//   quant is 4-bit (0..15), packed as nibbles.
//
// Nibble layout in qs[128]:
//   qs[iter*32 .. iter*32+31] encodes:
//     low  nibble = sub-block A (columns iter*64 .. iter*64+31)
//     high nibble = sub-block B (columns iter*64+32 .. iter*64+63)
//   for iter = 0..3.
// =====================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__)
static void expert_mul_mat_q4_k(
    const float * restrict x,
    const void * restrict w,
    float * restrict y,
    int n_embd,
    int n_ff)
{
    const int blk_per_row = n_embd / 256;
    const size_t row_stride = blk_per_row * sizeof(block_q4_K);

    for (int i = 0; i < n_ff; i += 4) {
        int n_rows = (i + 4 <= n_ff) ? 4 : (n_ff - i);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        const block_q4_K * r0 = (const block_q4_K *)((const char *)w + (i+0)*row_stride);
        const block_q4_K * r1 = (const block_q4_K *)((const char *)w + (i+1)*row_stride);
        const block_q4_K * r2 = (const block_q4_K *)((const char *)w + (i+2)*row_stride);
        const block_q4_K * r3 = (const block_q4_K *)((const char *)w + (i+3)*row_stride);

        for (int b = 0; b < blk_per_row; ++b) {

            #define PROC_Q4K_ROW(rp, acc) do { \
                const float d   = GGML_FP16_TO_FP32(rp[b].d); \
                const float min = GGML_FP16_TO_FP32(rp[b].dmin); \
                const uint8_t * qs = rp[b].qs; \
                uint8_t sc[8], mn[8]; \
                decode_q4k_scales(rp[b].scales, sc, mn); \
                /* 4 iterations x 2 halves = 8 sub-blocks of 32 */ \
                for (int iter = 0; iter < 4; ++iter) { \
                    /* Even sub-block: low nibbles, columns [iter*64 .. iter*64+31] */ \
                    float dA = d * (float)sc[iter*2 + 0]; \
                    float mA = -min * (float)mn[iter*2 + 0]; \
                    __m512 sdA = _mm512_set1_ps(dA); \
                    __m512 smA = _mm512_set1_ps(mA); \
                    /* Odd sub-block: high nibbles, columns [iter*64+32 .. iter*64+63] */ \
                    float dB = d * (float)sc[iter*2 + 1]; \
                    float mB = -min * (float)mn[iter*2 + 1]; \
                    __m512 sdB = _mm512_set1_ps(dB); \
                    __m512 smB = _mm512_set1_ps(mB); \
                    /* Each half processes 16 qs bytes -> 16 low + 16 high nibbles */ \
                    for (int half = 0; half < 2; ++half) { \
                        int base = 256*b + iter*64 + half*16; \
                        __m128i q4 = _mm_loadu_si128((const __m128i *)(qs + iter*32 + half*16)); \
                        /* Low nibbles: columns [base .. base+15] sub-block A */ \
                        __m128i q4_lo = _mm_and_si128(q4, _mm_set1_epi8(0x0F)); \
                        __m512i q32_lo = _mm512_cvtepu8_epi32(q4_lo); \
                        __m512 fA = _mm512_fmadd_ps(sdA, _mm512_cvtepi32_ps(q32_lo), smA); \
                        acc = _mm512_fmadd_ps(fA, _mm512_loadu_ps(x + base), acc); \
                        /* High nibbles: columns [base+32 .. base+47] sub-block B */ \
                        __m128i q4_hi = _mm_and_si128( \
                            _mm_srli_epi16(q4, 4), _mm_set1_epi8(0x0F)); \
                        __m512i q32_hi = _mm512_cvtepu8_epi32(q4_hi); \
                        __m512 fB = _mm512_fmadd_ps(sdB, _mm512_cvtepi32_ps(q32_hi), smB); \
                        acc = _mm512_fmadd_ps(fB, _mm512_loadu_ps(x + base + 32), acc); \
                    } \
                } \
            } while (0)

            PROC_Q4K_ROW(r0, acc0);
            if (n_rows > 1) PROC_Q4K_ROW(r1, acc1);
            if (n_rows > 2) PROC_Q4K_ROW(r2, acc2);
            if (n_rows > 3) PROC_Q4K_ROW(r3, acc3);
        }

        y[i+0] = hsum_512(acc0);
        if (n_rows > 1) y[i+1] = hsum_512(acc1);
        if (n_rows > 2) y[i+2] = hsum_512(acc2);
        if (n_rows > 3) y[i+3] = hsum_512(acc3);
    }
}
#endif

// =====================================================================
// IQ2_S expert matmul
// Block: { ggml_half d, uint8_t qs[64], uint8_t qh[8], uint8_t scales[8] }
// 256 weights per block, 2.5625 bpw.
//
// For each 256-weight block:
//   8 sub-blocks (ib32 = 0..7) of 32 elements each
//   Each sub-block has:
//     db[0] = d * (0.5 + (scales[ib32] & 0xf)) * 0.25  (for l=0,1)
//     db[1] = d * (0.5 + (scales[ib32] >>  4)) * 0.25  (for l=2,3)
//     For l=0..3: 8 values from iq2s_grid[10-bit index], sign-flipped
//
// The 10-bit grid index combines qs[l] (lower 8 bits) with
// 2 bits from qh[ib32] extracted at position (8-2*l).
//
// kmask_iq2xs[8] = { 1,2,4,8,16,32,64,128 } — sign bit mask per position.
//
// We dequantize 32 values into a stack buffer, then AVX-512 FMA with x.
// =====================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__)
static void expert_mul_mat_iq2_s(
    const float * restrict x,
    const void * restrict w,
    float * restrict y,
    int n_embd,
    int n_ff)
{
    // kmask_iq2xs and iq2s_grid are static const in ggml-common.h
    // (included via GGML_TABLE_BEGIN macro with GGML_COMMON_IMPL_C not defined here,
    //  so they are local static const in this translation unit)

    const int blk_per_row = n_embd / 256;
    const size_t row_stride = blk_per_row * sizeof(block_iq2_s);

    // Stack buffer for dequantized 32-float chunks (2 rows x 32)
    float dq[2][32];

    for (int i = 0; i < n_ff; i += 2) {
        int n_rows = (i + 2 <= n_ff) ? 2 : (n_ff - i);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();

        const block_iq2_s * r0 = (const block_iq2_s *)((const char *)w + (i+0)*row_stride);
        const block_iq2_s * r1 = (const block_iq2_s *)((const char *)w + (i+1)*row_stride);

        for (int b = 0; b < blk_per_row; ++b) {

            #define PROC_IQ2S_ROW(rp, acc, dbuf) do { \
                const float d = GGML_FP16_TO_FP32(rp[b].d); \
                const uint8_t * qs    = rp[b].qs; \
                const uint8_t * qh    = rp[b].qh; \
                const uint8_t * signs = qs + 32; /* upper half of qs area */ \
                int col_off = 256 * b; \
                for (int ib32 = 0; ib32 < 8; ++ib32) { \
                    float db0 = d * (0.5f + (rp[b].scales[ib32] & 0x0f)) * 0.25f; \
                    float db1 = d * (0.5f + (rp[b].scales[ib32] >>   4)) * 0.25f; \
                    float * dptr = dbuf; \
                    uint8_t qh_byte = qh[ib32]; \
                    for (int l = 0; l < 4; ++l) { \
                        float dl = (l < 2) ? db0 : db1; \
                        uint32_t gidx = qs[l] | ((qh_byte << (8 - 2*l)) & 0x300); \
                        uint64_t gp = iq2s_grid[gidx]; \
                        int8_t gv[8]; \
                        memcpy(gv, &gp, 8); \
                        uint8_t sb = signs[l]; \
                        dptr[0] = dl * (float)gv[0] * ((sb & 0x01) ? -1.f : 1.f); \
                        dptr[1] = dl * (float)gv[1] * ((sb & 0x02) ? -1.f : 1.f); \
                        dptr[2] = dl * (float)gv[2] * ((sb & 0x04) ? -1.f : 1.f); \
                        dptr[3] = dl * (float)gv[3] * ((sb & 0x08) ? -1.f : 1.f); \
                        dptr[4] = dl * (float)gv[4] * ((sb & 0x10) ? -1.f : 1.f); \
                        dptr[5] = dl * (float)gv[5] * ((sb & 0x20) ? -1.f : 1.f); \
                        dptr[6] = dl * (float)gv[6] * ((sb & 0x40) ? -1.f : 1.f); \
                        dptr[7] = dl * (float)gv[7] * ((sb & 0x80) ? -1.f : 1.f); \
                        dptr += 8; \
                    } \
                    qs    += 4; \
                    signs += 4; \
                    __m512 x_lo = _mm512_loadu_ps(x + col_off + 0); \
                    __m512 x_hi = _mm512_loadu_ps(x + col_off + 16); \
                    __m512 d_lo = _mm512_loadu_ps(dbuf + 0); \
                    __m512 d_hi = _mm512_loadu_ps(dbuf + 16); \
                    acc = _mm512_fmadd_ps(d_lo, x_lo, acc); \
                    acc = _mm512_fmadd_ps(d_hi, x_hi, acc); \
                    col_off += 32; \
                } \
            } while (0)

            PROC_IQ2S_ROW(r0, acc0, dq[0]);
            if (n_rows > 1) {
                PROC_IQ2S_ROW(r1, acc1, dq[1]);
            }
        }

        y[i+0] = hsum_512(acc0);
        if (n_rows > 1) y[i+1] = hsum_512(acc1);
    }
}
#endif

// =====================================================================
// CPU Pre-Dequant: IQ2_S → FP16 using AVX-512 (Feature 4C)
//
// Dequantizes IQ2_S expert weights directly to FP16 on CPU.
// Processes 32 values per sub-block iteration with 512-bit SIMD.
// Output is packed ggml_half (uint16_t) ready for DMA to GPU.
//
// Each block_iq2_s (82 bytes) produces 256 FP16 values (512 bytes).
// The GPU receives pre-dequantized FP16 — no GPU-side dequant needed.
// =====================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__)
GGML_API int ggml_avx512_dequant_iq2s_to_fp16(
    const void * iq2s_data,
    void * fp16_out,
    int n_values)
{
    const block_iq2_s * blk = (const block_iq2_s *)iq2s_data;
    uint16_t * GGML_RESTRICT out = (uint16_t *)fp16_out;
    int blk_count = n_values / 256;

    // Stack buffers for dequantized float chunks (2 sub-blocks x 32)
    float dq_buf[64];

    for (int b = 0; b < blk_count; b++) {
        const float d = GGML_FP16_TO_FP32(blk[b].d);
        const uint8_t * qs    = blk[b].qs;
        const uint8_t * qh    = blk[b].qh;
        const uint8_t * signs = blk[b].qs + 32; // upper half of qs area

        int out_off = b * 256;

        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db0 = d * (0.5f + (blk[b].scales[ib32] & 0x0f)) * 0.25f;
            float db1 = d * (0.5f + (blk[b].scales[ib32] >> 4)) * 0.25f;
            float * dptr = dq_buf;
            uint8_t qh_byte = qh[ib32];

            for (int l = 0; l < 4; l++) {
                float dl = (l < 2) ? db0 : db1;
                uint32_t gidx = qs[l] | ((qh_byte << (8 - 2*l)) & 0x300);
                uint64_t gp = iq2s_grid[gidx];
                int8_t gv[8];
                memcpy(gv, &gp, 8);
                uint8_t sb = signs[l];

                dptr[0] = dl * (float)gv[0] * ((sb & 0x01) ? -1.f : 1.f);
                dptr[1] = dl * (float)gv[1] * ((sb & 0x02) ? -1.f : 1.f);
                dptr[2] = dl * (float)gv[2] * ((sb & 0x04) ? -1.f : 1.f);
                dptr[3] = dl * (float)gv[3] * ((sb & 0x08) ? -1.f : 1.f);
                dptr[4] = dl * (float)gv[4] * ((sb & 0x10) ? -1.f : 1.f);
                dptr[5] = dl * (float)gv[5] * ((sb & 0x20) ? -1.f : 1.f);
                dptr[6] = dl * (float)gv[6] * ((sb & 0x40) ? -1.f : 1.f);
                dptr[7] = dl * (float)gv[7] * ((sb & 0x80) ? -1.f : 1.f);
                dptr += 8;
            }

            qs    += 4;
            signs += 4;

            // Convert 32 floats → 32 FP16 via AVX-512
            __m512 d_lo = _mm512_loadu_ps(dq_buf + 0);
            __m512 d_hi = _mm512_loadu_ps(dq_buf + 16);
            __m256i h_lo = _mm512_cvtps_ph(d_lo, _MM_FROUND_TO_NEAREST_INT);
            __m256i h_hi = _mm512_cvtps_ph(d_hi, _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256((__m256i*)(out + out_off + 0),  h_lo);
            _mm256_storeu_si256((__m256i*)(out + out_off + 16), h_hi);
            out_off += 32;
        }
    }

    return 0;
}
#endif

// =====================================================================
// CPU Pre-Dequant: NVFP4 → FP16 using AVX-512 (Feature 4C)
//
// Dequantizes NVFP4 expert weights directly to FP16 on CPU.
// Each block_nvfp4 (160 bytes) produces 256 FP16 values (512 bytes).
// =====================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__)

// E2M1 lookup table (replicated from ggml-nvfp4-quants.c)
static const float nvfp4_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
};

// UE4M3 decode (replicated from ggml-nvfp4-quants.c)
static inline float nvfp4_ue4m3_to_f32(uint8_t ue4) {
    if (ue4 < 8) {
        const float table[8] = {0.0f, 0.0625f, 0.125f, 0.1875f,
                                0.25f, 0.3125f, 0.375f, 0.4375f};
        return table[ue4];
    }
    const float table[8] = {1.0f, 1.125f, 1.25f, 1.375f,
                            1.5f, 1.625f, 1.75f, 1.875f};
    return table[ue4 - 8];
}

GGML_API int ggml_avx512_dequant_nvfp4_to_fp16(
    const void * nvfp4_data,
    void * fp16_out,
    int n_values)
{
    const block_nvfp4 * blk = (const block_nvfp4 *)nvfp4_data;
    uint16_t * GGML_RESTRICT out = (uint16_t *)fp16_out;
    int blk_count = n_values / QK_NVFP4;

    for (int i = 0; i < blk_count; i++) {
        const float tile_norm = blk[i].tile_norm;
        const int block_off = i * QK_NVFP4;

        // Process 16 values per sub-block (16 sub-blocks = 256 values)
        for (int b = 0; b < 16; b++) {
            const float scale = nvfp4_ue4m3_to_f32(blk[i].block_scales[b]) * tile_norm;
            const int sub_off = block_off + b * 16;

            // Dequant 16 E2M1 nibbles → FP32, then convert to FP16
            float f32_buf[16];
            for (int j = 0; j < 16; j++) {
                const int idx = b * 16 + j;
                const int byte_idx = idx / 2;
                uint8_t nibble;
                if (idx & 1) {
                    nibble = (blk[i].e2m1_nibbles[byte_idx] >> 4) & 0xF;
                } else {
                    nibble = blk[i].e2m1_nibbles[byte_idx] & 0xF;
                }
                f32_buf[j] = nvfp4_e2m1_lut[nibble] * scale;
            }

            // Convert 16 FP32 → 16 FP16 via AVX-512
            __m512 f32_vec = _mm512_loadu_ps(f32_buf);
            __m256i f16_vec = _mm512_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256((__m256i*)(out + sub_off), f16_vec);
        }
    }

    return 0;
}
#endif

// =====================================================================
// Generic dequant-to-FP16 dispatcher (Feature 4B/4C)
//
// Selects the right AVX-512 dequant kernel based on ggml_type.
// Returns 0 on success, -1 if type is not supported.
// =====================================================================
GGML_API int ggml_avx512_dequant_to_fp16(
    const void * src,
    void * fp16_out,
    int n_values,
    int type)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    switch (type) {
        case GGML_TYPE_IQ2_S:
            return ggml_avx512_dequant_iq2s_to_fp16(src, fp16_out, n_values);
        case GGML_TYPE_NVFP4:
            return ggml_avx512_dequant_nvfp4_to_fp16(src, fp16_out, n_values);
        case GGML_TYPE_F16: {
            // Already FP16 — direct copy
            memcpy(fp16_out, src, (size_t)n_values * sizeof(uint16_t));
            return 0;
        }
        case GGML_TYPE_F32: {
            // FP32→FP16 conversion via AVX-512
            const float * f32_src = (const float *)src;
            uint16_t * f16_dst = (uint16_t *)fp16_out;
            for (int i = 0; i < n_values; i += 16) {
                __m512 f32_vec = _mm512_loadu_ps(f32_src + i);
                __m256i f16_vec = _mm512_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
                _mm256_storeu_si256((__m256i*)(f16_dst + i), f16_vec);
            }
            return 0;
        }
        default:
            return -1;
    }
#else
    (void)src; (void)fp16_out; (void)n_values; (void)type;
    return -1;
#endif
}

// =====================================================================
// Public API dispatcher
//
// Computes y = W * x^T  (matrix-vector product) where:
//   x: FP32 input vector [n_embd]
//   w: quantized weight matrix [n_ff, n_embd] in GGML format
//   y: FP32 output vector [n_ff]
//
// type: ggml_type enum for the quant format of w
//
// Returns 0 on success, -1 if type is not supported (caller should
// fall back to generic ggml matmul).
// =====================================================================
GGML_API int ggml_avx512_expert_mul_mat(
    const float * restrict x,
    const void * restrict w,
    float * restrict y,
    int n_embd,
    int n_ff,
    int type)
{
#if defined(__AVX512F__) && defined(__AVX512BW__)
    switch (type) {
#if defined(__GNUC__) && __GNUC__ < 15
        case GGML_TYPE_F16:
            expert_mul_mat_f16(x, w, y, n_embd, n_ff);
            return 0;
#endif
        case GGML_TYPE_Q8_0:
            expert_mul_mat_q8_0(x, w, y, n_embd, n_ff);
            return 0;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q4_K_R4:
            expert_mul_mat_q4_k(x, w, y, n_embd, n_ff);
            return 0;
        case GGML_TYPE_IQ2_S:
            expert_mul_mat_iq2_s(x, w, y, n_embd, n_ff);
            return 0;
        default:
            return -1;
    }
#else
    (void)x; (void)w; (void)y;
    (void)n_embd; (void)n_ff; (void)type;
    return -1;
#endif
}

// =====================================================================
// Runtime query: is AVX-512 available?
// Returns 1 if available (compiled-in for native builds), 0 otherwise.
// =====================================================================
GGML_API int ggml_avx512_expert_available(void) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return 1;
#else
    return 0;
#endif
}
