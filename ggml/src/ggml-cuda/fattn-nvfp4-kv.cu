// fattn-nvfp4-kv.cu — NVFP4 KV Cache kernels for den_llama.cpp
//
// Ported from Project Den dengine/src/den_kv_cache.cu
// See fattn-nvfp4-kv.cuh for the public API and data structures.

#include "fattn-nvfp4-kv.cuh"
#include "common.cuh"
#include "ggml-cuda.h"

// Set by ggml-cuda.cu while CUDA graph capture is active (BeginCapture..EndCapture).
// During capture this store must launch on the capturing (main) stream so its work is
// JOINED to the capture; a cross-stream event-wait would fail capture with
// "capturing stream has unjoined work".
extern bool g_den_cuda_graph_capturing;

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_InterlockedCompareExchange)
#endif

// ═══════════════════════════════════════════════════════════
// Shared-memory ceiling (sm_120a: 99 KB)
// ═══════════════════════════════════════════════════════════

#ifndef DEN_SMEM_MAX_BYTES
#define DEN_SMEM_MAX_BYTES 101376
#endif

// ═══════════════════════════════════════════════════════════
// Host-side atomic max
// ═══════════════════════════════════════════════════════════

static inline void kv_atomic_max_i32(volatile int * ptr, int new_val) {
#ifdef _MSC_VER
    int old_val;
    do {
        old_val = *ptr;
        if (new_val <= old_val) return;
    } while (_InterlockedCompareExchange(
        (long volatile *)ptr, (long)new_val, (long)old_val) != (long)old_val);
#else
    int old_val;
    do {
        old_val = *ptr;
        if (new_val <= old_val) return;
    } while (!__sync_bool_compare_and_swap(ptr, old_val, new_val));
#endif
}

// ═══════════════════════════════════════════════════════════
// LUTs: E2M1 and UE4M3 decode tables
// ═══════════════════════════════════════════════════════════

__device__ __constant__ const float kv_e2m1_lut[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

__device__ __constant__ const float kv_ue4m3_lut[16] = {
    // UE4M3-unsigned scale table.
    // DATA-DRIVEN (measured on ornith-1.0-35b-APEX-I-Mini-MTP, 104K K blocks +
    // 104K V blocks): ideal_scale = max_abs/6. Median K=0.53, V=0.16; p99
    // K=1.47, V=0.84; real max ~2.6. Saturation at 1.875 is negligible (0.09% K,
    // 0.05% V). So density goes in 0.06-1.5 (covers ~99.9% of blocks); cap 1.5.
    // NO codes above 1.5 — a too-coarse scale underflows qval and kills
    // attention (a 4.0/16.0 code collapsed replay to 5.3%). The scale table is a
    // SAFETY lever (per-block error is ~identical across LUTs); the precision
    // win comes from the 8-bit element path (K8V8).
    0.0f,  0.0625f, 0.125f, 0.1875f,
    0.25f, 0.3125f, 0.375f, 0.4375f,
    0.5f,  0.5625f, 0.625f, 0.75f,
    0.875f, 1.0f,   1.25f,  1.5f
};

// ═══════════════════════════════════════════════════════════
// Scale-distribution probe instrumentation
// Histograms ideal_scale = max_abs/6 per 16-elem K/V block.
// Gated by DEN_NVFP4_KV_HIST=1 (host sets g_kv_hist_enabled).
// Hist pointer == NULL => no recording (zero overhead when off).
// ═══════════════════════════════════════════════════════════

#define DEN_KV_HIST_BINS 256
#define DEN_KV_HIST_W     0.0625f

__device__ unsigned int g_kv_hist_k[DEN_KV_HIST_BINS];
__device__ unsigned int g_kv_hist_v[DEN_KV_HIST_BINS];
__device__ unsigned long long g_kv_cnt_k;
__device__ unsigned long long g_kv_cnt_v;
__device__ unsigned int g_kv_hist_enabled;

__device__ __forceinline__ void kv_hist_record(
    const float * __restrict__ vec, int head_dim, unsigned int * hist)
{
    if (hist == NULL) return;
    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;
    for (int g = 0; g < n_groups; g++) {
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;
        float max_abs = 0.0f;
        for (int e = blk_start; e < blk_end; e++) {
            float av = vec[e];
            if (av < 0.0f) av = -av;
            if (av > max_abs) max_abs = av;
        }
        if (max_abs < 1e-10f) continue; // zero block — no scale is chosen
        float ideal = max_abs / 6.0f;
        int idx = (int)(ideal / DEN_KV_HIST_W);
        if (idx < 0) idx = 0;
        if (idx >= DEN_KV_HIST_BINS) idx = DEN_KV_HIST_BINS - 1;
        atomicAdd(&hist[idx], 1u);
    }
}

// ═══════════════════════════════════════════════════════════
// Device: quantize float vector → NVFP4 tile
// O(1) per-block: max_abs → nearest UE4M3 scale code
// ═══════════════════════════════════════════════════════════

__device__ static void kv_quantize_tile(
    const float * __restrict__ vec,
    uint8_t * tile,
    int head_dim,
    unsigned int * hist = NULL)
{
    kv_hist_record(vec, head_dim, hist);

    // Zero tile
    for (int i = 0; i < DEN_NVFP4_KV_TILE_BYTES; i += 16) {
        if (i + 16 <= DEN_NVFP4_KV_TILE_BYTES) {
            *(uint4 *)(tile + i) = make_uint4(0, 0, 0, 0);
        }
    }

    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;

    // RMS for tile norm
    double sum_sq = 0.0;
    for (int i = 0; i < head_dim; i++)
        sum_sq += (double)vec[i] * (double)vec[i];

    for (int g = 0; g < n_groups; g++) {
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;
        int n_in_blk = blk_end - blk_start;

        float max_abs = 0.0f;
        for (int e = 0; e < n_in_blk; e++) {
            float av = vec[blk_start + e];
            if (av < 0.0f) av = -av;
            if (av > max_abs) max_abs = av;
        }

        uint8_t scale_code = 0;
        if (max_abs >= 1e-10f) {
            float ideal_scale = max_abs / 6.0f;
            float best_err = fabsf(ideal_scale - kv_ue4m3_lut[1]);
            uint8_t best_code = 1;
            #pragma unroll
            for (int c = 2; c < 16; c++) {
                float err = fabsf(ideal_scale - kv_ue4m3_lut[c]);
                if (err < best_err) { best_err = err; best_code = (uint8_t)c; }
            }
            scale_code = best_code;
        }

        tile[g] = scale_code;
        float scale = kv_ue4m3_lut[scale_code];

        for (int e = 0; e < n_in_blk; e++) {
            float val = vec[blk_start + e];
            float qval = (scale > 1e-10f) ? val / scale : 0.0f;

            if (qval > 6.0f)  qval =  6.0f;
            if (qval < -6.0f) qval = -6.0f;

            uint8_t sgn = (qval < 0.0f) ? 0x08 : 0x00;
            float abs_q = (qval < 0.0f) ? -qval : qval;

            uint8_t mag;
            if      (abs_q >= 5.0f)  mag = 7;
            else if (abs_q >= 3.5f)  mag = 6;
            else if (abs_q >= 2.5f)  mag = 5;
            else if (abs_q >= 1.75f) mag = 4;
            else if (abs_q >= 1.25f) mag = 3;
            else if (abs_q >= 0.75f) mag = 2;
            else if (abs_q >= 0.25f) mag = 1;
            else                      mag = 0;

            uint8_t nibble = sgn | mag;

            int byte_idx = DEN_NVFP4_KV_TILE_SCALES + g * 8 + (e >> 1);
            if (e & 1)
                tile[byte_idx] = (tile[byte_idx] & 0x0F) | (nibble << 4);
            else
                tile[byte_idx] = (tile[byte_idx] & 0xF0) | (nibble & 0x0F);
        }
    }

    float tile_norm = (head_dim > 0) ? (float)sqrt(sum_sq / head_dim) : 1.0f;
    if (tile_norm < 1e-10f) tile_norm = 1.0f;
    *(float *)(tile + DEN_NVFP4_KV_TILE_NORM_OFF) = tile_norm;

    tile[DEN_NVFP4_KV_TILE_DISPATCH] = DEN_NVFP4_KV_META_SW;
    tile[DEN_NVFP4_KV_TILE_KSTRIDE]  = (head_dim + 63) / 64;
}

// ═══════════════════════════════════════════════════════════
// Device: dequantize single element from tile (inline for attn)
// ═══════════════════════════════════════════════════════════

__device__ __forceinline__ float kv_dequantize_element(
    const uint8_t * __restrict__ tile,
    int elem_idx)
{
    int group    = elem_idx / DEN_NVFP4_KV_TILE_GROUP_SZ;
    int in_group = elem_idx % DEN_NVFP4_KV_TILE_GROUP_SZ;

    uint8_t scale_code = tile[group];
    float scale = kv_ue4m3_lut[scale_code & 0x0F];

    int nibble_byte_idx = DEN_NVFP4_KV_TILE_SCALES + group * 8 + (in_group >> 1);
    uint8_t nibble = tile[nibble_byte_idx];
    if (in_group & 1) nibble >>= 4;
    else              nibble &= 0x0F;

    float val = kv_e2m1_lut[nibble & 0x07];
    if (nibble & 0x08) val = -val;
    return val * scale;
}

// ═══════════════════════════════════════════════════════════
// Device: dequantize entire tile → float vector
// ═══════════════════════════════════════════════════════════

__device__ static void kv_dequantize_tile(
    const uint8_t * __restrict__ tile,
    float * vec,
    int head_dim)
{
    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;

    for (int g = 0; g < n_groups; g++) {
        float scale = kv_ue4m3_lut[tile[g] & 0x0F];
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end   = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;

        for (int e = 0; e < blk_end - blk_start; e++) {
            int idx = blk_start + e;
            int nb_byte = DEN_NVFP4_KV_TILE_SCALES + g * 8 + (e >> 1);
            uint8_t nb = tile[nb_byte];
            if (e & 1) nb >>= 4; else nb &= 0x0F;
            float val = kv_e2m1_lut[nb & 0x07];
            if (nb & 0x08) val = -val;
            vec[idx] = val * scale;
        }
    }
}

// ═══════════════════════════════════════════════════════════
// K8 tile: uint8 quantize / dequantize (ThriftAttention K8V8)
// Same 16-group structure, but elements stored as full uint8
// instead of 4-bit E2M1 nibbles. 2× quality at 2× storage.
// ═══════════════════════════════════════════════════════════

__device__ static void kv_quantize_tile_k8(
    const float * __restrict__ vec,
    uint8_t * tile,
    int head_dim,
    unsigned int * hist = NULL)
{
    kv_hist_record(vec, head_dim, hist);

    for (int i = 0; i < DEN_NVFP4_KV_TILE_BYTES_K8; i += 16) {
        if (i + 16 <= DEN_NVFP4_KV_TILE_BYTES_K8)
            *(uint4 *)(tile + i) = make_uint4(0, 0, 0, 0);
    }

    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;

    double sum_sq = 0.0;
    for (int i = 0; i < head_dim; i++)
        sum_sq += (double)vec[i] * (double)vec[i];

    for (int g = 0; g < n_groups; g++) {
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;
        int n_in_blk = blk_end - blk_start;

        float max_abs = 0.0f;
        for (int e = 0; e < n_in_blk; e++) {
            float av = vec[blk_start + e];
            if (av < 0.0f) av = -av;
            if (av > max_abs) max_abs = av;
        }

        uint8_t scale_code = 0;
        if (max_abs >= 1e-10f) {
            float ideal_scale = max_abs / 6.0f;
            float best_err = fabsf(ideal_scale - kv_ue4m3_lut[1]);
            uint8_t best_code = 1;
            #pragma unroll
            for (int c = 2; c < 16; c++) {
                float err = fabsf(ideal_scale - kv_ue4m3_lut[c]);
                if (err < best_err) { best_err = err; best_code = (uint8_t)c; }
            }
            scale_code = best_code;
        }
        tile[g] = scale_code;
        float scale = kv_ue4m3_lut[scale_code];

        for (int e = 0; e < n_in_blk; e++) {
            float val = vec[blk_start + e];
            float qval = (scale > 1e-10f) ? val / (scale * 6.0f) : 0.0f;
            if (qval > 1.0f) qval = 1.0f;
            if (qval < -1.0f) qval = -1.0f;
            int u8 = (int)((qval + 1.0f) * 127.5f);
            if (u8 < 0) u8 = 0;
            if (u8 > 255) u8 = 255;
            tile[DEN_NVFP4_KV_TILE_SCALES + g * DEN_NVFP4_KV_TILE_GROUP_SZ + e] = (uint8_t)u8;
        }
    }

    float tile_norm = (head_dim > 0) ? (float)sqrt(sum_sq / head_dim) : 1.0f;
    if (tile_norm < 1e-10f) tile_norm = 1.0f;
    *(float *)(tile + DEN_NVFP4_KV_TILE_NORM_OFF_K8) = tile_norm;
    tile[DEN_NVFP4_KV_TILE_DISPATCH_K8] = DEN_NVFP4_KV_META_K8V8;
    tile[DEN_NVFP4_KV_TILE_KSTRIDE_K8]  = (head_dim + 63) / 64;
}

__device__ __forceinline__ float kv_dequantize_element_k8(
    const uint8_t * __restrict__ tile,
    int elem_idx)
{
    int group    = elem_idx / DEN_NVFP4_KV_TILE_GROUP_SZ;
    int in_group = elem_idx % DEN_NVFP4_KV_TILE_GROUP_SZ;
    float scale  = kv_ue4m3_lut[tile[group] & 0x0F];
    int byte_idx = DEN_NVFP4_KV_TILE_SCALES + group * DEN_NVFP4_KV_TILE_GROUP_SZ + in_group;
    uint8_t u8   = tile[byte_idx];
    float val = ((float)(int)u8 - 127.5f) / 127.5f * 6.0f;
    return val * scale;
}

// K8 tile: dequantize entire 288-byte tile → float vector (for host load path)
__device__ static void kv_dequantize_tile_k8(
    const uint8_t * __restrict__ tile,
    float * vec,
    int head_dim)
{
    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;

    for (int g = 0; g < n_groups; g++) {
        float scale = kv_ue4m3_lut[tile[g] & 0x0F];
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end   = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;

        for (int e = 0; e < blk_end - blk_start; e++) {
            int byte_idx = DEN_NVFP4_KV_TILE_SCALES + g * DEN_NVFP4_KV_TILE_GROUP_SZ + e;
            uint8_t u8   = tile[byte_idx];
            vec[blk_start + e] = (((float)(int)u8 - 127.5f) / 127.5f * 6.0f) * scale;
        }
    }
}

// ═══════════════════════════════════════════════════════════
// K6 tile: uint6 quantize / dequantize (K6V4 Asymmetric KV)
// 256 elements × 6 bits = 1536 bits = 192 bytes, packed as
// 4 elem → 3 bytes (4×6bit = 24bit = 3 bytes).
// Range: unsigned 0-63. Signed during dequant via bias 31.5.
// V tile: unchanged 4-bit E2M1 (existing K4V4 tile, 160B).
// ═══════════════════════════════════════════════════════════

// Pack 4 uint6 elements → 3 bytes
__device__ __forceinline__ void kv_pack_uint6_4to3(
    const uint8_t e[4], uint8_t packed[3])
{
    packed[0] = (e[0] & 0x3F) | ((e[1] & 0x03) << 6);
    packed[1] = ((e[1] >> 2) & 0x0F) | ((e[2] & 0x0F) << 4);
    packed[2] = ((e[2] >> 4) & 0x03) | ((e[3] & 0x3F) << 2);
}

// Unpack 3 bytes → 4 uint6 elements
__device__ __forceinline__ void kv_unpack_uint6_3to4(
    const uint8_t packed[3], uint8_t e[4])
{
    e[0] = packed[0] & 0x3F;
    e[1] = ((packed[0] >> 6) & 0x03) | ((packed[1] & 0x0F) << 2);
    e[2] = ((packed[1] >> 4) & 0x0F) | ((packed[2] & 0x03) << 4);
    e[3] = (packed[2] >> 2) & 0x3F;
}

__device__ static void kv_quantize_tile_k6(
    const float * __restrict__ vec,
    uint8_t * tile,
    int head_dim,
    unsigned int * hist = NULL)
{
    (void)hist; // K6 uses unsigned range — E2M1 scale hist not meaningful here

    // Zero tile
    for (int i = 0; i < DEN_NVFP4_KV_TILE_BYTES_K6; i += 16) {
        if (i + 16 <= DEN_NVFP4_KV_TILE_BYTES_K6)
            *(uint4 *)(tile + i) = make_uint4(0, 0, 0, 0);
    }

    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;

    double sum_sq = 0.0;
    for (int i = 0; i < head_dim; i++)
        sum_sq += (double)vec[i] * (double)vec[i];

    // 6-bit unsigned range: 0-63 (64 levels). Scale = max_abs/63.
    // Quant: u6 = round((val + 6*scale) / (12*scale) * 63) clamped to [0,63]
    //        u6 = round((val/scale + 6.0) / 12.0 * 63.0)
    // Signed val in [-6*scale, +6*scale] → u6 in [0, 63]
    for (int g = 0; g < n_groups; g++) {
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;
        int n_in_blk = blk_end - blk_start;

        float max_abs = 0.0f;
        for (int e = 0; e < n_in_blk; e++) {
            float av = vec[blk_start + e];
            if (av < 0.0f) av = -av;
            if (av > max_abs) max_abs = av;
        }

        uint8_t scale_code = 0;
        if (max_abs >= 1e-10f) {
            float ideal_scale = max_abs / 63.0f;  // K6: unsigned range 0-63
            float best_err = fabsf(ideal_scale - kv_ue4m3_lut[1]);
            uint8_t best_code = 1;
            #pragma unroll
            for (int c = 2; c < 16; c++) {
                float err = fabsf(ideal_scale - kv_ue4m3_lut[c]);
                if (err < best_err) { best_err = err; best_code = (uint8_t)c; }
            }
            scale_code = best_code;
        }

        tile[g] = scale_code;
        float scale = kv_ue4m3_lut[scale_code];

        // Quantize all 16 elements, then pack as 4×3byte groups
        uint8_t u6_buf[DEN_NVFP4_KV_TILE_GROUP_SZ];
        for (int e = 0; e < n_in_blk; e++) {
            float val = vec[blk_start + e];
            // Map signed [-6*scale, +6*scale] → unsigned [0, 63]
            float normed = (scale > 1e-10f) ? (val / (scale * 6.0f)) : 0.0f;
            if (normed >  1.0f) normed =  1.0f;
            if (normed < -1.0f) normed = -1.0f;
            // [-1, +1] → [0, 63]
            int u6 = (int)((normed + 1.0f) * 31.5f + 0.5f);
            if (u6 < 0)  u6 = 0;
            if (u6 > 63) u6 = 63;
            u6_buf[e] = (uint8_t)u6;
        }

        // Pad unused elements to zero
        for (int e = n_in_blk; e < DEN_NVFP4_KV_TILE_GROUP_SZ; e++)
            u6_buf[e] = 0;

        // Pack 16 elements → 12 bytes (4× 4elem→3byte groups)
        int byte_base = DEN_NVFP4_KV_TILE_SCALES + g * 12;
        for (int sg = 0; sg < 4; sg++) {
            kv_pack_uint6_4to3(&u6_buf[sg * 4], &tile[byte_base + sg * 3]);
        }
    }

    float tile_norm = (head_dim > 0) ? (float)sqrt(sum_sq / head_dim) : 1.0f;
    if (tile_norm < 1e-10f) tile_norm = 1.0f;
    *(float *)(tile + DEN_NVFP4_KV_TILE_NORM_OFF_K6) = tile_norm;
    tile[DEN_NVFP4_KV_TILE_DISPATCH_K6] = DEN_NVFP4_KV_META_K6V4;
    tile[DEN_NVFP4_KV_TILE_KSTRIDE_K6]  = (head_dim + 63) / 64;
}

// Dequantize single element from K6 tile (inline for attention).
// Signed output: val = (u6 - 31.5f) / 31.5f * 6.0f * scale
// Range: [-6*scale, +6*scale]
__device__ __forceinline__ float kv_dequantize_element_k6(
    const uint8_t * __restrict__ tile,
    int elem_idx)
{
    int group    = elem_idx / DEN_NVFP4_KV_TILE_GROUP_SZ;
    int in_group = elem_idx % DEN_NVFP4_KV_TILE_GROUP_SZ;

    float scale = kv_ue4m3_lut[tile[group] & 0x0F];

    // 16 elems per group, 4 sub-groups × 4 elem → 12 bytes
    int sg       = in_group / 4;           // sub-group 0-3
    int sg_off   = in_group % 4;           // element 0-3 within sub-group
    int byte_off = DEN_NVFP4_KV_TILE_SCALES + group * 12 + sg * 3;

    uint8_t packed[3] = { tile[byte_off], tile[byte_off+1], tile[byte_off+2] };
    uint8_t e[4];
    kv_unpack_uint6_3to4(packed, e);

    float val = ((float)(int)e[sg_off] - 31.5f) / 31.5f * 6.0f;
    return val * scale;
}

// K6 tile: dequantize entire 224-byte tile → float vector
__device__ static void kv_dequantize_tile_k6(
    const uint8_t * __restrict__ tile,
    float * vec,
    int head_dim)
{
    int n_groups = (head_dim + DEN_NVFP4_KV_TILE_GROUP_SZ - 1) / DEN_NVFP4_KV_TILE_GROUP_SZ;
    if (n_groups > DEN_NVFP4_KV_TILE_GROUPS) n_groups = DEN_NVFP4_KV_TILE_GROUPS;

    for (int g = 0; g < n_groups; g++) {
        float scale = kv_ue4m3_lut[tile[g] & 0x0F];
        int blk_start = g * DEN_NVFP4_KV_TILE_GROUP_SZ;
        int blk_end   = blk_start + DEN_NVFP4_KV_TILE_GROUP_SZ;
        if (blk_end > head_dim) blk_end = head_dim;

        for (int e = 0; e < blk_end - blk_start; e++) {
            int sg     = e / 4;
            int sg_off = e % 4;
            int byte_off = DEN_NVFP4_KV_TILE_SCALES + g * 12 + sg * 3;
            uint8_t packed[3] = { tile[byte_off], tile[byte_off+1], tile[byte_off+2] };
            uint8_t eu[4];
            kv_unpack_uint6_3to4(packed, eu);
            float val = ((float)(int)eu[sg_off] - 31.5f) / 31.5f * 6.0f;
            vec[blk_start + e] = val * scale;
        }
    }
}

// Kernel: dequantize 224-byte K6 tiles → float [n_kv_heads, head_dim]
__global__ void kv_dequantize_kernel_k6(
    const uint8_t * __restrict__ d_tiles,
    float         * __restrict__ d_vec,
    int n_kv_heads, int head_dim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_kv_heads) return;
    const uint8_t * tile = d_tiles + (size_t)h * DEN_NVFP4_KV_TILE_BYTES_K6;
    float * vec = d_vec + (size_t)h * head_dim;
    kv_dequantize_tile_k6(tile, vec, head_dim);
}

// Kernel: dequantize 288-byte k8 tiles → float [n_kv_heads, head_dim]
__global__ void kv_dequantize_kernel_k8(
    const uint8_t * __restrict__ d_tiles,
    float         * __restrict__ d_vec,
    int n_kv_heads, int head_dim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_kv_heads) return;
    const uint8_t * tile = d_tiles + (size_t)h * DEN_NVFP4_KV_TILE_BYTES_K8;
    float * vec = d_vec + (size_t)h * head_dim;
    kv_dequantize_tile_k8(tile, vec, head_dim);
}

// ═══════════════════════════════════════════════════════════
// Kernel: quantize float [n_kv_heads, head_dim] → tiles
// ═══════════════════════════════════════════════════════════

__global__ void kv_quantize_kernel(
    const float * __restrict__ d_vec,
    uint8_t     * __restrict__ d_tiles,
    int n_kv_heads, int head_dim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_kv_heads) return;
    const float * vec  = d_vec   + (size_t)h * head_dim;
    uint8_t     * tile = d_tiles + (size_t)h * DEN_NVFP4_KV_TILE_BYTES;
    kv_quantize_tile(vec, tile, head_dim);
}

// ═══════════════════════════════════════════════════════════
// Kernel: dequantize tiles → float [n_kv_heads, head_dim]
// ═══════════════════════════════════════════════════════════

__global__ void kv_dequantize_kernel(
    const uint8_t * __restrict__ d_tiles,
    float         * __restrict__ d_vec,
    int n_kv_heads, int head_dim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_kv_heads) return;
    const uint8_t * tile = d_tiles + (size_t)h * DEN_NVFP4_KV_TILE_BYTES;
    float * vec = d_vec + (size_t)h * head_dim;
    kv_dequantize_tile(tile, vec, head_dim);
}

// ═══════════════════════════════════════════════════════════
// Kernel: fused NVFP4 attention (3-phase: QK^T, softmax, V sum)
//
// Each block = one head, blockDim = head_dim (128 or 256 threads).
// Shared memory: smem[seq_len scores + n_warps warp sums] float.
// Q@K^T uses on-the-fly tile dequant via kv_dequantize_element.
// ═══════════════════════════════════════════════════════════

__global__ void kv_nvfp4_attention_kernel(
    const float * __restrict__ d_Q,
    const float * __restrict__ d_k_tail,
    const float * __restrict__ d_v_tail,
    const uint8_t * __restrict__ d_k_tiles,
    const uint8_t * __restrict__ d_v_tiles,
    float * __restrict__ d_output,
    float * __restrict__ d_hot_scores,
    int   * __restrict__ d_hot_positions,
    int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int max_seq, int tail_tokens,
    int thrift_attention)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;

    // GQA head map: all query heads in a group share ONE KV head.
    //   gqa_ratio = n_heads / n_kv_heads  (query heads per KV head)
    //   kv_head   = head / gqa_ratio
    // e.g. 35B Ornith (16 query heads, 2 KV heads): heads 0-7 -> KV 0, heads 8-15 -> KV 1.
    // This is the canonical grouped form; identical to `head * n_kv_heads / n_heads`
    // when n_heads % n_kv_heads == 0, but explicit about the grouping.
    const int gqa_ratio = (n_kv_heads > 0) ? (n_heads / n_kv_heads) : 1;
    int kv_head = (gqa_ratio <= 1) ? head : (head / gqa_ratio);
    int k_tile_bytes, v_tile_bytes;
    if (thrift_attention == 2) { // K6V4: K=6-bit packed, V=4-bit E2M1
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K6;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
    } else if (thrift_attention) { // K8V8
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K8;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K8;
    } else { // K4V4
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
    }

    // PRECISION TAIL: the latest `tail_tokens` cache positions live at F32 in
    // d_k_tail/d_v_tail (slot = position % tail_tokens). Positions older than the
    // tail window are quantized NVFP4 tiles at absolute position t in d_k_tiles.
    // A position p is in the tail iff p >= seq_len - tail_tokens (clamped to 0).
    int tail_start = seq_len - tail_tokens;
    if (tail_start < 0) tail_start = 0;

    extern __shared__ float smem[];
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);
    int tid = threadIdx.x;
    float q_val = d_Q[(size_t)head * head_dim + tid];
    float * smem_scores = smem;
    float * warp_sums   = smem + seq_len;

    // ── Register-Resident Hot-Token State ──
    // Thread 0 loads the previous step's top-4 attention scores + positions
    // into registers. These are tiny (8 values), no shared memory needed.
    // Used for: (1) validation that hot tokens are stable across steps,
    // (2) future priority-read path for K/V register cache.
    int   prev_hot_pos[4]   = {0, 0, 0, 0};
    float prev_hot_score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (tid == 0 && d_hot_positions && seq_len > 1) {
        int off = head * DEN_NVFP4_KV_REGISTER_HOT_TOKENS;
        #pragma unroll
        for (int i = 0; i < DEN_NVFP4_KV_REGISTER_HOT_TOKENS; i++) {
            prev_hot_pos[i]   = d_hot_positions[off + i];
            prev_hot_score[i] = d_hot_scores[off + i];
        }
    }

    // Phase 1: Q @ K^T for all cached tokens
    for (int t = 0; t < seq_len; t++) {
        float k_val;
        if (t >= tail_start) {
            // PRECISION TAIL: exact F32 K for a recent token.
            int slot = t % tail_tokens;
            k_val = d_k_tail[((size_t)slot * n_kv_heads + kv_head) * head_dim + tid];
        } else {
            // Old context: NVFP4 tile at absolute position t.
            const uint8_t * tile = d_k_tiles +
                ((size_t)t * n_kv_heads + kv_head) * k_tile_bytes;
            if (thrift_attention == 2) {
                k_val = kv_dequantize_element_k6(tile, tid);
            } else if (thrift_attention) {
                k_val = kv_dequantize_element_k8(tile, tid);
            } else {
                k_val = kv_dequantize_element(tile, tid);
            }
        }

        float partial = q_val * k_val;
        float wsum = partial;
        unsigned active = __activemask();
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            wsum += __shfl_xor_sync(active, wsum, offset);

        if (tid % 32 == 0) warp_sums[tid / 32] = wsum;
        __syncthreads();

        if (tid == 0) {
            float total = 0.0f;
            int n_warps = (head_dim + 31) / 32;
            for (int w = 0; w < n_warps; w++) total += warp_sums[w];
            smem_scores[t] = total * inv_sqrt_hd;
        }
    }
    __syncthreads();

    // Phase 2: Softmax (single-threaded) + Register-Resident Hot-Token Tracking
    // Thread 0: compute softmax, then find the 4 tokens with highest
    // attention mass. These "hot" positions persist to the next decode step
    // via the d_hot_scores/d_hot_positions GPU buffer (loaded into regs above).
    if (tid == 0) {
        float mx = smem_scores[0];
        for (int t = 1; t < seq_len; t++)
            if (smem_scores[t] > mx) mx = smem_scores[t];

        float sum_exp = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            smem_scores[t] = expf(smem_scores[t] - mx);
            sum_exp += smem_scores[t];
        }

        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t < seq_len; t++)
            smem_scores[t] *= inv_sum;

        // ── Register-Resident Hot-Token Tracking ──
        // Find top-K attention scores via register-resident insertion sort.
        // 4 floats + 4 ints = 8 registers on thread 0 (negligible).
        int   hot_pos[4]   = {0, 0, 0, 0};
        float hot_score[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int t = 0; t < seq_len; t++) {
            float score = smem_scores[t];
            // Insertion sort into descending top-4 (register only, no smem).
            #pragma unroll
            for (int i = 0; i < DEN_NVFP4_KV_REGISTER_HOT_TOKENS; i++) {
                if (score > hot_score[i]) {
                    // Shift tail down by 1.
                    #pragma unroll
                    for (int j = DEN_NVFP4_KV_REGISTER_HOT_TOKENS - 1; j > i; j--) {
                        hot_score[j] = hot_score[j - 1];
                        hot_pos[j]   = hot_pos[j - 1];
                    }
                    hot_score[i] = score;
                    hot_pos[i]   = t;
                    break;
                }
            }
        }

        // Write hot tokens to global buffer for NEXT decode step.
        // This is the only global write — everything above was register-only.
        if (d_hot_positions) {
            int off = head * DEN_NVFP4_KV_REGISTER_HOT_TOKENS;
            #pragma unroll
            for (int i = 0; i < DEN_NVFP4_KV_REGISTER_HOT_TOKENS; i++) {
                d_hot_positions[off + i] = hot_pos[i];
                d_hot_scores[off + i]    = hot_score[i];
            }
        }
    }
    __syncthreads();

    // Phase 3: Weighted V sum
    float output_val = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        float v_val;
        if (t >= tail_start) {
            // PRECISION TAIL: exact F32 V for a recent token.
            int slot = t % tail_tokens;
            v_val = d_v_tail[((size_t)slot * n_kv_heads + kv_head) * head_dim + tid];
        } else {
            // Old context: NVFP4 tile at absolute position t.
            const uint8_t * tile = d_v_tiles +
                ((size_t)t * n_kv_heads + kv_head) * v_tile_bytes;
            if (thrift_attention == 2) {
                v_val = kv_dequantize_element(tile, tid);  // V is 4-bit E2M1 in K6V4
            } else if (thrift_attention) {
                v_val = kv_dequantize_element_k8(tile, tid);
            } else {
                v_val = kv_dequantize_element(tile, tid);
            }
        }
        output_val += smem_scores[t] * v_val;
    }

    d_output[(size_t)head * head_dim + tid] = output_val;
}

// ═══════════════════════════════════════════════════════════
// Kernel: store + quantize K/V (token 0 → anchor, rest → tiles)
// ═══════════════════════════════════════════════════════════

__global__ void kv_store_quantize_kernel(
    const float * __restrict__ d_k,
    const float * __restrict__ d_v,
    uint8_t     * __restrict__ d_k_tiles,
    uint8_t     * __restrict__ d_v_tiles,
    float       * __restrict__ d_k_tail,
    float       * __restrict__ d_v_tail,
    int n_kv_heads, int head_dim,
    int seq_pos, int max_seq, int tail_tokens,
    int store_k, int store_v,
    int thrift_attention)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_kv_heads) return;
    if (tail_tokens < 1) return;

    int k_tile_bytes, v_tile_bytes;
    if (thrift_attention == 2) { // K6V4: K=6-bit packed, V=4-bit E2M1
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K6;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
    } else if (thrift_attention) { // K8V8
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K8;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K8;
    } else { // K4V4
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
    }

    unsigned int * hist_k = (g_kv_hist_enabled) ? g_kv_hist_k : NULL;
    unsigned int * hist_v = (g_kv_hist_enabled) ? g_kv_hist_v : NULL;

    // tail slot this token occupies (ring): tail[seq_pos % tail_tokens]
    int slot = seq_pos % tail_tokens;

    // ── 1) EVICT the token aging OUT of the precision tail into the NVFP4 tiles ──
    // The token at absolute position p = seq_pos - tail_tokens currently lives in
    // tail[slot] (same ring slot we're about to overwrite). Once seq_pos reaches
    // tail_tokens, position 0 leaves the window; each subsequent store evicts the
    // token that is exactly tail_tokens behind. Quantize it BEFORE the new token
    // overwrites the slot, so attention still finds it (as a tile) afterwards.
    if (seq_pos >= tail_tokens) {
        int evict_pos = seq_pos - tail_tokens;              // absolute position aging out
        if (evict_pos < max_seq) {
            const float * k_src = d_k_tail + ((size_t)slot * n_kv_heads + h) * head_dim;
            const float * v_src = d_v_tail + ((size_t)slot * n_kv_heads + h) * head_dim;
            if (store_k) {
                uint8_t * k_tile = d_k_tiles + ((size_t)evict_pos * n_kv_heads + h) * k_tile_bytes;
                if (thrift_attention == 2) {
                    kv_quantize_tile_k6(k_src, k_tile, head_dim, hist_k);
                } else if (thrift_attention) {
                    kv_quantize_tile_k8(k_src, k_tile, head_dim, hist_k);
                } else {
                    kv_quantize_tile(k_src, k_tile, head_dim, hist_k);
                }
            }
            if (store_v) {
                uint8_t * v_tile = d_v_tiles + ((size_t)evict_pos * n_kv_heads + h) * v_tile_bytes;
                if (thrift_attention == 2) {
                    kv_quantize_tile(v_src, v_tile, head_dim, hist_v);  // V stays 4-bit in K6V4
                } else if (thrift_attention) {
                    kv_quantize_tile_k8(v_src, v_tile, head_dim, hist_v);
                } else {
                    kv_quantize_tile(v_src, v_tile, head_dim, hist_v);
                }
            }
        }
    }

    // ── 2) STORE the new token at F32 in the precision tail ──
    if (store_k) {
        const float * k_src = d_k + (size_t)h * head_dim;
        float * k_dst = d_k_tail + ((size_t)slot * n_kv_heads + h) * head_dim;
        for (int i = 0; i < head_dim; i++) k_dst[i] = k_src[i];
    }
    if (store_v) {
        const float * v_src = d_v + (size_t)h * head_dim;
        float * v_dst = d_v_tail + ((size_t)slot * n_kv_heads + h) * head_dim;
        for (int i = 0; i < head_dim; i++) v_dst[i] = v_src[i];
    }
}

// ═══════════════════════════════════════════════════════════
// Global cache instance
// ═══════════════════════════════════════════════════════════

den_nvfp4_kv_cache g_nvfp4_kv;

// Scale-distribution probe host gate (defined below, used by store above)
static int kv_hist_host_enabled(void);

bool den_nvfp4_kv_is_wanted(void) {
    static int checked = 0;
    static int wanted = 0;
    if (!checked) {
        const char * env = getenv("DEN_NVFP4_KV_CACHE");
        // env=1: explicitly enabled, env=0: explicitly disabled, unset: auto-detect
        if (env && env[0] == '0') {
            wanted = 0;
        } else {
            wanted = 1; // enabled by default or via env=1 — auto-enable handled by init caller
        }
        checked = 1;
    }
    return wanted;
}

bool den_nvfp4_kv_is_active(void) {
    if (!den_nvfp4_kv_is_wanted()) return false;
    return g_nvfp4_kv.enabled && g_nvfp4_kv.initialized;
}

void den_nvfp4_kv_set_active_cache(den_nvfp4_kv_cache * cache) {
    (void)cache; // unused — the global is the active one
}

// Public API: called from llama_init_from_model.
void ggml_backend_cuda_nvfp4_kv_init(
    int n_attn_layers, int n_kv_heads, int head_dim, int max_seq,
    int thrift_attention)
{
    if (g_nvfp4_kv.initialized) return;
    if (head_dim != 128 && head_dim != 256) return;
    if (max_seq > DEN_NVFP4_KV_MAX_SEQ) max_seq = DEN_NVFP4_KV_MAX_SEQ;
    if (max_seq < 1) max_seq = 1;
    den_nvfp4_kv_init(&g_nvfp4_kv, n_attn_layers, n_kv_heads, head_dim, max_seq, thrift_attention);
}

void ggml_backend_cuda_nvfp4_kv_reset_all(void) {
    // Drain in-flight kernels (warmup on main_stream) before zeroing buffers.
    // Without this, a warmup store on main_stream races with the memset on the
    // default stream — stale warmup data survives the reset → 1.0% match.
    cudaDeviceSynchronize();
    den_nvfp4_kv_reset_all_seq_len(&g_nvfp4_kv);
}

// Lazy init: called from ggml-cuda.cu SET_ROWS hook on first cache access.
void den_nvfp4_kv_lazy_init(int n_kv_heads, int head_dim, int max_seq) {
    if (g_nvfp4_kv.initialized) return;
    if (head_dim != 128 && head_dim != 256) {
        // NVFP4 tile format supports head_dim=128 or 256 (16 groups x 16 elems)
        return;
    }
    // Clamp max_seq to supported range
    if (max_seq > DEN_NVFP4_KV_MAX_SEQ) max_seq = DEN_NVFP4_KV_MAX_SEQ;
    if (max_seq < 1) max_seq = 1;
    den_nvfp4_kv_init(&g_nvfp4_kv, DEN_NVFP4_KV_MAX_LAYERS, n_kv_heads, head_dim, max_seq, 0);
}

// ═══════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════

int den_nvfp4_kv_init(den_nvfp4_kv_cache * cache,
                       int n_attn_layers, int n_kv_heads,
                       int head_dim, int max_seq,
                       int thrift_attention)
{
    if (!cache) return -1;
    memset(cache, 0, sizeof(*cache));

    cache->enabled = (head_dim == 128 || head_dim == 256); // tile format: 16 groups x 16 elems
    if (!cache->enabled) {
        fprintf(stderr, "KV NVFP4: disabled (head_dim=%d, need 128 or 256)\n", head_dim);
        return 0;
    }

    // Probe mode: reset/enable histogram here (host init, runs before any CUDA
    // graph capture begins). Accumulation happens inside the store kernel, which
    // is capture-safe. The dump is deferred to den_nvfp4_kv_free (post-capture).
    den_nvfp4_kv_hist_begin();

    cache->thrift_attention = thrift_attention;
    cache->n_attn_layers = n_attn_layers;
    cache->n_kv_heads    = n_kv_heads;
    cache->head_dim      = head_dim;
    cache->max_seq       = (max_seq > 0 && max_seq <= DEN_NVFP4_KV_MAX_SEQ)
                           ? max_seq : DEN_NVFP4_KV_MAX_SEQ;

    // PRECISION TAIL: resolve size — env DEN_NVFP4_KV_TAIL overrides the default
    // (DEN_NVFP4_KV_TAIL_TOKENS). Clamp to [1, max_seq]: the tail can never exceed
    // the cache, and needs at least 1 token. When tail_tokens == max_seq the entire
    // cache is F32 (no tiles allocated).
    cache->tail_tokens = DEN_NVFP4_KV_TAIL_TOKENS;
    {
        const char * env_tail = getenv(DEN_NVFP4_KV_TAIL_ENV);
        if (env_tail && env_tail[0]) {
            int v = atoi(env_tail);
            if (v > 0) cache->tail_tokens = v;
        }
        if (cache->tail_tokens > cache->max_seq) cache->tail_tokens = cache->max_seq;
        if (cache->tail_tokens < 1) cache->tail_tokens = 1;
    }

    cudaError_t err = cudaStreamCreateWithFlags(
        (cudaStream_t *)&cache->cuda_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        fprintf(stderr, "KV NVFP4: stream create failed: %s\n",
                cudaGetErrorString(err));
        cache->enabled = 0;
    }

    cache->layers = (den_nvfp4_kv_layer *)calloc(n_attn_layers, sizeof(den_nvfp4_kv_layer));
    if (!cache->layers) {
        fprintf(stderr, "KV NVFP4: layer alloc failed\n");
        den_nvfp4_kv_free(cache);
        return -1;
    }

    // PRECISION TAIL F32 buffers: [tail_tokens * n_kv_heads * head_dim] floats.
    size_t tail_bytes = (size_t)n_kv_heads * head_dim * cache->tail_tokens * sizeof(float);
    // NVFP4 tile buffers: only tokens OLDER than the tail window get quantized.
    int k_tile_bytes, v_tile_bytes;
    if (thrift_attention == 2) { // K6V4: asymmetric — K at 6-bit (224B), V at 4-bit (160B)
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K6;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
    } else if (thrift_attention) { // K8V8: symmetric 8-bit
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K8;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES_K8;
    } else { // K4V4: symmetric 4-bit
        k_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
        v_tile_bytes = DEN_NVFP4_KV_TILE_BYTES;
    }
    int tile_count = cache->max_seq - cache->tail_tokens;
    if (tile_count < 0) tile_count = 0;
    size_t k_tiles_per_layer = (size_t)tile_count * n_kv_heads * k_tile_bytes;
    size_t v_tiles_per_layer = (size_t)tile_count * n_kv_heads * v_tile_bytes;

    int l = 0;
    for (l = 0; l < n_attn_layers; l++) {
        den_nvfp4_kv_layer * layer = &cache->layers[l];

        if (cudaMalloc(&layer->d_k_tail, tail_bytes) != cudaSuccess) goto fail;
        if (cudaMalloc(&layer->d_v_tail, tail_bytes) != cudaSuccess) goto fail;

        if (tile_count > 0) {
            if (cudaMalloc(&layer->d_k_tiles, k_tiles_per_layer) != cudaSuccess) goto fail;
            if (cudaMalloc(&layer->d_v_tiles, v_tiles_per_layer) != cudaSuccess) goto fail;
            if (cudaMalloc(&layer->d_scratch_tile, (size_t)n_kv_heads * k_tile_bytes) != cudaSuccess) goto fail;
        }

        if (cudaMallocHost(&layer->h_readback, tail_bytes) != cudaSuccess) goto fail;

        // Register-resident hot-token state buffers (tiny: n_kv_heads * 4 ints + 4 floats)
        {
            size_t hot_bytes = (size_t)n_kv_heads * DEN_NVFP4_KV_REGISTER_HOT_TOKENS;
            if (cudaMalloc(&layer->d_hot_scores,    hot_bytes * sizeof(float)) != cudaSuccess) goto fail;
            if (cudaMalloc(&layer->d_hot_positions, hot_bytes * sizeof(int))   != cudaSuccess) goto fail;
            cudaMemset(layer->d_hot_scores,    0, hot_bytes * sizeof(float));
            cudaMemset(layer->d_hot_positions, 0, hot_bytes * sizeof(int));
        }

        cudaMemset(layer->d_k_tail, 0, tail_bytes);
        cudaMemset(layer->d_v_tail, 0, tail_bytes);

        // Zero tile buffers — cudaMalloc does NOT zero memory; stale pages
        // from a prior process survive and cause binary-state toggle (GOOD/BAD).
        if (tile_count > 0) {
            cudaMemset(layer->d_k_tiles,      0, k_tiles_per_layer);
            cudaMemset(layer->d_v_tiles,      0, v_tiles_per_layer);
            cudaMemset(layer->d_scratch_tile, 0, (size_t)n_kv_heads * k_tile_bytes);
        }

        layer->max_seq     = cache->max_seq;
        layer->tail_tokens = cache->tail_tokens;
        layer->seq_len     = 0;
        layer->n_kv_heads  = n_kv_heads;
        layer->head_dim    = head_dim;
    }

    cache->initialized = 1;

    // Reset all seq_len to 0 (safety: warmup may have stored dummy tokens)
    den_nvfp4_kv_reset_all_seq_len(cache);

    {
        const char * mode_str = (thrift_attention == 2) ? "K6V4 Asymmetric"
                              : thrift_attention         ? "K8V8 ThriftAttention"
                              :                           "K4V4";
        fprintf(stderr,
            "KV NVFP4: ENABLED (%s, %d layers, %d KV heads, head_dim=%d, max_seq=%d, PRECISION TAIL=%d tokens F32)\n"
            "  tiles(%d): K=%.1f MB V=%.1f MB   tail F32(K+V)=%.2f MB   BF16-equiv=%.1f MB\n",
            mode_str,
            n_attn_layers, n_kv_heads, head_dim, cache->max_seq, cache->tail_tokens,
            tile_count,
            (double)k_tiles_per_layer * n_attn_layers / (1024.0 * 1024.0),
            (double)v_tiles_per_layer * n_attn_layers / (1024.0 * 1024.0),
            (double)2.0 * tail_bytes * n_attn_layers / (1024.0 * 1024.0),
            (double)n_attn_layers * cache->max_seq * n_kv_heads * head_dim * 2 / (1024.0 * 1024.0));
    }

    return 0;

fail:
    fprintf(stderr, "KV NVFP4: cudaMalloc failed at layer %d: %s\n",
            l, cudaGetErrorString(cudaGetLastError()));
    den_nvfp4_kv_free(cache);
    return -1;
}

int den_nvfp4_kv_store(den_nvfp4_kv_cache * cache, int layer,
                        int seq_pos, const float * d_k, const float * d_v,
                        cudaStream_t main_stream)
{
    if (!cache || !cache->initialized) return -1;
    if (layer < 0 || layer >= cache->n_attn_layers) return -1;

    // Allow nullptr for K-only or V-only stores
    int store_k = (d_k != nullptr) ? 1 : 0;
    int store_v = (d_v != nullptr) ? 1 : 0;
    if (!store_k && !store_v) return -1;

    den_nvfp4_kv_layer * kv_layer = &cache->layers[layer];

    if (seq_pos < 0 || seq_pos >= kv_layer->max_seq) {
        fprintf(stderr, "KV NVFP4: seq_pos %d out of range [0, %d)\n",
                seq_pos, kv_layer->max_seq);
        return -1;
    }

    // Always run the store on the caller's main compute stream when provided
    // (falls back to the cache's own stream only if none is given). Same-stream
    // launch guarantees FIFO ordering after the K/V producer on main and before
    // any consumer on main — no cross-stream event wait is needed, which also
    // keeps CUDA graph capture valid.
    cudaStream_t stream = main_stream ? main_stream : (cudaStream_t)cache->cuda_stream;
    int block_size = 128;
    int grid_size  = (cache->n_kv_heads + block_size - 1) / block_size;

    // Clear any stale CUDA errors before launching
    cudaGetLastError();

    // Pass non-null device pointer for the "unused" side (kernel won't touch it)
    const float * k_ptr = store_k ? d_k : d_v;
    const float * v_ptr = store_v ? d_v : d_k;

    kv_store_quantize_kernel<<<grid_size, block_size, 0, stream>>>(
        k_ptr, v_ptr,
        kv_layer->d_k_tiles, kv_layer->d_v_tiles,
        kv_layer->d_k_tail, kv_layer->d_v_tail,
        cache->n_kv_heads, cache->head_dim,
        seq_pos, kv_layer->max_seq, kv_layer->tail_tokens,
        store_k, store_v,
        cache->thrift_attention);

    // Probe mode: throttled periodic dump (every 4 tokens, deduped across layers,
    // and NOT during graph capture). Attention layers are indexed 3,7,11,... so we
    // gate only on seq_pos (dedupe via last-dumped) instead of layer==0.
    if (kv_hist_host_enabled() && !g_den_cuda_graph_capturing && (seq_pos & 3) == 0) {
        static int last_dump_seq = -1;
        if (seq_pos != last_dump_seq) {
            last_dump_seq = seq_pos;
            den_nvfp4_kv_hist_dump();
        }
    }

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int den_nvfp4_kv_load(den_nvfp4_kv_cache * cache, int layer,
                       float * d_k_out, float * d_v_out, int seq_pos,
                       cudaStream_t main_stream)
{
    if (!cache || !cache->initialized) return -1;
    if (layer < 0 || layer >= cache->n_attn_layers) return -1;

    den_nvfp4_kv_layer * kv_layer = &cache->layers[layer];
    if (seq_pos < 0 || seq_pos >= kv_layer->seq_len) return -1;

    // Use the caller's main compute stream when provided (FIFO with main's
    // consumers); fall back to the cache's own stream otherwise.
    cudaStream_t stream = main_stream ? main_stream : (cudaStream_t)cache->cuda_stream;

    int tail_start = kv_layer->seq_len - kv_layer->tail_tokens;
    if (tail_start < 0) tail_start = 0;

    if (seq_pos >= tail_start) {
        // PRECISION TAIL: recent token lives at F32 in the tail buffer, slot = pos % tail_tokens
        int slot = seq_pos % kv_layer->tail_tokens;
        size_t per_token_bytes = (size_t)cache->n_kv_heads * cache->head_dim * sizeof(float);
        size_t offset_bytes    = (size_t)slot * per_token_bytes;
        CUDA_CHECK(cudaMemcpyAsync(d_k_out, (char *)kv_layer->d_k_tail + offset_bytes,
                                   per_token_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_v_out, (char *)kv_layer->d_v_tail + offset_bytes,
                                   per_token_bytes, cudaMemcpyDeviceToDevice, stream));
    } else {
        int tile_idx = seq_pos; // tiles hold absolute positions older than the tail
        int k_tb, v_tb;
        if (cache->thrift_attention == 2) {
            k_tb = DEN_NVFP4_KV_TILE_BYTES_K6;  // K: 224B packed 6-bit
            v_tb = DEN_NVFP4_KV_TILE_BYTES;     // V: 160B 4-bit E2M1
        } else if (cache->thrift_attention) {
            k_tb = DEN_NVFP4_KV_TILE_BYTES_K8;
            v_tb = DEN_NVFP4_KV_TILE_BYTES_K8;
        } else {
            k_tb = DEN_NVFP4_KV_TILE_BYTES;
            v_tb = DEN_NVFP4_KV_TILE_BYTES;
        }
        const uint8_t * k_tile_base = kv_layer->d_k_tiles +
            (size_t)tile_idx * cache->n_kv_heads * k_tb;
        const uint8_t * v_tile_base = kv_layer->d_v_tiles +
            (size_t)tile_idx * cache->n_kv_heads * v_tb;

        int block_size = 128;
        int grid_size  = (cache->n_kv_heads + block_size - 1) / block_size;

        if (cache->thrift_attention == 2) {
            kv_dequantize_kernel_k6<<<grid_size, block_size, 0, stream>>>(
                k_tile_base, d_k_out, cache->n_kv_heads, cache->head_dim);
            kv_dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
                v_tile_base, d_v_out, cache->n_kv_heads, cache->head_dim);
        } else if (cache->thrift_attention) {
            kv_dequantize_kernel_k8<<<grid_size, block_size, 0, stream>>>(
                k_tile_base, d_k_out, cache->n_kv_heads, cache->head_dim);
            kv_dequantize_kernel_k8<<<grid_size, block_size, 0, stream>>>(
                v_tile_base, d_v_out, cache->n_kv_heads, cache->head_dim);
        } else {
            kv_dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
                k_tile_base, d_k_out, cache->n_kv_heads, cache->head_dim);
            kv_dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
                v_tile_base, d_v_out, cache->n_kv_heads, cache->head_dim);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

int den_nvfp4_kv_attention(den_nvfp4_kv_cache * cache, int layer,
                            const float * d_Q, float * d_output, int n_heads,
                            cudaStream_t main_stream)
{
    if (!cache || !cache->initialized) return -1;
    if (layer < 0 || layer >= cache->n_attn_layers) return -1;

    den_nvfp4_kv_layer * kv_layer = &cache->layers[layer];
    int seq_len = kv_layer->seq_len;
    if (seq_len < 1) return -1;

    int n_warps_smem = (cache->head_dim + 31) / 32;
    size_t smem_bytes = ((size_t)seq_len + (size_t)n_warps_smem) * sizeof(float);
    if (smem_bytes > DEN_SMEM_MAX_BYTES) smem_bytes = DEN_SMEM_MAX_BYTES;

    // Always run attention on the caller's main compute stream when provided.
    // d_output is read by the NEXT graph node on the main compute stream, so the
    // write must be FIFO-ordered on the same stream to close the cross-stream
    // read-before-write race. Falls back to the cache's own stream if none given.
    cudaStream_t stream = main_stream ? main_stream : (cudaStream_t)cache->cuda_stream;

    // Clear stale errors before launch
    cudaGetLastError();

    // blockDim = head_dim (128 or 256); smem holds seq_len scores + n_warps warp sums
    kv_nvfp4_attention_kernel<<<n_heads, cache->head_dim, smem_bytes, stream>>>(
        d_Q,
        kv_layer->d_k_tail, kv_layer->d_v_tail,
        kv_layer->d_k_tiles, kv_layer->d_v_tiles,
        d_output,
        kv_layer->d_hot_scores, kv_layer->d_hot_positions,
        n_heads, cache->n_kv_heads, cache->head_dim,
        seq_len, kv_layer->max_seq, kv_layer->tail_tokens,
        cache->thrift_attention);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int den_nvfp4_kv_seq_len(const den_nvfp4_kv_cache * cache, int layer) {
    if (!cache || !cache->initialized) return 0;
    if (layer < 0 || layer >= cache->n_attn_layers) return 0;
    return cache->layers[layer].seq_len;
}

int den_nvfp4_kv_set_seq_len(den_nvfp4_kv_cache * cache, int layer, int len) {
    if (!cache || !cache->initialized) return -1;
    if (layer < 0 || layer >= cache->n_attn_layers) return -1;
    if (len < 0 || len > cache->max_seq) return -1;
    cache->layers[layer].seq_len = len;
    return 0;
}

void den_nvfp4_kv_reset_all_seq_len(den_nvfp4_kv_cache * cache) {
    if (!cache || !cache->initialized) return;
    for (int l = 0; l < cache->n_attn_layers; l++) {
        den_nvfp4_kv_layer * layer = &cache->layers[l];
        layer->seq_len = 0;

        // Zero tail buffers — seq_len=0 guarantees no reads, but stale
        // warmup data in the ring buffer can produce 1.0%-match garbage if
        // any code path reads a position that hasn't been overwritten yet.
        size_t tail_bytes = (size_t)layer->n_kv_heads * layer->head_dim
                          * layer->tail_tokens * sizeof(float);
        if (tail_bytes > 0) {
            cudaMemset(layer->d_k_tail, 0, tail_bytes);
            cudaMemset(layer->d_v_tail, 0, tail_bytes);
        }

        // Zero hot-token tracking buffers — stale positions from prior
        // context would bias the next decode step toward invalid tokens.
        {
            size_t hot_bytes = (size_t)layer->n_kv_heads * DEN_NVFP4_KV_REGISTER_HOT_TOKENS;
            if (hot_bytes > 0 && layer->d_hot_scores && layer->d_hot_positions) {
                cudaMemset(layer->d_hot_scores,    0, hot_bytes * sizeof(float));
                cudaMemset(layer->d_hot_positions, 0, hot_bytes * sizeof(int));
            }
        }

        // Also zero tiles to guarantee zero-score (silent) stale reads.
        int k_tb, v_tb;
        if (cache->thrift_attention == 2) {
            k_tb = DEN_NVFP4_KV_TILE_BYTES_K6;
            v_tb = DEN_NVFP4_KV_TILE_BYTES;
        } else if (cache->thrift_attention) {
            k_tb = DEN_NVFP4_KV_TILE_BYTES_K8;
            v_tb = DEN_NVFP4_KV_TILE_BYTES_K8;
        } else {
            k_tb = DEN_NVFP4_KV_TILE_BYTES;
            v_tb = DEN_NVFP4_KV_TILE_BYTES;
        }
        int tile_count = layer->max_seq - layer->tail_tokens;
        if (tile_count < 0) tile_count = 0;
        if (tile_count > 0) {
            size_t kt_bytes = (size_t)tile_count * layer->n_kv_heads * k_tb;
            size_t vt_bytes = (size_t)tile_count * layer->n_kv_heads * v_tb;
            cudaMemset(layer->d_k_tiles, 0, kt_bytes);
            cudaMemset(layer->d_v_tiles, 0, vt_bytes);
        }
    }
}

// ── Scale-distribution probe: host control ──────────────────
static int g_kv_hist_on = -1; // -1 = uninitialized
static int kv_hist_host_enabled(void) {
    if (g_kv_hist_on < 0) {
        const char * env = getenv("DEN_NVFP4_KV_HIST");
        g_kv_hist_on = (env && env[0] == '1') ? 1 : 0;
    }
    return g_kv_hist_on;
}

void den_nvfp4_kv_hist_begin(void) {
    if (!kv_hist_host_enabled()) return;
    cudaDeviceSynchronize();
    void * addr = NULL;
    cudaGetSymbolAddress(&addr, g_kv_hist_k);
    cudaMemset(addr, 0, sizeof(g_kv_hist_k));
    cudaGetSymbolAddress(&addr, g_kv_hist_v);
    cudaMemset(addr, 0, sizeof(g_kv_hist_v));
    const int one = 1;
    cudaMemcpyToSymbol(g_kv_hist_enabled, &one, sizeof(int));
}

void den_nvfp4_kv_hist_dump(void) {
    if (!kv_hist_host_enabled()) return;
    cudaDeviceSynchronize();
    unsigned int hk[DEN_KV_HIST_BINS], hv[DEN_KV_HIST_BINS];
    cudaMemcpyFromSymbol(hk, g_kv_hist_k, sizeof(hk));
    cudaMemcpyFromSymbol(hv, g_kv_hist_v, sizeof(hv));
    FILE * f = fopen("kv_scale_hist.txt", "w");
    if (!f) return;
    unsigned long long tk = 0, tv = 0;
    for (int i = 0; i < DEN_KV_HIST_BINS; i++) { tk += hk[i]; tv += hv[i]; }
    fprintf(f, "# NVFP4 KV ideal_scale histogram (ideal_scale = max_abs/6 per 16-elem block)\n");
    fprintf(f, "# bin i covers ideal_scale in [i*0.0625, (i+1)*0.0625)\n");
    fprintf(f, "# K blocks=%llu  V blocks=%llu\n", tk, tv);
    fprintf(f, "# idx lo hi k v\n");
    for (int i = 0; i < DEN_KV_HIST_BINS; i++) {
        float lo = i * DEN_KV_HIST_W;
        float hi = (i + 1) * DEN_KV_HIST_W;
        if (hk[i] || hv[i])
            fprintf(f, "%d %.6f %.6f %u %u\n", i, lo, hi, hk[i], hv[i]);
    }
    fclose(f);
}

void den_nvfp4_kv_free(den_nvfp4_kv_cache * cache) {
    if (!cache) return;

    // Probe mode: dump cumulative histogram at teardown (post-capture, safe to sync).
    den_nvfp4_kv_hist_dump();

    for (int l = 0; l < cache->n_attn_layers; l++) {
        den_nvfp4_kv_layer * layer = &cache->layers[l];
        if (layer->d_k_tail)       cudaFree(layer->d_k_tail);
        if (layer->d_v_tail)       cudaFree(layer->d_v_tail);
        if (layer->d_k_tiles)      cudaFree(layer->d_k_tiles);
        if (layer->d_v_tiles)      cudaFree(layer->d_v_tiles);
        if (layer->d_scratch_tile) cudaFree(layer->d_scratch_tile);
        if (layer->h_readback)     cudaFreeHost(layer->h_readback);
        if (layer->d_hot_scores)   cudaFree(layer->d_hot_scores);
        if (layer->d_hot_positions)cudaFree(layer->d_hot_positions);
    }
    free(cache->layers);
    cache->layers = nullptr;
    if (cache->cuda_stream) {
        cudaStreamDestroy((cudaStream_t)cache->cuda_stream);
        cache->cuda_stream = nullptr;
    }
    cache->initialized = 0;
}

double den_nvfp4_kv_compression_ratio(const den_nvfp4_kv_cache * cache) {
    if (!cache || !cache->initialized) return 1.0;
    size_t bf16_per_layer = (size_t)cache->max_seq * cache->n_kv_heads *
                            cache->head_dim * 2;
    int tile_count = cache->max_seq - cache->tail_tokens;
    if (tile_count < 0) tile_count = 0;
    size_t k_tb, v_tb;
    if (cache->thrift_attention == 2) {
        k_tb = DEN_NVFP4_KV_TILE_BYTES_K6;  // K: 224B
        v_tb = DEN_NVFP4_KV_TILE_BYTES;     // V: 160B
    } else if (cache->thrift_attention) {
        k_tb = DEN_NVFP4_KV_TILE_BYTES_K8;
        v_tb = DEN_NVFP4_KV_TILE_BYTES_K8;
    } else {
        k_tb = DEN_NVFP4_KV_TILE_BYTES;
        v_tb = DEN_NVFP4_KV_TILE_BYTES;
    }
    size_t avg_tb = (k_tb + v_tb) / 2; // average tile size for K+V pair
    // NVFP4 footprint = tiles for the quantized region + F32 tail (K+V).
    size_t nvfp4_per_layer = (size_t)tile_count * cache->n_kv_heads * avg_tb +
                             (size_t)cache->tail_tokens * cache->n_kv_heads *
                             cache->head_dim * 2 * (size_t)sizeof(float);
    if (nvfp4_per_layer == 0) return 1.0;
    return (double)bf16_per_layer / (double)nvfp4_per_layer;
}

// ═══════════════════════════════════════════════════════════
// Post-set-rows hook (stub — wired in Step 4)
// ═══════════════════════════════════════════════════════════

void den_nvfp4_kv_post_set_rows(const float * d_dst, const float * d_src,
                                int n_kv_heads, int head_dim,
                                int seq_pos, int layer) {
    if (!den_nvfp4_kv_is_active()) return;
    den_nvfp4_kv_store(&g_nvfp4_kv, layer, seq_pos, d_src, d_src, nullptr);
    (void)d_dst;
    (void)n_kv_heads;
    (void)head_dim;
}
