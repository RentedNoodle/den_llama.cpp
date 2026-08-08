// fattn-nvfp4-kv.cu — NVFP4 KV Cache kernels for den_llama.cpp
//
// Ported from Project Den dengine/src/den_kv_cache.cu
// See fattn-nvfp4-kv.cuh for the public API and data structures.

#include "fattn-nvfp4-kv.cuh"
#include "common.cuh"
#include "ggml-cuda.h"

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
    0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f,
    0.375f, 0.4375f, 1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f
};

// ═══════════════════════════════════════════════════════════
// Device: quantize float vector → NVFP4 tile
// O(1) per-block: max_abs → nearest UE4M3 scale code
// ═══════════════════════════════════════════════════════════

__device__ static void kv_quantize_tile(
    const float * __restrict__ vec,
    uint8_t * tile,
    int head_dim)
{
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
    const float * __restrict__ d_k_anchor,
    const float * __restrict__ d_v_anchor,
    const uint8_t * __restrict__ d_k_tiles,
    const uint8_t * __restrict__ d_v_tiles,
    float * __restrict__ d_output,
    int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int max_seq, int is_anchor)
{
    int head = blockIdx.x;
    if (head >= n_heads) return;

    int kv_head = (n_kv_heads == n_heads) ? head : (head * n_kv_heads / n_heads);

    extern __shared__ float smem[];
    float inv_sqrt_hd = 1.0f / sqrtf((float)head_dim);
    int tid = threadIdx.x;
    float q_val = d_Q[(size_t)head * head_dim + tid];
    float * smem_scores = smem;
    float * warp_sums   = smem + seq_len;

    // Phase 1: Q @ K^T for all cached tokens
    for (int t = 0; t < seq_len; t++) {
        float k_val;
        if (t == 0 && is_anchor) {
            k_val = d_k_anchor[(size_t)kv_head * head_dim + tid];
        } else {
            int tile_idx = is_anchor ? (t - 1) : t;
            if (tile_idx < 0) tile_idx = 0;
            const uint8_t * tile = d_k_tiles +
                ((size_t)tile_idx * n_kv_heads + kv_head) * DEN_NVFP4_KV_TILE_BYTES;
            k_val = kv_dequantize_element(tile, tid);
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

    // Phase 2: Softmax (single-threaded)
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
    }
    __syncthreads();

    // Phase 3: Weighted V sum
    float output_val = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        float v_val;
        if (t == 0 && is_anchor) {
            v_val = d_v_anchor[(size_t)kv_head * head_dim + tid];
        } else {
            int tile_idx = is_anchor ? (t - 1) : t;
            if (tile_idx < 0) tile_idx = 0;
            const uint8_t * tile = d_v_tiles +
                ((size_t)tile_idx * n_kv_heads + kv_head) * DEN_NVFP4_KV_TILE_BYTES;
            v_val = kv_dequantize_element(tile, tid);
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
    float       * __restrict__ d_k_anchor,
    float       * __restrict__ d_v_anchor,
    int n_kv_heads, int head_dim,
    int tile_idx, int max_seq,
    int store_k, int store_v)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_kv_heads) return;

    if (tile_idx == 0) {
        if (store_k) {
            const float * k_src = d_k + (size_t)h * head_dim;
            float * k_dst = d_k_anchor + (size_t)h * head_dim;
            for (int i = 0; i < head_dim; i++) {
                k_dst[i] = k_src[i];
            }
        }
        if (store_v) {
            const float * v_src = d_v + (size_t)h * head_dim;
            float * v_dst = d_v_anchor + (size_t)h * head_dim;
            for (int i = 0; i < head_dim; i++) {
                v_dst[i] = v_src[i];
            }
        }
    } else {
        int t = tile_idx - 1;
        if (t >= max_seq - 1) return;
        if (store_k) {
            const float * k_src = d_k + (size_t)h * head_dim;
            uint8_t * k_tile = d_k_tiles + ((size_t)t * n_kv_heads + h) * DEN_NVFP4_KV_TILE_BYTES;
            kv_quantize_tile(k_src, k_tile, head_dim);
        }
        if (store_v) {
            const float * v_src = d_v + (size_t)h * head_dim;
            uint8_t * v_tile = d_v_tiles + ((size_t)t * n_kv_heads + h) * DEN_NVFP4_KV_TILE_BYTES;
            kv_quantize_tile(v_src, v_tile, head_dim);
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Global cache instance
// ═══════════════════════════════════════════════════════════

den_nvfp4_kv_cache g_nvfp4_kv;

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
    int n_attn_layers, int n_kv_heads, int head_dim, int max_seq)
{
    if (g_nvfp4_kv.initialized) return;
    if (head_dim != 128 && head_dim != 256) return;
    if (max_seq > DEN_NVFP4_KV_MAX_SEQ) max_seq = DEN_NVFP4_KV_MAX_SEQ;
    if (max_seq < 1) max_seq = 1;
    den_nvfp4_kv_init(&g_nvfp4_kv, n_attn_layers, n_kv_heads, head_dim, max_seq);
}

void ggml_backend_cuda_nvfp4_kv_reset_all(void) {
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
    den_nvfp4_kv_init(&g_nvfp4_kv, DEN_NVFP4_KV_MAX_LAYERS, n_kv_heads, head_dim, max_seq);
}

// ═══════════════════════════════════════════════════════════
// Host API
// ═══════════════════════════════════════════════════════════

int den_nvfp4_kv_init(den_nvfp4_kv_cache * cache,
                       int n_attn_layers, int n_kv_heads,
                       int head_dim, int max_seq)
{
    if (!cache) return -1;
    memset(cache, 0, sizeof(*cache));

    cache->enabled = (head_dim == 128 || head_dim == 256); // tile format: 16 groups x 16 elems
    if (!cache->enabled) {
        fprintf(stderr, "KV NVFP4: disabled (head_dim=%d, need 128 or 256)\n", head_dim);
        return 0;
    }

    cache->n_attn_layers = n_attn_layers;
    cache->n_kv_heads    = n_kv_heads;
    cache->head_dim      = head_dim;
    cache->max_seq       = (max_seq > 0 && max_seq <= DEN_NVFP4_KV_MAX_SEQ)
                           ? max_seq : DEN_NVFP4_KV_MAX_SEQ;

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

    size_t anchor_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);
    size_t tiles_per_layer = (size_t)(cache->max_seq - 1) * n_kv_heads * DEN_NVFP4_KV_TILE_BYTES;

    int l = 0;
    for (l = 0; l < n_attn_layers; l++) {
        den_nvfp4_kv_layer * layer = &cache->layers[l];

        if (cudaMalloc(&layer->d_k_anchor, anchor_bytes) != cudaSuccess) goto fail;
        if (cudaMalloc(&layer->d_v_anchor, anchor_bytes) != cudaSuccess) goto fail;

        if (cache->max_seq > 1) {
            if (cudaMalloc(&layer->d_k_tiles, tiles_per_layer) != cudaSuccess) goto fail;
            if (cudaMalloc(&layer->d_v_tiles, tiles_per_layer) != cudaSuccess) goto fail;
            if (cudaMalloc(&layer->d_scratch_tile, (size_t)n_kv_heads * DEN_NVFP4_KV_TILE_BYTES) != cudaSuccess) goto fail;
        }

        if (cudaMallocHost(&layer->h_readback, anchor_bytes) != cudaSuccess) goto fail;

        cudaMemset(layer->d_k_anchor, 0, anchor_bytes);
        cudaMemset(layer->d_v_anchor, 0, anchor_bytes);

        layer->max_seq    = cache->max_seq;
        layer->seq_len    = 0;
        layer->n_kv_heads = n_kv_heads;
        layer->head_dim   = head_dim;
    }

    cache->initialized = 1;

    // Reset all seq_len to 0 (safety: warmup may have stored dummy tokens)
    den_nvfp4_kv_reset_all_seq_len(cache);

    fprintf(stderr,
        "KV NVFP4: ENABLED (%d layers, %d KV heads, head_dim=%d, max_seq=%d)\n"
        "  BF16=%.1f MB vs NVFP4=%.1f MB per layer\n",
        n_attn_layers, n_kv_heads, head_dim, cache->max_seq,
        (double)n_attn_layers * cache->max_seq * n_kv_heads * head_dim * 2 / (1024.0 * 1024.0),
        (double)n_attn_layers * ((size_t)(cache->max_seq - 1) * n_kv_heads * DEN_NVFP4_KV_TILE_BYTES + anchor_bytes) / (1024.0 * 1024.0));

    return 0;

fail:
    fprintf(stderr, "KV NVFP4: cudaMalloc failed at layer %d: %s\n",
            l, cudaGetErrorString(cudaGetLastError()));
    den_nvfp4_kv_free(cache);
    return -1;
}

int den_nvfp4_kv_store(den_nvfp4_kv_cache * cache, int layer,
                        int seq_pos, const float * d_k, const float * d_v)
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

    cudaStream_t stream = (cudaStream_t)cache->cuda_stream;
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
        kv_layer->d_k_anchor, kv_layer->d_v_anchor,
        cache->n_kv_heads, cache->head_dim,
        seq_pos, kv_layer->max_seq, store_k, store_v);

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

int den_nvfp4_kv_load(den_nvfp4_kv_cache * cache, int layer,
                       float * d_k_out, float * d_v_out, int seq_pos)
{
    if (!cache || !cache->initialized) return -1;
    if (layer < 0 || layer >= cache->n_attn_layers) return -1;

    den_nvfp4_kv_layer * kv_layer = &cache->layers[layer];
    if (seq_pos < 0 || seq_pos >= kv_layer->seq_len) return -1;

    cudaStream_t stream = (cudaStream_t)cache->cuda_stream;

    if (seq_pos == 0) {
        size_t bytes = (size_t)cache->n_kv_heads * cache->head_dim * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d_k_out, kv_layer->d_k_anchor, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_v_out, kv_layer->d_v_anchor, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    } else {
        int tile_idx = seq_pos - 1;
        const uint8_t * k_tile_base = kv_layer->d_k_tiles +
            (size_t)tile_idx * cache->n_kv_heads * DEN_NVFP4_KV_TILE_BYTES;
        const uint8_t * v_tile_base = kv_layer->d_v_tiles +
            (size_t)tile_idx * cache->n_kv_heads * DEN_NVFP4_KV_TILE_BYTES;

        int block_size = 128;
        int grid_size  = (cache->n_kv_heads + block_size - 1) / block_size;

        kv_dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
            k_tile_base, d_k_out, cache->n_kv_heads, cache->head_dim);
        kv_dequantize_kernel<<<grid_size, block_size, 0, stream>>>(
            v_tile_base, d_v_out, cache->n_kv_heads, cache->head_dim);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

int den_nvfp4_kv_attention(den_nvfp4_kv_cache * cache, int layer,
                            const float * d_Q, float * d_output, int n_heads)
{
    if (!cache || !cache->initialized) return -1;
    if (layer < 0 || layer >= cache->n_attn_layers) return -1;

    den_nvfp4_kv_layer * kv_layer = &cache->layers[layer];
    int seq_len = kv_layer->seq_len;
    if (seq_len < 1) return -1;

    int n_warps_smem = (cache->head_dim + 31) / 32;
    size_t smem_bytes = ((size_t)seq_len + (size_t)n_warps_smem) * sizeof(float);
    if (smem_bytes > DEN_SMEM_MAX_BYTES) smem_bytes = DEN_SMEM_MAX_BYTES;

    cudaStream_t stream = (cudaStream_t)cache->cuda_stream;

    // Clear stale errors before launch
    cudaGetLastError();

    // blockDim = head_dim (128 or 256); smem holds seq_len scores + n_warps warp sums
    kv_nvfp4_attention_kernel<<<n_heads, cache->head_dim, smem_bytes, stream>>>(
        d_Q,
        kv_layer->d_k_anchor, kv_layer->d_v_anchor,
        kv_layer->d_k_tiles, kv_layer->d_v_tiles,
        d_output,
        n_heads, cache->n_kv_heads, cache->head_dim,
        seq_len, kv_layer->max_seq, 1);

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
        cache->layers[l].seq_len = 0;
    }
}

void den_nvfp4_kv_free(den_nvfp4_kv_cache * cache) {
    if (!cache) return;
    for (int l = 0; l < cache->n_attn_layers; l++) {
        den_nvfp4_kv_layer * layer = &cache->layers[l];
        if (layer->d_k_anchor)    cudaFree(layer->d_k_anchor);
        if (layer->d_v_anchor)    cudaFree(layer->d_v_anchor);
        if (layer->d_k_tiles)     cudaFree(layer->d_k_tiles);
        if (layer->d_v_tiles)     cudaFree(layer->d_v_tiles);
        if (layer->d_scratch_tile)cudaFree(layer->d_scratch_tile);
        if (layer->h_readback)    cudaFreeHost(layer->h_readback);
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
    size_t nvfp4_per_layer = (size_t)(cache->max_seq - 1) * cache->n_kv_heads *
                             DEN_NVFP4_KV_TILE_BYTES +
                             (size_t)cache->n_kv_heads * cache->head_dim * 4;
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
    den_nvfp4_kv_store(&g_nvfp4_kv, layer, seq_pos, d_src, d_src);
    (void)d_dst;
    (void)n_kv_heads;
    (void)head_dim;
}
