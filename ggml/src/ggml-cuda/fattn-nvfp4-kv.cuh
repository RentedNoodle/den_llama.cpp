// fattn-nvfp4-kv.cuh — NVFP4 KV Cache Quantization for den_llama.cpp
//
// 3.1× compression of K/V cache using 160B NULLGLASS tiles
// (E2M1 nibbles + UE4M3 block scales). Fused attention kernel
// with on-the-fly dequantization.
//
// Ported from Project Den dengine/src/den_kv_cache.cu + den_kv_cache.h
// Architecture: separate-buffer side-channel — canonical KV cache
// stays F32; companion NVFP4 tile buffers allocated by CUDA backend.
//
// Hardware: OMMA.SF.16864 on sm_120a (GB203 RTX 5070 Ti)

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════
// Tile geometry
// ═══════════════════════════════════════════════════════════

#define DEN_NVFP4_KV_TILE_BYTES      160
#define DEN_NVFP4_KV_TILE_ELEMS      256
#define DEN_NVFP4_KV_TILE_SCALES     16
#define DEN_NVFP4_KV_TILE_NIBBLES    128
#define DEN_NVFP4_KV_TILE_GROUPS     16
#define DEN_NVFP4_KV_TILE_GROUP_SZ   16
#define DEN_NVFP4_KV_TILE_NORM_OFF   144
#define DEN_NVFP4_KV_TILE_DISPATCH   148
#define DEN_NVFP4_KV_TILE_KSTRIDE    149

#define DEN_NVFP4_KV_META_SW         0x30
#define DEN_NVFP4_KV_MAX_SEQ         4096
#define DEN_NVFP4_KV_MAX_LAYERS      64

// ═══════════════════════════════════════════════════════════
// Per-layer NVFP4 KV cache storage (GPU resident)
// ═══════════════════════════════════════════════════════════

typedef struct {
    float  * d_k_anchor;     // [n_kv_heads * head_dim] F32 anchor (token 0)
    float  * d_v_anchor;     // [n_kv_heads * head_dim] F32 anchor (token 0)
    uint8_t * d_k_tiles;     // [(max_seq-1) * n_kv_heads * 160] NVFP4 tiles
    uint8_t * d_v_tiles;     // [(max_seq-1) * n_kv_heads * 160] NVFP4 tiles
    uint8_t * d_scratch_tile;// [n_kv_heads * 160]
    float  * h_readback;     // pinned host readback
    int seq_len;
    int max_seq;
    int n_kv_heads;
    int head_dim;
} den_nvfp4_kv_layer;

// ═══════════════════════════════════════════════════════════
// Top-level cache state
// ═══════════════════════════════════════════════════════════

typedef struct {
    den_nvfp4_kv_layer * layers;
    int n_attn_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq;
    int enabled;
    int initialized;
    void * cuda_stream;
} den_nvfp4_kv_cache;

// ═══════════════════════════════════════════════════════════
// Public host API
// ═══════════════════════════════════════════════════════════

int  den_nvfp4_kv_init (den_nvfp4_kv_cache * cache,
                        int n_attn_layers, int n_kv_heads,
                        int head_dim, int max_seq);
int  den_nvfp4_kv_store(den_nvfp4_kv_cache * cache, int layer,
                        int seq_pos, const float * d_k, const float * d_v);
int  den_nvfp4_kv_load (den_nvfp4_kv_cache * cache, int layer,
                        float * d_k_out, float * d_v_out, int seq_pos);
int  den_nvfp4_kv_attention(den_nvfp4_kv_cache * cache, int layer,
                            const float * d_Q, float * d_output, int n_heads);
int  den_nvfp4_kv_seq_len(const den_nvfp4_kv_cache * cache, int layer);
int  den_nvfp4_kv_set_seq_len(den_nvfp4_kv_cache * cache, int layer, int len);
void den_nvfp4_kv_reset_all_seq_len(den_nvfp4_kv_cache * cache);
void den_nvfp4_kv_free  (den_nvfp4_kv_cache * cache);
double den_nvfp4_kv_compression_ratio(const den_nvfp4_kv_cache * cache);

// Check if NVFP4 KV is wanted (env var) and active (initialized)
bool den_nvfp4_kv_is_wanted(void);
bool den_nvfp4_kv_is_active(void);
bool den_nvfp4_kv_has_cache_for(const float * d_kv_tensor);
void den_nvfp4_kv_set_active_cache(den_nvfp4_kv_cache * cache);

// Post-set-rows hook: quantize K/V to NVFP4 tiles after each SET_ROWS op
void den_nvfp4_kv_post_set_rows(const float * d_dst, const float * d_src,
                                int n_kv_heads, int head_dim,
                                int seq_pos, int layer);

// Global instance — extern for access from fattn.cu dispatch
extern den_nvfp4_kv_cache g_nvfp4_kv;

#ifdef __cplusplus
}
#endif
