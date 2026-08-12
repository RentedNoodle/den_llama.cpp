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

#define DEN_NVFP4_KV_TILE_BYTES      160   // K4V4: 16 scales + 128 nibbles + 4 norm + 2 meta + 10 pad
#define DEN_NVFP4_KV_TILE_BYTES_K8   288   // K8V8: 16 scales + 256 uint8 + 4 norm + 2 meta + 10 pad
#define DEN_NVFP4_KV_TILE_ELEMS      256
#define DEN_NVFP4_KV_TILE_SCALES     16
#define DEN_NVFP4_KV_TILE_NIBBLES    128
#define DEN_NVFP4_KV_TILE_UINT8      256   // K8 element data size
#define DEN_NVFP4_KV_TILE_GROUPS     16
#define DEN_NVFP4_KV_TILE_GROUP_SZ   16
#define DEN_NVFP4_KV_TILE_NORM_OFF   144
#define DEN_NVFP4_KV_TILE_NORM_OFF_K8 272
#define DEN_NVFP4_KV_TILE_DISPATCH   148
#define DEN_NVFP4_KV_TILE_DISPATCH_K8 276
#define DEN_NVFP4_KV_TILE_KSTRIDE    149
#define DEN_NVFP4_KV_TILE_KSTRIDE_K8 277

#define DEN_NVFP4_KV_META_SW         0x30  // K4V4: both 4-bit E2M1 nibbles
#define DEN_NVFP4_KV_META_K8V8       0x31  // K8V8: keys 8-bit uint8, values 8-bit uint8
// PRECISION TAIL: the LATEST DEN_NVFP4_KV_TAIL_TOKENS cache positions are kept at
// F32 (exact); tokens older than the tail window are quantized to NVFP4 tiles.
// Attention weights recent tokens most, so an exact recent window is near-lossless.
// Default 256 is testable on 16 GB; override at runtime via env DEN_NVFP4_KV_TAIL.
// The previous design kept only the first 4 tokens ("KVSink anchors") exact — that
// was 256x too small AND conceptually inverted (attention barely weights the oldest
// 4 tokens). This reverses the concept: exact window now slides at the FRONT.
#define DEN_NVFP4_KV_TAIL_TOKENS    256    // default precision-tail size (F32), runtime-overridable
#define DEN_NVFP4_KV_TAIL_ENV       "DEN_NVFP4_KV_TAIL"
#define DEN_NVFP4_KV_MAX_SEQ         4096
#define DEN_NVFP4_KV_MAX_LAYERS      64

// ═══════════════════════════════════════════════════════════
// Per-layer NVFP4 KV cache storage (GPU resident)
// ═══════════════════════════════════════════════════════════

typedef struct {
    float  * d_k_tail;       // [tail_tokens * n_kv_heads * head_dim] F32 — latest tail_tokens tokens (PRECISION TAIL)
    float  * d_v_tail;       // [tail_tokens * n_kv_heads * head_dim] F32 — latest tail_tokens tokens (PRECISION TAIL)
    uint8_t * d_k_tiles;     // [(max_seq - tail_tokens) * n_kv_heads * 160] NVFP4 tiles (tokens OLDER than the tail)
    uint8_t * d_v_tiles;     // [(max_seq - tail_tokens) * n_kv_heads * 160] NVFP4 tiles
    uint8_t * d_scratch_tile;// [n_kv_heads * 160]
    // B2 GQA dequant cache: the SAME K/V tile bytes are dequantized by gqa_ratio
    // query-head blocks independently (one block per query head). These mirrors
    // hold the exact FP32 dequantized tile values (populated once at store/evict
    // time), so all gqa_ratio query-head blocks read FP32 from L2 instead of
    // re-running the per-element tile dequant. Size: tile_count * n_kv_heads * head_dim.
    float  * d_k_dequant;    // [(max_seq - tail_tokens) * n_kv_heads * head_dim] F32 — dequantized K mirror
    float  * d_v_dequant;    // [(max_seq - tail_tokens) * n_kv_heads * head_dim] F32 — dequantized V mirror
    float  * h_readback;     // pinned host readback
    int seq_len;
    int max_seq;
    int tail_tokens;         // precision-tail size (latest N tokens kept F32); 0 => cache disabled
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
    int tail_tokens;       // precision-tail size (resolved: default or DEN_NVFP4_KV_TAIL env)
    int enabled;
    int initialized;
    int thrift_attention; // K8V8: keys at 8-bit, values at 8-bit
    void * cuda_stream;
} den_nvfp4_kv_cache;

// ═══════════════════════════════════════════════════════════
// Public host API
// ═══════════════════════════════════════════════════════════

int  den_nvfp4_kv_init (den_nvfp4_kv_cache * cache,
                        int n_attn_layers, int n_kv_heads,
                        int head_dim, int max_seq,
                        int thrift_attention);
int  den_nvfp4_kv_store(den_nvfp4_kv_cache * cache, int layer,
                        int seq_pos, const float * d_k, const float * d_v,
                        cudaStream_t main_stream);
int  den_nvfp4_kv_load (den_nvfp4_kv_cache * cache, int layer,
                        float * d_k_out, float * d_v_out, int seq_pos,
                        cudaStream_t main_stream);
int  den_nvfp4_kv_attention(den_nvfp4_kv_cache * cache, int layer,
                            const float * d_Q, float * d_output, int n_heads,
                            cudaStream_t main_stream);
int  den_nvfp4_kv_seq_len(const den_nvfp4_kv_cache * cache, int layer);
int  den_nvfp4_kv_set_seq_len(den_nvfp4_kv_cache * cache, int layer, int len);
void den_nvfp4_kv_reset_all_seq_len(den_nvfp4_kv_cache * cache);
void den_nvfp4_kv_free  (den_nvfp4_kv_cache * cache);
double den_nvfp4_kv_compression_ratio(const den_nvfp4_kv_cache * cache);

// Scale-distribution probe (DEN_NVFP4_KV_HIST=1): begin/reset + dump histogram
void den_nvfp4_kv_hist_begin(void);
void den_nvfp4_kv_hist_dump(void);

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
