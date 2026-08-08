// den_format.h — NVFP4 Tile Format + Scale Decode + DEN Native Format Spec
//
// PORTED from C:\Den\den-nvfp4-optimizations\dengine\include\den_format.h
//         + C:\Den\den-nvfp4-optimizations\denrt\denrt_format.h
// Merged into one self-contained header for llama.cpp.
//
// Can be used standalone — no dengine-specific dependencies.
//
// Author: Project Den | Alpha 0.0.1
// Ported: 2026-08-08 → I:\den_llama.cpp\include\den_format.h

#ifndef DEN_FORMAT_H
#define DEN_FORMAT_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// §1 — DEN Native Format Specification (from denrt_format.h)
// ═══════════════════════════════════════════════════════════════════════════════

#define DEN_MAGIC 0x4E454400   // "DEN\0"
#define DEN_VERSION 0x00050000

// Architecture enum
typedef enum {
  DEN_ARCH_QWEN35 = 1,
  DEN_ARCH_QWEN36_MOE = 2,
  DEN_ARCH_QWEN36_DENSE = 3,
  DEN_ARCH_DEEPSEEK_V4 = 4,
  DEN_ARCH_QWEN3_TTS = 5,
  DEN_ARCH_LANCE = 6,
  DEN_ARCH_IMAGE = 7,
  DEN_ARCH_DIT = 8,           // single-stream DiT (Ideogram-4, FLUX)
  DEN_ARCH_MMDIT = 9,         // dual-stream MMDiT (PiD)
  DEN_ARCH_MOT = 10,          // Mixture-of-Transformers (Cosmos3)
  DEN_ARCH_VIDEO_DIT = 11,    // video DiT, dual-checkpoint (Bernini-R / Wan2.2)
  DEN_ARCH_CUSTOM = 255,
} den_arch_t;

// Hardware target
typedef enum {
  DEN_TARGET_NVFP4 = 1,
  DEN_TARGET_BF16 = 2,
  DEN_TARGET_F32 = 3,
  DEN_TARGET_F16 = 4,
  DEN_TARGET_INT8 = 5,
} den_hw_target_t;

// Flags
#define DEN_FLAG_HOT_COLD_LAYOUT (1 << 0)
#define DEN_FLAG_PAGE_ALIGNED (1 << 1)
#define DEN_FLAG_SCALE_INLINE (1 << 2)
#define DEN_FLAG_MMAP_OPTIMIZED (1 << 3)
#define DEN_FLAG_EXPERT_PAGING (1 << 4)
#define DEN_FLAG_HADAMARD_READY (1 << 5)
#define DEN_FLAG_L2_PERSISTENT (1 << 6)
#define DEN_FLAG_IMAGE (1 << 10)
#define DEN_FLAG_DIT_SINGLE_STREAM (1 << 11)  // single-stream DiT (text+image tokens unified)
#define DEN_FLAG_DIT_CFG_DUAL_BRANCH (1 << 12) // dual-branch classifier-free guidance
#define DEN_FLAG_DIT_FLOW_MATCHING (1 << 13)   // flow-matching objective (not DDPM)
#define DEN_FLAG_DIT_VIDEO_TEMPORAL (1 << 14)  // video DiT with temporal attention
#define DEN_FLAG_MOT_DUAL_TOWER (1 << 15)      // Mixture-of-Transformers dual tower
#define DEN_FLAG_EXPERT_DUAL_CHECKPOINT (1 << 16) // dual high-noise/low-noise expert checkpoints

// 256 slot IDs for O(1) lookup
#define DEN_MAX_SLOTS 256
#define DEN_LAYER_STRIDE 32
#define DEN_LAYER_STRIDE_LANCE 64
#define DEN_LAYER_STRIDE_DIT 64       // DiT layers use 64-slot stride
#define DEN_LAYER_STRIDE_MOT 128      // MoT dual-tower: 64 AR + 64 Diffusion

#define DEN_TOWER_OFFSET_REASONER 0   // Cosmos3 AR tower at slot offset 0
#define DEN_TOWER_OFFSET_GENERATOR 64 // Cosmos3 diffusion tower at slot offset 64

#define DEN_SLOT_TOKEN_EMBD 0
#define DEN_SLOT_OUTPUT_NORM 1
#define DEN_SLOT_OUTPUT 2
#define DEN_SLOT_LAYER_BASE(layer) (3 + (layer) * DEN_LAYER_STRIDE)

// Per-layer sub-slots
#define DEN_SLOT_ATTN_QKV 0
#define DEN_SLOT_ATTN_Q 1
#define DEN_SLOT_ATTN_K 2
#define DEN_SLOT_ATTN_V 3
#define DEN_SLOT_ATTN_O 4
#define DEN_SLOT_ATTN_Q_NORM 5
#define DEN_SLOT_ATTN_K_NORM 6
#define DEN_SLOT_MLP_GATE 8
#define DEN_SLOT_MLP_UP 9
#define DEN_SLOT_MLP_DOWN 10
#define DEN_SLOT_GDN_IN_PROJ_X 12
#define DEN_SLOT_GDN_IN_PROJ_Z 13
#define DEN_SLOT_GDN_OUT_PROJ 14
#define DEN_SLOT_GDN_A_LOG 15
#define DEN_SLOT_GDN_DT_BIAS 16
#define DEN_SLOT_GDN_CONV1D 17
#define DEN_SLOT_GDN_D_PROJ 18
#define DEN_SLOT_GDN_NORM 19
// NOTE: Slots 20 and 24 intentionally share sub-slot IDs
#define DEN_SLOT_GDN_A_NORM 20
#define DEN_SLOT_INPUT_LAYERNORM 20  // shares sub-slot 20 with GDN_A_NORM
#define DEN_SLOT_POST_ATTN_NORM 21
#define DEN_SLOT_PRE_MLP_NORM 22
#define DEN_SLOT_POST_MLP_NORM 23
// NOTE: Sub-slot 24 shared between GDN QKV and MoE gate
#define DEN_SLOT_GDN_IN_PROJ_QKV 24
#define DEN_SLOT_MOE_GATE 24
#define DEN_SLOT_GDN_IN_PROJ_B 25

// TTS/ASR audio-specific slots (OmniVoice, Qwen3-TTS)
#define DEN_SLOT_TTS_AUDIO_EMBED        240  // audio codebook embedding table
#define DEN_SLOT_TTS_AUDIO_HEAD         241  // audio prediction head (linear)
#define DEN_SLOT_TTS_CODECBOOK_OFFSETS  242  // codebook layer offsets (metadata)

// DiT (Diffusion Transformer) per-layer sub-slots
#define DEN_SLOT_DIT_TIME_EMBED    32
#define DEN_SLOT_DIT_MODULATION    33
#define DEN_SLOT_DIT_SELF_Q        34
#define DEN_SLOT_DIT_SELF_K        35
#define DEN_SLOT_DIT_SELF_V        36
#define DEN_SLOT_DIT_SELF_O        37
#define DEN_SLOT_DIT_XATTN_Q       38
#define DEN_SLOT_DIT_XATTN_K       39
#define DEN_SLOT_DIT_XATTN_V       40
#define DEN_SLOT_DIT_XATTN_O       41
#define DEN_SLOT_DIT_FFN_GATE      42
#define DEN_SLOT_DIT_FFN_UP        43
#define DEN_SLOT_DIT_FFN_DOWN      44
#define DEN_SLOT_DIT_ADALN_SILU    45
#define DEN_SLOT_DIT_LN1           46
#define DEN_SLOT_DIT_LN2           47
#define DEN_SLOT_DIT_LN3           48
#define DEN_SLOT_DIT_FINAL_NORM    49
#define DEN_SLOT_DIT_PROJ_IN       50
#define DEN_SLOT_DIT_PROJ_OUT      51

// MMDiT (dual-stream) additional sub-slots
#define DEN_SLOT_MMDIT_IMG_QKV     52
#define DEN_SLOT_MMDIT_TXT_QKV     53
#define DEN_SLOT_MMDIT_IMG_NORM    54
#define DEN_SLOT_MMDIT_TXT_NORM    55

// DiT layer base formula
#define DEN_SLOT_DIT_LAYER_BASE(layer) (3 + (layer) * DEN_LAYER_STRIDE_DIT)

// Image data slots
#define DEN_SLOT_IMAGE_TILE_DATA 200
#define DEN_SLOT_IMAGE_THUMBNAIL 201
#define DEN_SLOT_IMAGE_UNDO_STACK 202

// Color space enum
typedef enum {
  DEN_COLOR_SRGB = 0,
  DEN_COLOR_LINEAR = 1,
  DEN_COLOR_DISPLAY_P3 = 2,
} den_color_space_t;

// NVFP4 tile geometry
#define DEN_NVFP4_TILE_ELEMS 256
#define DEN_NVFP4_TILE_SIZE 160

// 4096B header
#define DEN_HEADER_SIZE 4096

// Image metadata sidecar
#define DEN_IMAGE_METADATA_OFFSET 256

typedef struct {
  uint32_t image_width;
  uint32_t image_height;
  uint32_t image_channels;
  uint32_t image_color_space;      // den_color_space_t
  uint32_t image_compression;      // 0=none, 1=RLE, 2=NVENC_H265
  uint32_t image_layer_count;
  uint32_t image_undo_count;
  uint32_t image_tile_size;
  uint64_t image_history_offset;
  uint64_t image_layer_offset;
  uint64_t image_preview_offset;
  uint64_t image_metadata_offset;
} den_image_metadata_t;

typedef struct {
  uint32_t magic;
  uint32_t version;
  uint32_t arch;
  uint32_t flags;

  uint32_t n_layers;
  uint32_t n_heads;
  uint32_t n_kv_heads;
  uint32_t hidden_size;
  uint32_t ffn_size;
  uint32_t vocab_size;
  uint32_t max_seq_len;
  uint32_t n_rot;
  uint32_t n_experts;
  uint32_t n_experts_used;
  float rope_theta;
  float rms_norm_eps;

  uint32_t ssm_state_size;
  uint32_t ssm_conv_kernel;
  uint32_t ssm_inner_size;
  uint32_t ssm_group_count;
  uint32_t ssm_time_step_rank;
  uint32_t full_attention_interval;

  uint32_t mtp_layer_count;
  uint32_t ssm_value_size;
  uint32_t _padding[2];

  uint32_t tensor_count;
  uint32_t index_offset;
  uint64_t data_offset;
  uint64_t total_data_size;

  uint32_t hot_tier_count;
  uint32_t warm_tier_count;
  uint32_t cold_tier_count;
  uint64_t hot_tier_size;
  uint64_t warm_tier_size;
  uint64_t cold_tier_size;

  uint8_t _reserved[4096 - 168];
} den_header_t;

// Image metadata accessor
static inline den_image_metadata_t*
den_get_image_metadata(const den_header_t *hdr) {
  return (den_image_metadata_t*)((const uint8_t*)hdr + DEN_IMAGE_METADATA_OFFSET);
}

// Tensor index entry (128B per tensor)
#define DEN_TENSOR_ENTRY_SIZE 128

#define DEN_TFLAG_HOT_TIER (1 << 0)
#define DEN_TFLAG_WARM_TIER (1 << 1)
#define DEN_TFLAG_COLD_TIER (1 << 2)
#define DEN_TFLAG_EXPERT (1 << 3)
#define DEN_TFLAG_SHARED (1 << 4)
#define DEN_TFLAG_TRANSPOSED (1 << 5)
#define DEN_TFLAG_SCALE_ADJACENT (1 << 6)
#define DEN_TFLAG_HADAMARD (1 << 7)
#define DEN_TFLAG_CHUNK_SKIP (1 << 8)   // skip entire chunk header load for inactive MoE experts
#define DEN_TFLAG_FIREWALL (1 << 10)

typedef struct {
  uint32_t slot;
  uint32_t hw_target;
  uint32_t ndim;
  uint32_t flags;
  int64_t dims[4];
  uint64_t numel;
  uint64_t data_offset;
  uint64_t data_size;
  uint64_t scale_offset;
  uint64_t scale_size;
  uint32_t tile_k;
  uint32_t tile_n;
  uint32_t n_tiles;
  uint32_t scale_count;
  uint64_t norm_offset;
  uint32_t norm_size;
  uint32_t block_size;
  uint32_t grid_size;
  uint32_t smem_bytes;
} den_tensor_entry_t;

static inline int den_validate_header(const den_header_t *hdr) {
  if (hdr->magic != DEN_MAGIC) return -1;
  if (hdr->version != DEN_VERSION && hdr->version != 0x00000001
      && hdr->version != 0x00010000) return -2;
  if (hdr->version != 0x00010000) {
    if (hdr->tensor_count == 0 || hdr->tensor_count > 1024) return -3;
    if (hdr->index_offset != DEN_HEADER_SIZE) return -4;
    if (hdr->n_layers == 0 || hdr->hidden_size == 0) return -5;
  }
  return 0;
}

static inline const den_tensor_entry_t *
den_get_tensor(const den_header_t *hdr, const den_tensor_entry_t *index,
               uint32_t slot) {
  for (uint32_t i = 0; i < hdr->tensor_count; i++)
    if (index[i].slot == slot) return &index[i];
  return NULL;
}

static inline const den_tensor_entry_t *
den_get_layer_tensor(const den_header_t *hdr, const den_tensor_entry_t *index,
                     uint32_t layer, uint32_t sub_slot) {
  return den_get_tensor(hdr, index, DEN_SLOT_LAYER_BASE(layer) + sub_slot);
}

// ═══════════════════════════════════════════════════════════════════════════════
// §2 — NVFP4 tile geometry (NULLGLASS format: E2M1 weights + scales)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Meta-tile: reserved byte layout at tile[148..159] ──
// Byte 148: scale_format(bits 7-6) | dispatch_code(bits 4-6) | tile_format(bits 0-3)
#define DEN_TILE_FORMAT_MASK    0x0F  // bits 0-3
#define DEN_TILE_FORMAT_STD     0x00  // Standard 160B NULLGLASS
#define DEN_TILE_FORMAT_SPARSE  0x01  // Run-length encoded zero runs
#define DEN_TILE_FORMAT_HOLO    0x02  // Holographic (scales from parent)
#define DEN_TILE_FORMAT_HIPREC  0x03  // High-precision FP8

#define DEN_TILE_DISPATCH_MASK  0x70  // bits 4-6: compute path hint
#define DEN_DISPATCH_OMMA_4X    0x10  // OMMA 4X mxf4nvf4 path
#define DEN_DISPATCH_OMMA_1X    0x20  // OMMA 1X mxf8f6f4 path
#define DEN_DISPATCH_SW         0x30  // Software dequant fallback
#define DEN_DISPATCH_SKIP       0x40  // Near-zero tile, skip entirely

#define DEN_TILE_SCALE_FORMAT   0xC0  // bits 7-6: 0=UE4M3, 1=UE8M0, 2=E4M3
#define DEN_TILE_SCALE_UE4M3    0x00  // UE4M3 4-bit LUT scales (Project Den legacy)
#define DEN_TILE_SCALE_UE8M0    0x40  // UE8M0 8-bit exponent scales (OMMA native)
#define DEN_TILE_SCALE_E4M3     0x80  // E4M3 FP8 block scales (NVIDIA NVFP4 standard)

/* Extended op routing (bits 7-4): superset of DISPATCH_MASK. */
#define DEN_TILE_OPMASK         0xF0  // bits 7-4: full tile operation routing
#define DEN_TILE_OP_SKIP        0x20  // skip this tile (no compute, zero contribution)

// E4M3 FP8 decode: 1 sign, 4 exponent, 3 mantissa
static inline float den_e4m3_to_f32(uint8_t x) {
    int s = (x >> 7) & 1;
    int e = (x >> 3) & 0xF;
    int m = x & 0x7;
    if (e == 0) {
        // Subnormal: 2^(-6) * 0.m
        float v = (float)m * 0.0078125f;  // 2^(-7) = 1/128, then * m/2
        return s ? -v : v;
    } else if (e == 15) {
        // NaN/Inf: treat as zero (NVFP4 convention)
        return 0.0f;
    } else {
        // Normal: 2^(e-7) * 1.m
        int exp = e - 7;
        float v = ldexpf((float)(8 + m), exp - 3);  // (8+m)/8 * 2^exp
        return s ? -v : v;
    }
}

// Unified scale decode: reads tile[148] format bits, dispatches to correct decoder
static inline float den_tile_scale_decode(uint8_t scale_byte, int scale_format) {
    if (scale_format == DEN_TILE_SCALE_E4M3) return den_e4m3_to_f32(scale_byte);
    if (scale_format == DEN_TILE_SCALE_UE8M0) {
        // UE8M0: value = 2^(byte - 127)
        if (scale_byte == 0) { union { uint32_t u; float f; } v = { 0x00400000u }; return v.f; }
        union { uint32_t u; float f; } v = { (uint32_t)scale_byte << 23 };
        return v.f;
    }
    // UE4M3 LUT: 0x00-0x0F -> [0, 0.0625, ..., 0.4375]
    //             0x38-0x3F -> [1.0, 1.125, ..., 1.875]
    if (scale_byte <= 0x0F) return (float)scale_byte * 0.0625f;
    return 1.0f + (float)(scale_byte - 0x38) * 0.125f;
}

static inline int den_tile_scale_fmt(const uint8_t *t) { return (t)[148] & DEN_TILE_SCALE_FORMAT; }
#define DEN_TILE_SCALE_IS_E4M3(t)  (den_tile_scale_fmt(t) == DEN_TILE_SCALE_E4M3)
#define DEN_TILE_SCALE_IS_UE8M0(t) (den_tile_scale_fmt(t) == DEN_TILE_SCALE_UE8M0)

// Byte 149: K-stride (effective K / 64, 1..4, 0=full)
#define DEN_TILE_KSTRIDE_MASK   0xFF
#define DEN_KSTRIDE_OFFSET      149

// Bytes 150-151: scale parent tile offset (for DEN_TILE_FORMAT_HOLO)
#define DEN_HOLO_PARENT_OFFSET  150

// NVFP4 tile header macro: extract fields from raw tile bytes
#define DEN_TILE_FORMAT(t)      ((t[148]) & DEN_TILE_FORMAT_MASK)
#define DEN_TILE_DISPATCH(t)    ((t[148]) & DEN_TILE_DISPATCH_MASK)
#define DEN_TILE_KSTRIDE(t)     ((t[149]) ? (t[149]) : 4)  // 0->full
#define DEN_TILE_HOLO_PARENT(t) (*(uint16_t*)((t) + DEN_HOLO_PARENT_OFFSET))

// ── Self-describing tile format extension (tile[160..191]) ──────────────

#define TILE_FORMAT_MAP_OFF   160  // 32-byte format nibble map at tile[160..191]
#define TILE_FORMAT_VERSION   191  // format version byte at end of map
#define TILE_FORMAT_CURRENT  0x01  // current format map version

// Per-sub-block format codes (4-bit nibble values)
#define FMT_SKIP    0x0  // skip (no compute, output zero)
#define FMT_E2M1    0x1  // NVFP4 E2M1 + UE4M3 scale (standard)
#define FMT_FP8     0x2  // FP8 E4M3 (8-bit float)
#define FMT_FP16    0x3  // FP16 (16-bit float)
#define FMT_ZERO    0x4  // zero sub-block (no data, output zero)

static inline int tile_get_sub_block_format(const unsigned char *tile, int sub_block) {
    if (tile[TILE_FORMAT_VERSION] != TILE_FORMAT_CURRENT)
        return FMT_E2M1;  // legacy tile, default to E2M1
    int byte_idx = TILE_FORMAT_MAP_OFF + (sub_block >> 1);
    return (sub_block & 1) ? (tile[byte_idx] >> 4) : (tile[byte_idx] & 0x0F);
}

// ═══════════════════════════════════════════════════════════════════════════════
// §3 — Format-encoded virtual address dispatch
// CUDA 64-bit device addresses only use 48 bits (bits 0-47).
// Bits 48-55 encode format code (256 combinations).
// Bits 56-63 spare (dimension tag, reserved).
// ═══════════════════════════════════════════════════════════════════════════════

#define PTR_FORMAT_MASK      0x00FF000000000000ULL  // bits 48-55
#define PTR_FORMAT_SHIFT     48

// Format codes for weight tensor virtual address dispatch
#define VA_FMT_BF16       0x01
#define VA_FMT_NVFP4      0x02
#define VA_FMT_WH4        0x03
#define VA_FMT_INT8       0x04
#define VA_FMT_FP8        0x05

// Encode format code into upper bits of a device pointer.
static inline void *ptr_encode_format(void *ptr, int format) {
    uint64_t p = (uint64_t)ptr;
    p &= ~PTR_FORMAT_MASK;
    p |= ((uint64_t)format & 0xFF) << PTR_FORMAT_SHIFT;
    return (void*)p;
}

// Decode format code from a device pointer.
static inline int ptr_decode_format(const void *ptr) {
    return (int)(((uint64_t)ptr & PTR_FORMAT_MASK) >> PTR_FORMAT_SHIFT);
}

// Strip format bits to recover the real 48-bit device address.
static inline void *ptr_strip_format(const void *ptr) {
    return (void*)((uint64_t)ptr & 0x0000FFFFFFFFFFFFULL);
}

#ifdef __cplusplus
}
#endif

#endif // DEN_FORMAT_H
