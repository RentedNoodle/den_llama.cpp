// llama-den-loader.h — .den file loader for llama.cpp
//
// Opens .den files, reads header, validates magic, parses tensor inventory,
// and converts .den tensor formats to GGML tensor formats.
//
// The .den format uses slot-based tensor indexing (O(1) lookup) with a fixed
// 4096B header, 128B per-tensor index entries, and a tiered data layout.
// NVFP4 weights use 160B NULLGLASS tiles (E2M1 nibbles + per-sub-block scales)
// which are converted to GGML block_nvfp4 blocks on read.
//
// Ported from dengine/src/den_core.c — Project Den
// Canonical format spec: denrt/denrt_format.h

#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// ── .den format constants (from denrt_format.h) ──────────────────────────

#define DEN_MAGIC       0x4E454400  // "DEN\0"
#define DEN_VERSION_V5  0x00050000
#define DEN_VERSION_V1  0x00010000
#define DEN_HEADER_SIZE 4096
#define DEN_TENSOR_ENTRY_SIZE 128
#define DEN_MAX_SLOTS   256

// Architecture enum (subset — full list in denrt_format.h)
enum den_arch_e {
    DEN_ARCH_QWEN35         = 1,
    DEN_ARCH_QWEN36_MOE     = 2,
    DEN_ARCH_QWEN36_DENSE   = 3,
    DEN_ARCH_DEEPSEEK_V4    = 4,
    DEN_ARCH_CUSTOM         = 255,
};

// Hardware target
enum den_hw_target_e {
    DEN_TARGET_NVFP4 = 1,
    DEN_TARGET_BF16  = 2,
    DEN_TARGET_F32   = 3,
    DEN_TARGET_F16   = 4,
    DEN_TARGET_INT8  = 5,
};

// Per-layer stride (32 sub-slots per layer for LLM)
#define DEN_LAYER_STRIDE 32

// Global slots
#define DEN_SLOT_TOKEN_EMBD   0
#define DEN_SLOT_OUTPUT_NORM  1
#define DEN_SLOT_OUTPUT       2

// Layer base — each layer uses 32 consecutive slots
#define DEN_SLOT_LAYER_BASE(layer) (3 + (layer) * DEN_LAYER_STRIDE)

// Per-layer sub-slots (LLM models)
// Attention
#define DEN_SUB_ATTN_QKV      0   // fused QKV (dense models)
#define DEN_SUB_ATTN_Q        1
#define DEN_SUB_ATTN_K        2
#define DEN_SUB_ATTN_V        3
#define DEN_SUB_ATTN_O        4
#define DEN_SUB_ATTN_Q_NORM   5
#define DEN_SUB_ATTN_K_NORM   6
// MoE router (sub-slot 7, only in MoE layers)
#define DEN_SUB_MOE_ROUTER    7
// Dense MLP (sub-slots 8-10)
#define DEN_SUB_MLP_GATE      8
#define DEN_SUB_MLP_UP        9
#define DEN_SUB_MLP_DOWN     10
// MoE expert gate_up (fused gate+up per expert, sub-slot 11)
#define DEN_SUB_MOE_GATE_UP  11
// GDN / SSM (sub-slots 12-20, 24-25)
#define DEN_SUB_GDN_IN_X      12
#define DEN_SUB_GDN_IN_Z      13
#define DEN_SUB_GDN_OUT       14
#define DEN_SUB_GDN_A_LOG     15
#define DEN_SUB_GDN_DT_BIAS   16
#define DEN_SUB_GDN_CONV1D    17
#define DEN_SUB_GDN_D_PROJ    18
#define DEN_SUB_GDN_NORM      19
#define DEN_SUB_GDN_A_NORM    20  // shares sub-slot 20 with INPUT_LAYERNORM
#define DEN_SUB_INPUT_LAYERNORM 20
#define DEN_SUB_POST_ATTN_NORM 21
#define DEN_SUB_PRE_MLP_NORM  22
#define DEN_SUB_POST_MLP_NORM 23
// Sub-slot 24 shared between GDN QKV and MoE gate
#define DEN_SUB_GDN_QKV       24
#define DEN_SUB_MOE_GATE      24
#define DEN_SUB_GDN_PROJ_B    25
// MoE expert down (sub-slot 26)
#define DEN_SUB_MOE_DOWN      26
// Shared expert (sub-slots 27-29)
#define DEN_SUB_SHARED_GATE   27
#define DEN_SUB_SHARED_UP     28
#define DEN_SUB_SHARED_DOWN   29
// Shared expert gate weight (sub-slot 30)
#define DEN_SUB_SHARED_GATE_W 30

// NULLGLASS tile constants
#define DEN_NVFP4_TILE_ELEMS 256
#define DEN_NVFP4_TILE_SIZE  160

// Tile metadata byte offsets (within 160B tile)
#define DEN_TILE_META_OFFSET  128  // first metadata byte (scale data starts here)
#define DEN_TILE_SCALEFMT     148  // scale_format (bits 7-6) + dispatch (bits 4-6) + tile_format (bits 0-3)
#define DEN_TILE_KSTRIDE      149  // K-stride (effective K / 64, 1..4, 0=full)
#define DEN_TILE_PARENT_LO    150  // holographic parent tile offset (low byte)
#define DEN_TILE_PARENT_HI    151  // holographic parent tile offset (high byte)
#define DEN_TILE_EXT_OFFSET   160  // extended 32B format map (tile[160..191])

// Scale formats (bits 7-6 of tile[148])
#define DEN_TILE_SCALE_UE4M3  0x00  // 4-bit UE4M3 LUT scales (8 bytes for 16 sub-blocks)
#define DEN_TILE_SCALE_UE8M0  0x40  // 8-bit UE8M0 exponent scales (16 bytes)
#define DEN_TILE_SCALE_E4M3   0x80  // 8-bit E4M3 FP8 block scales (16 bytes)
#define DEN_TILE_SCALEFMT_MASK 0xC0

// Dispatch codes (bits 4-6 of tile[148])
#define DEN_DISPATCH_OMMA_4X  0x10
#define DEN_DISPATCH_OMMA_1X  0x20
#define DEN_DISPATCH_SW       0x30
#define DEN_DISPATCH_SKIP     0x40
#define DEN_DISPATCH_MASK     0x70

// Tile format (bits 0-3 of tile[148])
#define DEN_TILE_FORMAT_STD   0x00
#define DEN_TILE_FORMAT_SPARSE 0x01
#define DEN_TILE_FORMAT_HOLO  0x02
#define DEN_TILE_FORMAT_HIPREC 0x03
#define DEN_TILE_FORMAT_MASK  0x0F

// WH4 discovery: tile[149] == 8 means WHT-domain NVFP4
#define DEN_KSTRIDE_WH4       8

// ═══════════════════════════════════════════════════════════════════════════
// llama_den_loader — .den file reader for llama.cpp
// ═══════════════════════════════════════════════════════════════════════════

class llama_den_loader {
public:
    llama_den_loader();
    ~llama_den_loader();

    // Disallow copy
    llama_den_loader(const llama_den_loader &) = delete;
    llama_den_loader & operator=(const llama_den_loader &) = delete;

    // Open a .den file. Returns true on success.
    // On failure, call get_error() for a description.
    bool open(const char * path);

    // Number of tensors in the file
    size_t get_tensor_count() const;

    // Tensor name in llama.cpp convention (e.g. "blk.0.attn_q.weight")
    std::string get_tensor_name(size_t i) const;

    // GGML type for this tensor
    ggml_type get_tensor_type(size_t i) const;

    // Logical shape (element counts, not byte sizes)
    std::vector<int64_t> get_tensor_shape(size_t i) const;

    // Byte size of tensor data in GGML format (destination buffer size for read_tensor_data)
    size_t get_tensor_size(size_t i) const;

    // Read tensor data into dst, converting from .den format to GGML format.
    // NVFP4: NULLGLASS 160B tiles → GGML block_nvfp4 blocks
    // BF16:  direct memcpy
    // F32:   direct memcpy
    // dst must be at least get_tensor_size(i) bytes.
    void read_tensor_data(size_t i, void * dst, size_t size) const;

    // Architecture name string (e.g. "qwen35", "qwen36-moe")
    const std::string & get_arch_name() const;

    // Human-readable error description from last failed open()
    const std::string & get_error() const;

    // Close the file and release all memory
    void close();

    // ── Model info accessors (from .den header) ──────────────────────────
    uint32_t get_n_layers()         const;
    uint32_t get_hidden_size()      const;
    uint32_t get_ffn_size()         const;
    uint32_t get_n_heads()          const;
    uint32_t get_n_kv_heads()       const;
    uint32_t get_head_dim()         const;
    uint32_t get_n_rot()            const;
    uint32_t get_vocab_size()       const;
    uint32_t get_n_experts()        const;
    uint32_t get_n_experts_used()   const;
    uint32_t get_ssm_state_size()   const;
    uint32_t get_ssm_conv_kernel()  const;
    uint32_t get_ssm_inner_size()   const;
    uint32_t get_ssm_value_size()   const;
    uint32_t get_full_attn_interval() const;
    float    get_rope_theta()       const;
    float    get_rms_norm_eps()     const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};
