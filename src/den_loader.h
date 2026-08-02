// den_loader.h -- .den native format parser for den_llama.cpp
//
// Ported from dengine (src/den_core.c + denrt/denrt_format.h). Reads the
// slot-based .den header and tensor index and exposes:
//   * parsed header fields
//   * tensor entries (slot / hw_target / dims / data offset+size)
//   * slot -> llama.cpp qwen35 tensor name mapping
//   * hparam derivation (ssm value-head count / value dim / attention head dim)
//
// The .den format: 4096-byte header, then 128-byte slot entries, then packed
// tensor data. Slots are keyed as . No GGUF, no KV.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace denrt {

constexpr uint32_t DEN_MAGIC       = 0x4E454400; // "DEN\0"
constexpr uint32_t DEN_HEADER_SIZE = 4096;
constexpr uint32_t DEN_ENTRY_SIZE  = 128;
constexpr uint32_t DEN_LAYER_STRIDE = 32;

// Hardware targets (denrt_format.h)
enum den_hw : uint32_t {
    DEN_HW_NVFP4 = 1,
    DEN_HW_BF16  = 2,
    DEN_HW_F32   = 3,
    DEN_HW_F16   = 4,
    DEN_HW_INT8  = 5,
};

// Parsed 4096-byte header (field offsets per denrt_format.h, all LE)
struct den_header {
    uint32_t magic    = 0;
    uint32_t version  = 0;
    uint32_t arch     = 0;
    uint32_t flags    = 0;
    uint32_t n_layers = 0;
    uint32_t n_heads  = 0;
    uint32_t n_kv_heads = 0;
    uint32_t hidden_size = 0;
    uint32_t ffn_size = 0;
    uint32_t vocab_size = 0;
    uint32_t max_seq_len = 0;
    uint32_t n_rot    = 0;
    uint32_t n_experts = 0;
    uint32_t n_experts_used = 0;
    float    rope_theta = 0.0f;
    float    rms_norm_eps = 1e-6f;
    uint32_t ssm_state_size = 0;      // key head dim
    uint32_t ssm_conv_kernel = 0;     // conv kernel
    uint32_t ssm_inner_size = 0;      // key_dim (dengine convention)
    uint32_t ssm_group_count = 0;     // num key heads
    uint32_t ssm_time_step_rank = 0;  // (dengine writes num-key-heads here for v1)
    uint32_t full_attention_interval = 0;
    uint32_t mtp_layer_count = 0;
    uint32_t ssm_value_size = 0;      // value head dim (or 0)
    uint32_t tensor_count = 0;
    uint32_t index_offset = 0;
    uint64_t data_offset = 0;
    uint64_t total_data_size = 0;
};

// 128-byte tensor index entry
struct den_tensor {
    uint32_t slot     = 0;
    uint32_t hw_target = 0;
    uint32_t ndim     = 0;
    uint32_t flags    = 0;
    int64_t  dims[4]  = {0, 0, 0, 0};
    uint64_t numel    = 0;
    uint64_t data_offset = 0; // relative to header.data_offset
    uint64_t data_size = 0;   // bytes in file
};

// Parse the 4096-byte header. Returns false on bad magic/version.
bool parse_header(const uint8_t * bytes, den_header * out);

// Parse n tensor entries from the index region.
std::vector<den_tensor> parse_entries(const uint8_t * bytes, size_t n);

// Sniff DEN_MAGIC at the start of the file.
bool is_den_file(const char * path);

// Read header + entries from a .den file. Returns false on error.
bool load(const char * path, den_header * hdr, std::vector<den_tensor> * entries);

// Map a slot to the llama.cpp qwen35 tensor name. Returns empty for unknown slots.
std::string slot_to_name(uint32_t slot, uint32_t n_layers);

// Derive llama.cpp hparams that the .den header does not store directly:
//   *ssm_dt_rank  = number of value heads (from ssm_dt/A_log or in_proj_a dims)
//   *ssm_d_inner  = value_dim (from ssm_out / attn_gate dims)
//   *head_dim     = attention head dim (from attn_q dims)
// Falls back to header-derived defaults if shapes are absent.
void derive_hparams(const den_header & hdr, const std::vector<den_tensor> & entries,
                    uint32_t * ssm_dt_rank, uint32_t * ssm_d_inner, uint32_t * head_dim);

} // namespace denrt
