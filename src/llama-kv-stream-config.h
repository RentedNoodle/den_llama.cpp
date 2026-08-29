#pragma once

#include <cstdint>
#include <string>

struct llama_kv_stream_config {
    uint64_t stage_bytes         = 0;
    uint64_t minimum_stage_bytes = 0;

    bool arch_qwen35     = false;
    bool context_default = false;
    bool single_sequence = false;
    bool flash_attention = false;
    bool kv_offload      = false;
    bool cache_q8_q4     = false;
};

struct llama_kv_stream_config_result {
    bool valid   = false;
    bool enabled = false;
    std::string error;
};

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config);

struct llama_kv_stream_pool_layout_params {
    uint64_t pool_bytes = 0;
    uint64_t page_bytes = 0;
    uint32_t layer_count = 0;
    uint32_t scratch_pages = 0;
};

struct llama_kv_stream_pool_layout {
    bool valid = false;
    std::string error;

    uint32_t resident_pages_per_layer  = 0;
    uint32_t resident_tokens_per_layer = 0;
    uint64_t scratch_bytes  = 0;
    uint64_t resident_bytes = 0;
    uint64_t unused_bytes   = 0;
};

llama_kv_stream_pool_layout llama_kv_stream_pool_layout_make(
    const llama_kv_stream_pool_layout_params & params);
