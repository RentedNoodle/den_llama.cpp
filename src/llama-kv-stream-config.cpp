#include "llama-kv-stream-config.h"

#include <limits>

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config) {
    if (config.stage_bytes == 0) {
        return { true, false, {} };
    }

    auto invalid = [](const char * error) {
        return llama_kv_stream_config_result { false, false, error };
    };

    if (!config.arch_qwen35) {
        return invalid("block KV streaming currently supports only Qwen3.5");
    }
    if (!config.context_default) {
        return invalid("block KV streaming currently supports only the target context, not MTP/draft contexts");
    }
    if (!config.single_sequence) {
        return invalid("block KV streaming requires exactly one sequence (-np 1)");
    }
    if (!config.flash_attention) {
        return invalid("block KV streaming requires Flash Attention");
    }
    if (!config.kv_offload) {
        return invalid("block KV streaming requires GPU KV offload");
    }
    if (!config.cache_q8_q4) {
        return invalid("block KV streaming currently requires K Q8_0 and V Q4_0");
    }
    if (config.minimum_stage_bytes == 0 || config.stage_bytes < config.minimum_stage_bytes) {
        return invalid("block KV streaming stage is too small for one 256-token cache page");
    }

    return { true, true, {} };
}

llama_kv_stream_pool_layout llama_kv_stream_pool_layout_make(
        const llama_kv_stream_pool_layout_params & params) {
    llama_kv_stream_pool_layout result;

    auto invalid = [&](const char * error) {
        result.error = error;
        return result;
    };

    if (params.pool_bytes == 0 || params.page_bytes == 0 ||
            params.layer_count == 0 || params.scratch_pages == 0) {
        return invalid("pool, page, layer, and scratch counts must be nonzero");
    }
    if (params.page_bytes > std::numeric_limits<uint64_t>::max()/params.scratch_pages) {
        return invalid("scratch byte count overflow");
    }
    result.scratch_bytes = params.page_bytes*params.scratch_pages;
    if (result.scratch_bytes >= params.pool_bytes) {
        return invalid("pool has no resident capacity after reserving scratch pages");
    }
    if (params.page_bytes > std::numeric_limits<uint64_t>::max()/params.layer_count) {
        return invalid("per-layer partition byte count overflow");
    }

    const uint64_t bytes_per_round = params.page_bytes*params.layer_count;
    const uint64_t pages = (params.pool_bytes - result.scratch_bytes)/bytes_per_round;
    if (pages == 0 || pages > std::numeric_limits<uint32_t>::max()/256U) {
        return invalid("pool cannot hold one resident page per layer");
    }

    result.resident_pages_per_layer  = pages;
    result.resident_tokens_per_layer = pages*256U;
    result.resident_bytes = pages*bytes_per_round;
    result.unused_bytes = params.pool_bytes - result.scratch_bytes - result.resident_bytes;
    result.valid = true;
    return result;
}
