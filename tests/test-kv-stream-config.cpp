#include "llama-kv-stream-config.h"
#include "testing.h"

int main() {
    testing t;

    t.test("streaming is opt-in", [](testing & t) {
        llama_kv_stream_config config;
        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("disabled config is valid", result.valid);
        t.assert_true("disabled config remains disabled", !result.enabled);
    });

    t.test("supported target configuration is accepted", [](testing & t) {
        llama_kv_stream_config config;
        config.stage_bytes        = 64ULL*1024ULL*1024ULL;
        config.minimum_stage_bytes = 1664ULL*256ULL;
        config.arch_qwen35        = true;
        config.context_default    = true;
        config.single_sequence    = true;
        config.flash_attention    = true;
        config.kv_offload         = true;
        config.cache_q8_q4        = true;

        const auto result = llama_kv_stream_config_validate(config);
        t.assert_true("config is valid", result.valid);
        t.assert_true("config is enabled", result.enabled);
    });

    t.test("each unsupported condition fails loudly", [](testing & t) {
        llama_kv_stream_config base;
        base.stage_bytes         = 64ULL*1024ULL*1024ULL;
        base.minimum_stage_bytes = 1664ULL*256ULL;
        base.arch_qwen35         = true;
        base.context_default     = true;
        base.single_sequence     = true;
        base.flash_attention     = true;
        base.kv_offload          = true;
        base.cache_q8_q4         = true;

        auto expect_invalid = [&](const char * name, const llama_kv_stream_config & config) {
            const auto result = llama_kv_stream_config_validate(config);
            t.assert_true(name, !result.valid && !result.enabled && !result.error.empty());
        };

        auto config = base;
        config.arch_qwen35 = false;
        expect_invalid("non-Qwen architecture", config);
        config = base;
        config.context_default = false;
        expect_invalid("draft/MTP context", config);
        config = base;
        config.single_sequence = false;
        expect_invalid("parallel sequences", config);
        config = base;
        config.flash_attention = false;
        expect_invalid("Flash Attention disabled", config);
        config = base;
        config.kv_offload = false;
        expect_invalid("KV offload disabled", config);
        config = base;
        config.cache_q8_q4 = false;
        expect_invalid("unsupported cache types", config);
        config = base;
        config.stage_bytes = config.minimum_stage_bytes - 1;
        expect_invalid("stage smaller than one page", config);
    });

    t.test("pool is partitioned evenly across layers with one scratch page", [](testing & t) {
        const auto layout = llama_kv_stream_pool_layout_make({
            /*.pool_bytes   =*/ 64ULL*1024ULL*1024ULL,
            /*.page_bytes   =*/ 1664ULL*256ULL,
            /*.layer_count  =*/ 16,
            /*.scratch_pages=*/ 1,
        });

        t.assert_true("layout is valid", layout.valid);
        t.assert_equal(uint32_t(9), layout.resident_pages_per_layer);
        t.assert_equal(uint32_t(9*256), layout.resident_tokens_per_layer);
        t.assert_equal(1664ULL*256ULL, layout.scratch_bytes);
        t.assert_equal(
            64ULL*1024ULL*1024ULL,
            layout.scratch_bytes + layout.resident_bytes + layout.unused_bytes);
    });

    t.test("pool rejects missing scratch or resident capacity", [](testing & t) {
        auto layout = llama_kv_stream_pool_layout_make({ 0, 1664ULL*256ULL, 16, 1 });
        t.assert_true("zero pool", !layout.valid);

        layout = llama_kv_stream_pool_layout_make({ 1664ULL*256ULL, 1664ULL*256ULL, 16, 1 });
        t.assert_true("scratch-only pool", !layout.valid);

        layout = llama_kv_stream_pool_layout_make({ 64ULL*1024ULL*1024ULL, 0, 16, 1 });
        t.assert_true("zero page", !layout.valid);
    });

    return t.summary();
}
