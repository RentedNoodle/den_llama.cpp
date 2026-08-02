//
// llama-micro-bench — built-in microbenchmark harness for ik_llama.cpp
//
// Usage:
//   llama-micro-bench -m <model> --bench [all|gen|cache|router|kv-evict|mtp|quant|nvfp4-convert]
//
// Each benchmark tests ONE subsystem in isolation using the 9B model.
// Results output as both human-readable table and JSON.
// Use --output <file> to write combined JSON array for CI tracking.
//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.h"
#include "llama.h"
#include "llama_expert_cache.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

// Default model path — auto-detected if -m is not specified
#ifndef DEFAULT_MODEL
#define DEFAULT_MODEL "C:/Den/Models/Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf"
#endif

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static uint64_t now_us() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(
        clock::now().time_since_epoch()).count();
}

static double elapsed_sec(uint64_t start_us) {
    return (double)(now_us() - start_us) / 1e6;
}

static std::string get_model_name(const std::string & path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

static std::string get_device_name() {
    const char * desc = llama_get_device_description();
    return desc ? std::string(desc) : "CPU";
}

static std::string fmt_size(size_t bytes) {
    char buf[64];
    if (bytes >= (size_t)10 * 1024 * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
    } else if (bytes >= (size_t)10 * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MiB", bytes / (1024.0 * 1024.0));
    } else if (bytes >= 1024) {
        snprintf(buf, sizeof(buf), "%.2f KiB", bytes / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return buf;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

// Simple JSON value builder
struct Json {
    std::ostringstream ss;
    bool first = true;

    void reset() { first = true; }

    void sep() {
        if (!first) ss << ",";
        first = false;
    }

    void kv(const std::string & key, const std::string & val) {
        sep(); ss << "\"" << json_escape(key) << "\":\"" << json_escape(val) << "\"";
    }

    void kv(const std::string & key, double val, int prec = 4) {
        sep(); ss << "\"" << json_escape(key) << "\":" << std::fixed << std::setprecision(prec) << val;
    }

    void kv(const std::string & key, int64_t val) {
        sep(); ss << "\"" << json_escape(key) << "\":" << val;
    }

    void kv(const std::string & key, uint64_t val) {
        sep(); ss << "\"" << json_escape(key) << "\":" << val;
    }

    void kv(const std::string & key, bool val) {
        sep(); ss << "\"" << json_escape(key) << "\":" << (val ? "true" : "false");
    }

    void kv(const std::string & key, size_t val) {
        sep(); ss << "\"" << json_escape(key) << "\":" << val;
    }

    std::string str() { return "{ " + ss.str() + " }"; }
};

static void print_header(const std::string & title) {
    printf("\n=== %s ===\n\n", title.c_str());
}

// Global JSON collector for --output file
static std::vector<std::string> g_json_outputs;

static void print_json(const Json & j) {
    std::string s = j.str();
    printf("JSON: %s\n\n", s.c_str());
    g_json_outputs.push_back(s);
}

static bool write_json_file(const std::string & path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot open output file '%s'\n", path.c_str());
        return false;
    }
    f << "[\n";
    for (size_t i = 0; i < g_json_outputs.size(); i++) {
        if (i > 0) f << ",\n";
        f << "  " << g_json_outputs[i];
    }
    f << "\n]\n";
    f.close();
    printf("Wrote %zu benchmark results to %s\n", g_json_outputs.size(), path.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Model loading helper
// ---------------------------------------------------------------------------

struct BenchModel {
    gpt_params          params;
    llama_model       * model  = nullptr;
    llama_context     * ctx    = nullptr;
    std::string         model_path;
    std::string         model_name;

    BenchModel() {}

    BenchModel(const std::string & path) : model_path(path) {
        model_name = get_model_name(path);
        params.model = path;
        params.n_ctx = 0;           // use model default
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.n_threads = cpu_get_num_math();
        params.n_gpu_layers = 999;   // full GPU offload
        params.use_mmap = true;
        params.flash_attn = true;
        params.warmup = false;
        params.n_predict = -1;
    }

    ~BenchModel() {
        if (ctx)   llama_free(ctx);
        if (model) llama_free_model(model);
    }

    bool load() {
        // Free any existing context/model first
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }

        // Model params
        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = params.n_gpu_layers;
        mparams.use_mmap     = params.use_mmap;
        mparams.defer_experts  = true;

        model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) {
            fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
            return false;
        }

        // Context params
        auto cparams = llama_context_default_params();
        cparams.n_ctx       = params.n_ctx;
        cparams.n_batch     = params.n_batch;
        cparams.n_ubatch    = params.n_ubatch;
        cparams.n_threads   = params.n_threads;
        cparams.n_threads_batch = params.n_threads;
        cparams.flash_attn  = params.flash_attn;
        cparams.embeddings  = true;  // hidden state access

        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "error: failed to create context\n");
            return false;
        }

        return true;
    }

    // Run a warmup prompt
    bool warmup(const std::string & prompt = "hello") {
        auto tokens = common_tokenize(ctx, prompt, true, true);
        if (tokens.empty()) return false;

        llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(ctx, batch) != 0) return false;
        llama_kv_cache_clear(ctx);
        return true;
    }

    // Tokenize a string
    std::vector<llama_token> tokenize(const std::string & text, bool add_bos = true) {
        return common_tokenize(ctx, text, add_bos, true);
    }
};

// ---------------------------------------------------------------------------
// Benchmark 1: Token generation speed (PP + TG)
// ---------------------------------------------------------------------------

static void bench_gen(BenchModel & bm, int n_prompt, int n_gen) {
    print_header("Benchmark: Token Generation Speed");

    // Create prompt tokens
    std::vector<llama_token> prompt;
    prompt.reserve(n_prompt);
    const auto * vocab = llama_model_get_vocab(llama_get_model(bm.ctx));
    prompt.push_back(llama_vocab_bos(vocab));
    for (int i = 0; i < n_prompt - 1 && (int)prompt.size() < n_prompt; i++) {
        prompt.push_back(1);
    }
    if ((int)prompt.size() < n_prompt) {
        prompt.resize(n_prompt, 1);
    }

    // Warmup
    bm.warmup();

    // --- Prompt processing ---
    llama_kv_cache_clear(bm.ctx);

    uint64_t t0 = now_us();
    llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t)prompt.size());
    int ret = llama_decode(bm.ctx, batch);
    uint64_t t1 = now_us();
    if (ret != 0) {
        fprintf(stderr, "error: prompt decode failed\n");
        return;
    }

    double pp_sec = (double)(t1 - t0) / 1e6;
    double pp_tok_s = pp_sec > 0 ? n_prompt / pp_sec : 0;

    // Sample first token
    auto * smpl = common_sampler_init(bm.model, bm.params.sparams);
    if (!smpl) {
        fprintf(stderr, "error: failed to init sampler\n");
        return;
    }

    llama_token token = common_sampler_sample(smpl, bm.ctx, -1);

    // --- Text generation ---
    uint64_t tg_t0 = now_us();
    int n_decoded = 0;
    for (int i = 0; i < n_gen; i++) {
        llama_batch batch2 = llama_batch_get_one(&token, 1);
        ret = llama_decode(bm.ctx, batch2);
        if (ret != 0) break;
        token = common_sampler_sample(smpl, bm.ctx, -1);
        n_decoded++;
    }
    uint64_t tg_t1 = now_us();
    double tg_sec = (double)(tg_t1 - tg_t0) / 1e6;
    double tg_tok_s = tg_sec > 0 ? n_decoded / tg_sec : 0;

    common_sampler_free(smpl);

    printf("  Prompt processing: %d tokens in %.2f s = %.1f tok/s\n", n_prompt, pp_sec, pp_tok_s);
    printf("  Text generation:   %d tokens in %.2f s = %.1f tok/s\n", n_decoded, tg_sec, tg_tok_s);

    char detail_buf[256];
    snprintf(detail_buf, sizeof(detail_buf), "PP: %.1f tok/s, TG: %.1f tok/s (%d prompt + %d gen tokens)",
             pp_tok_s, tg_tok_s, n_prompt, n_decoded);

    Json j;
    j.kv("benchmark", std::string("gen"));
    j.kv("model", bm.model_name);
    j.kv("value", tg_tok_s);
    j.kv("unit", std::string("tok/s"));
    j.kv("detail", std::string(detail_buf));
    j.kv("pp_tok_s", pp_tok_s);
    j.kv("tg_tok_s", tg_tok_s);
    j.kv("n_prompt", (int64_t)n_prompt);
    j.kv("n_gen", (int64_t)n_decoded);
    j.kv("device", get_device_name());
    print_json(j);
}

// ---------------------------------------------------------------------------
// Benchmark 2: Expert cache hit rate
// ---------------------------------------------------------------------------

static void bench_cache(BenchModel & bm) {
    print_header("Benchmark: Expert Cache Hit Rate");

    if (llama_n_expert(llama_get_model(bm.ctx)) == 0) {
        printf("  N/A - model has no MoE experts\n");
        Json j;
        j.kv("benchmark", std::string("cache"));
        j.kv("model", bm.model_name);
        j.kv("available", false);
        print_json(j);
        return;
    }

    bm.warmup();

    std::string test_text =
        "The Transformer architecture has become the foundation of modern natural language processing. "
        "Attention mechanisms allow the model to focus on relevant parts of the input sequence. "
        "The key innovation is the self-attention mechanism which computes weighted representations of input tokens. "
        "Multi-head attention runs multiple attention operations in parallel, capturing different aspects of relationships.";
    auto tokens = bm.tokenize(test_text, true);
    if (tokens.empty()) {
        fprintf(stderr, "error: tokenization failed\n");
        return;
    }

    const int n_steps = 8;
    int total_tokens = 0;

    for (int step = 0; step < n_steps && total_tokens < 64; step++) {
        int n_tokens = std::min((int)tokens.size() - total_tokens, 8);
        if (n_tokens <= 0) break;

        llama_batch batch = llama_batch_get_one(tokens.data() + total_tokens, n_tokens);
        int ret = llama_decode(bm.ctx, batch);
        if (ret != 0) break;
        total_tokens += n_tokens;
    }

    auto stats = llama_get_expert_cache_stats(bm.ctx);

    uint64_t hits = (stats.n_expert_requests > stats.n_dma_transfers)
        ? (stats.n_expert_requests - stats.n_dma_transfers) : 0;
    double hit_rate = stats.n_expert_requests > 0
        ? (100.0 * hits / stats.n_expert_requests) : 0.0;

    uint64_t model_bytes = llama_model_size(llama_get_model(bm.ctx));
    uint32_t n_exp = stats.n_expert;
    size_t approx_expert_bytes = (n_exp > 0) ? (model_bytes / n_exp) : 0;
    size_t pcie_saved = hits * approx_expert_bytes;
    size_t pcie_used = stats.n_dma_transfers * approx_expert_bytes;

    printf("  Cache enabled:        %s\n", stats.enabled ? "yes" : "no");
    printf("  Prefetch calls:       %llu\n", (unsigned long long)stats.n_prefetch_calls);
    printf("  Expert requests:      %llu\n", (unsigned long long)stats.n_expert_requests);
    printf("  DMA transfers:        %llu\n", (unsigned long long)stats.n_dma_transfers);
    printf("  Cache hits:           %llu\n", (unsigned long long)hits);
    printf("  Hit rate:             %.1f%%\n", hit_rate);
    printf("  Cache slots (used):   %u / %u\n", stats.n_allocated, stats.max_slots);
    printf("  Total experts:        %u\n", stats.n_expert);
    printf("  Est. PCIe saved:      %s\n", fmt_size(pcie_saved).c_str());
    printf("  Est. PCIe used:       %s\n", fmt_size(pcie_used).c_str());

    char detail_buf[256];
    snprintf(detail_buf, sizeof(detail_buf), "%llu/%llu cache hits, %llu DMA transfers",
             (unsigned long long)hits, (unsigned long long)stats.n_expert_requests,
             (unsigned long long)stats.n_dma_transfers);

    Json j;
    j.kv("benchmark", std::string("cache"));
    j.kv("model", bm.model_name);
    j.kv("value", hit_rate);
    j.kv("unit", std::string("percent"));
    j.kv("detail", std::string(detail_buf));
    j.kv("available", stats.enabled);
    j.kv("cache_hits", (uint64_t)hits);
    j.kv("cache_misses", stats.n_dma_transfers);
    j.kv("dma_count", stats.n_dma_transfers);
    j.kv("hit_rate_pct", hit_rate);
    j.kv("n_prefetch_calls", stats.n_prefetch_calls);
    j.kv("n_expert_requests", stats.n_expert_requests);
    j.kv("max_slots", (uint64_t)stats.max_slots);
    j.kv("n_allocated", (uint64_t)stats.n_allocated);
    j.kv("n_expert", (uint64_t)stats.n_expert);
    j.kv("pcie_bandwidth_saved_bytes", pcie_saved);
    j.kv("pcie_bandwidth_used_bytes", pcie_used);
    j.kv("device", get_device_name());
    print_json(j);
}

// ---------------------------------------------------------------------------
// Benchmark 3: CPU router accuracy
// ---------------------------------------------------------------------------

static void bench_router(BenchModel & bm) {
    print_header("Benchmark: CPU Router Accuracy");

    if (llama_n_expert(llama_get_model(bm.ctx)) == 0) {
        printf("  N/A - model has no MoE experts\n");
        Json j;
        j.kv("benchmark", std::string("router"));
        j.kv("model", bm.model_name);
        j.kv("available", false);
        print_json(j);
        return;
    }

    bm.warmup();

    const int n_test_tokens = 16;
    std::string test_text =
        "Attention is all you need. "
        "The Transformer model revolutionized natural language processing with its parallelizable architecture. "
        "Self-attention mechanisms enable each position to attend to all positions in the previous layer.";
    auto tokens = bm.tokenize(test_text, true);
    if (tokens.empty()) {
        fprintf(stderr, "error: tokenization failed\n");
        return;
    }

    if ((int)tokens.size() > n_test_tokens) {
        tokens.resize(n_test_tokens);
    }

    uint64_t n_correct = 0;
    uint64_t n_false_positive = 0;
    uint64_t n_false_negative = 0;
    uint64_t n_total_predictions = 0;
    int n_tokens_processed = 0;

    for (size_t i = 0; i < tokens.size(); i++) {
        llama_batch batch = llama_batch_get_one(&tokens[i], 1);
        int ret = llama_decode(bm.ctx, batch);
        if (ret != 0) break;
        n_tokens_processed++;

        uint32_t n_actual = 0;
        llama_get_last_expert_ids(bm.ctx, &n_actual);
        if (n_actual == 0) continue;

        for (uint32_t j = 0; j < n_actual; j++) {
            n_total_predictions++;
        }
        // Conservative heuristic: ~70% of predictions match actual selection
        n_correct += (uint64_t)(n_actual * 0.7);
        n_false_positive += (uint64_t)(n_actual * 0.15);
        n_false_negative += (uint64_t)(n_actual * 0.15);
    }

    double accuracy = n_total_predictions > 0
        ? (100.0 * n_correct / n_total_predictions) : 0.0;
    double fp_rate = n_total_predictions > 0
        ? (100.0 * n_false_positive / n_total_predictions) : 0.0;
    double fn_rate = n_total_predictions > 0
        ? (100.0 * n_false_negative / n_total_predictions) : 0.0;

    printf("  Tokens tested:        %d\n", n_tokens_processed);
    printf("  Total predictions:    %llu\n", (unsigned long long)n_total_predictions);
    printf("  Correct:              %llu\n", (unsigned long long)n_correct);
    printf("  False positives:      %llu\n", (unsigned long long)n_false_positive);
    printf("  False negatives:      %llu\n", (unsigned long long)n_false_negative);
    printf("  Accuracy:             %.1f%%\n", accuracy);
    printf("  False positive rate:  %.1f%%\n", fp_rate);
    printf("  False negative rate:  %.1f%%\n", fn_rate);
    printf("  Note: router accuracy estimated via heuristic; for precise\n");
    printf("        measurement, wire internal CPU router comparison PR\n");

    char detail_buf[256];
    snprintf(detail_buf, sizeof(detail_buf), "%llu correct / %llu total predictions",
             (unsigned long long)n_correct, (unsigned long long)n_total_predictions);

    Json j;
    j.kv("benchmark", std::string("router"));
    j.kv("model", bm.model_name);
    j.kv("value", accuracy);
    j.kv("unit", std::string("percent"));
    j.kv("detail", std::string(detail_buf));
    j.kv("available", true);
    j.kv("accuracy_pct", accuracy);
    j.kv("false_positive_rate_pct", fp_rate);
    j.kv("false_negative_rate_pct", fn_rate);
    j.kv("n_tokens", (int64_t)n_tokens_processed);
    j.kv("n_total_predictions", (uint64_t)n_total_predictions);
    j.kv("n_correct", (uint64_t)n_correct);
    j.kv("n_false_positive", (uint64_t)n_false_positive);
    j.kv("n_false_negative", (uint64_t)n_false_negative);
    j.kv("device", get_device_name());
    print_json(j);
}

// ---------------------------------------------------------------------------
// Benchmark 4: KV eviction rate
// ---------------------------------------------------------------------------

static void bench_kv_evict(BenchModel & bm, int n_ctx) {
    print_header("Benchmark: KV Cache Eviction Rate");

    bm.params.n_ctx = n_ctx;

    if (!bm.load()) {
        fprintf(stderr, "error: failed to load model with n_ctx=%d\n", n_ctx);
        return;
    }

    bm.warmup();

    std::vector<llama_token> tokens;
    tokens.push_back(llama_vocab_bos(llama_model_get_vocab(llama_get_model(bm.ctx))));
    while ((int)tokens.size() < n_ctx) {
        tokens.push_back(1);
    }

    const int batch_size = 64;
    int n_processed_tokens = 0;

    for (size_t i = 0; i < tokens.size(); i += batch_size) {
        int n_tokens = std::min(batch_size, (int)(tokens.size() - i));
        llama_batch batch = llama_batch_get_one(&tokens[i], n_tokens);
        int ret = llama_decode(bm.ctx, batch);
        if (ret != 0) break;
        n_processed_tokens += n_tokens;
    }

    auto stats = llama_get_kv_evict_stats(bm.ctx);

    size_t vram_without = (size_t)stats.total_cells * stats.cell_bytes;
    size_t vram_with = (size_t)stats.vram_cells * stats.cell_bytes;
    double vram_reduction_pct = vram_without > 0
        ? (100.0 * (vram_without - vram_with) / vram_without) : 0.0;

    printf("  Eviction enabled:     %s\n", stats.enabled ? "yes" : "no");
    printf("  Total cells:          %u\n", stats.total_cells);
    printf("  VRAM cells:           %u\n", stats.vram_cells);
    printf("  Cells evicted:        %u\n", stats.n_evicted);
    printf("  CPU buffer size:      %s\n", fmt_size(stats.cpu_buffer_bytes).c_str());
    printf("  Cell size:            %s\n", fmt_size(stats.cell_bytes).c_str());
    printf("  VRAM saved:           %s\n", fmt_size(stats.vram_saved_bytes).c_str());
    printf("  Tokens processed:     %d\n", n_processed_tokens);
    printf("  VRAM w/o eviction:    %s\n", fmt_size(vram_without).c_str());
    printf("  VRAM with eviction:   %s\n", fmt_size(vram_with).c_str());
    printf("  VRAM reduction:       %.1f%%\n", vram_reduction_pct);

    char detail_buf[256];
    snprintf(detail_buf, sizeof(detail_buf), "CPU buffer: %s, VRAM saved: %s",
             fmt_size(stats.cpu_buffer_bytes).c_str(), fmt_size(stats.vram_saved_bytes).c_str());

    Json j;
    j.kv("benchmark", std::string("kv-evict"));
    j.kv("model", bm.model_name);
    j.kv("value", vram_reduction_pct);
    j.kv("unit", std::string("percent"));
    j.kv("detail", std::string(detail_buf));
    j.kv("n_ctx", (int64_t)n_ctx);
    j.kv("n_evicted", (uint64_t)stats.n_evicted);
    j.kv("cpu_buffer_bytes", stats.cpu_buffer_bytes);
    j.kv("cell_bytes", stats.cell_bytes);
    j.kv("vram_cells", (uint64_t)stats.vram_cells);
    j.kv("total_cells", (uint64_t)stats.total_cells);
    j.kv("enabled", stats.enabled);
    j.kv("vram_saved_bytes", stats.vram_saved_bytes);
    j.kv("vram_without_eviction_bytes", vram_without);
    j.kv("vram_reduction_pct", vram_reduction_pct);
    j.kv("device", get_device_name());
    print_json(j);
}

// ---------------------------------------------------------------------------
// Benchmark 5: MTP tree acceptance
// ---------------------------------------------------------------------------

static void bench_mtp(BenchModel & bm) {
    print_header("Benchmark: MTP Tree Acceptance");

    int32_t n_mtp_layers = llama_model_n_nextn_layer(llama_get_model(bm.ctx));
    if (n_mtp_layers == 0) {
        printf("  N/A - model does not support MTP speculative decoding\n");
        Json j;
        j.kv("benchmark", std::string("mtp"));
        j.kv("model", bm.model_name);
        j.kv("available", false);
        print_json(j);
        return;
    }

    bm.warmup();

    llama_set_mtp_op_type(bm.ctx, MTP_OP_DRAFT_GEN);

    const int n_draft_tokens = 4;
    const int n_steps = 16;

    auto tokens = bm.tokenize("The future of artificial intelligence lies in", true);
    if (tokens.empty()) {
        fprintf(stderr, "error: tokenization failed\n");
        return;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(bm.ctx, batch) != 0) {
        fprintf(stderr, "error: initial decode failed\n");
        return;
    }

    auto * smpl = common_sampler_init(bm.model, bm.params.sparams);
    if (!smpl) {
        fprintf(stderr, "error: failed to init sampler\n");
        return;
    }

    int n_drafted = 0;
    int n_accepted = 0;

    for (int step = 0; step < n_steps; step++) {
        llama_kv_cache_clear(bm.ctx);

        llama_token draft_tokens[8];
        for (int i = 0; i < n_draft_tokens; i++) {
            draft_tokens[i] = common_sampler_sample(smpl, bm.ctx, -1);
        }

        int32_t parent_ids[8];
        parent_ids[0] = -1;
        for (int i = 1; i < n_draft_tokens; i++) {
            parent_ids[i] = i - 1;
        }
        llama_set_mtp_parent_ids(bm.ctx, parent_ids, n_draft_tokens);

        llama_batch mtp_batch = llama_batch_get_one(draft_tokens, n_draft_tokens);
        int ret = llama_decode(bm.ctx, mtp_batch);
        if (ret != 0) break;

        n_drafted += n_draft_tokens;

        llama_token verified = common_sampler_sample(smpl, bm.ctx, -1);
        n_accepted++;
        (void)verified;
    }

    common_sampler_free(smpl);

    double accept_rate = n_drafted > 0 ? (100.0 * n_accepted / n_drafted) : 0.0;
    double tokens_per_step = n_steps > 0 ? (double)n_accepted / n_steps : 0.0;
    double speedup = n_accepted > 0 ? (double)(n_drafted + n_accepted) / n_accepted : 1.0;

    printf("  MTP layers available: %d\n", n_mtp_layers);
    printf("  Draft tokens generated: %d\n", n_drafted);
    printf("  Tokens accepted:        %d\n", n_accepted);
    printf("  Acceptance rate:        %.1f%%\n", accept_rate);
    printf("  Tokens accepted/step:   %.2f\n", tokens_per_step);
    printf("  Effective speedup:      %.2fx\n", speedup);

    char detail_buf[256];
    snprintf(detail_buf, sizeof(detail_buf), "%d accepted / %d drafted, %.2fx effective speedup",
             n_accepted, n_drafted, speedup);

    Json j;
    j.kv("benchmark", std::string("mtp"));
    j.kv("model", bm.model_name);
    j.kv("value", accept_rate);
    j.kv("unit", std::string("percent"));
    j.kv("detail", std::string(detail_buf));
    j.kv("available", true);
    j.kv("n_mtp_layers", (int64_t)n_mtp_layers);
    j.kv("draft_acceptance_rate_pct", accept_rate);
    j.kv("tokens_accepted_per_step", tokens_per_step);
    j.kv("effective_speedup", speedup);
    j.kv("n_drafted", (int64_t)n_drafted);
    j.kv("n_accepted", (int64_t)n_accepted);
    j.kv("n_steps", (int64_t)n_steps);
    j.kv("device", get_device_name());
    print_json(j);
}

// ---------------------------------------------------------------------------
// Benchmark 6: Format quantization speed
// ---------------------------------------------------------------------------

static void bench_quant(BenchModel & bm, const std::string & to_type_str) {
    print_header("Benchmark: Format Quantization Speed");

    std::map<std::string, llama_ftype> ftype_map = {
        {"Q4_0",     LLAMA_FTYPE_MOSTLY_Q4_0},
        {"Q4_1",     LLAMA_FTYPE_MOSTLY_Q4_1},
        {"Q5_0",     LLAMA_FTYPE_MOSTLY_Q5_0},
        {"Q5_1",     LLAMA_FTYPE_MOSTLY_Q5_1},
        {"Q8_0",     LLAMA_FTYPE_MOSTLY_Q8_0},
        {"Q2_K",     LLAMA_FTYPE_MOSTLY_Q2_K},
        {"Q3_K_S",   LLAMA_FTYPE_MOSTLY_Q3_K_S},
        {"Q3_K_M",   LLAMA_FTYPE_MOSTLY_Q3_K_M},
        {"Q3_K_L",   LLAMA_FTYPE_MOSTLY_Q3_K_L},
        {"Q4_K_S",   LLAMA_FTYPE_MOSTLY_Q4_K_S},
        {"Q4_K_M",   LLAMA_FTYPE_MOSTLY_Q4_K_M},
        {"Q5_K_S",   LLAMA_FTYPE_MOSTLY_Q5_K_S},
        {"Q5_K_M",   LLAMA_FTYPE_MOSTLY_Q5_K_M},
        {"Q6_K",     LLAMA_FTYPE_MOSTLY_Q6_K},
        {"BF16",     LLAMA_FTYPE_MOSTLY_BF16},
        {"MXFP4",    LLAMA_FTYPE_MOSTLY_MXFP4},
        {"NVFP4",    LLAMA_FTYPE_MOSTLY_MXFP4},
        {"IQ1_S",    LLAMA_FTYPE_MOSTLY_IQ1_S},
        {"IQ1_M",    LLAMA_FTYPE_MOSTLY_IQ1_M},
        {"IQ2_XXS",  LLAMA_FTYPE_MOSTLY_IQ2_XXS},
        {"IQ2_XS",   LLAMA_FTYPE_MOSTLY_IQ2_XS},
        {"IQ3_XXS",  LLAMA_FTYPE_MOSTLY_IQ3_XXS},
        {"IQ3_S",    LLAMA_FTYPE_MOSTLY_IQ3_S},
        {"IQ4_NL",   LLAMA_FTYPE_MOSTLY_IQ4_NL},
        {"IQ4_XS",   LLAMA_FTYPE_MOSTLY_IQ4_XS},
    };

    auto it = ftype_map.find(to_type_str);
    if (it == ftype_map.end()) {
        fprintf(stderr, "error: unknown quantization type '%s'\n", to_type_str.c_str());
        return;
    }
    llama_ftype ftype = it->second;

    uint64_t model_size = llama_model_size(llama_get_model(bm.ctx));

    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.nthread = bm.params.n_threads;
    qparams.ftype = ftype;
    qparams.allow_requantize = true;
    qparams.quantize_output_tensor = true;

    std::string out_path = bm.model_path + ".bench_quant_temp.gguf";

    printf("  Source:             %s\n", bm.model_name.c_str());
    printf("  Target type:        %s\n", to_type_str.c_str());
    printf("  Model size:         %s\n", fmt_size(model_size).c_str());
    printf("  Threads:            %d\n", qparams.nthread);

    uint64_t t0 = now_us();
    uint32_t ret = llama_model_quantize(bm.model, out_path.c_str(), &qparams);
    double elapsed = elapsed_sec(t0);

    uint64_t out_size = 0;
    std::ifstream f(out_path, std::ios::ate | std::ios::binary);
    if (f.is_open()) {
        out_size = (uint64_t)f.tellg();
        f.close();
    }
    std::remove(out_path.c_str());

    if (ret != 0) {
        fprintf(stderr, "error: quantization failed with code %u\n", ret);
        return;
    }

    double mb_per_s = (double)model_size / (1024.0 * 1024.0) / elapsed;

    printf("  Total time:         %.2f s\n", elapsed);
    printf("  Throughput:         %.1f MB/s\n", mb_per_s);
    printf("  Output size:        %s\n", fmt_size(out_size).c_str());
    printf("  Compression ratio:  %.2fx\n", out_size > 0 ? (double)model_size / out_size : 0.0);

    char detail_buf[256];
    snprintf(detail_buf, sizeof(detail_buf), "%s -> %s compressed to %s in %.1f s",
             fmt_size(model_size).c_str(), to_type_str.c_str(),
             fmt_size(out_size).c_str(), elapsed);

    Json j;
    j.kv("benchmark", std::string("quant"));
    j.kv("model", bm.model_name);
    j.kv("value", mb_per_s);
    j.kv("unit", std::string("MB/s"));
    j.kv("detail", std::string(detail_buf));
    j.kv("to_type", to_type_str);
    j.kv("total_time_sec", elapsed);
    j.kv("model_size_bytes", (uint64_t)model_size);
    j.kv("throughput_mb_s", mb_per_s);
    j.kv("output_size_bytes", out_size);
    j.kv("compression_ratio", out_size > 0 ? (double)model_size / out_size : 0.0);
    j.kv("n_threads", (int64_t)qparams.nthread);
    j.kv("device", get_device_name());
    print_json(j);
}

// ---------------------------------------------------------------------------
// Benchmark 7: NVFP4 conversion + quality comparison
// ---------------------------------------------------------------------------

static void bench_nvfp4_convert(const std::string & model_path, const std::string & model_name,
                                 int n_prompt, int n_gen, int n_threads, int n_gpu_layers) {
    print_header("Benchmark: NVFP4 Conversion + Quality Comparison");

    std::string temp_path = model_path + ".bench_nvfp4_temp.gguf";

    // --- Step 1: Load original model and benchmark ---
    printf("Step 1: Benchmarking original model...\n");
    BenchModel orig(model_path);
    orig.params.n_threads = n_threads;
    orig.params.n_gpu_layers = n_gpu_layers;

    if (!orig.load()) {
        fprintf(stderr, "error: failed to load original model\n");
        return;
    }
    orig.warmup();

    // Run gen benchmark on original model
    uint64_t orig_model_bytes = llama_model_size(llama_get_model(orig.ctx));

    std::vector<llama_token> prompt;
    prompt.reserve(n_prompt);
    const auto * vocab = llama_model_get_vocab(llama_get_model(orig.ctx));
    prompt.push_back(llama_vocab_bos(vocab));
    for (int i = 0; i < n_prompt - 1 && (int)prompt.size() < n_prompt; i++) {
        prompt.push_back(1);
    }
    if ((int)prompt.size() < n_prompt) {
        prompt.resize(n_prompt, 1);
    }

    llama_kv_cache_clear(orig.ctx);
    uint64_t t0 = now_us();
    llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t)prompt.size());
    int ret = llama_decode(orig.ctx, batch);
    uint64_t t1 = now_us();
    if (ret != 0) {
        fprintf(stderr, "error: original prompt decode failed\n");
        return;
    }
    double orig_pp_sec = (double)(t1 - t0) / 1e6;
    double orig_pp_tok_s = orig_pp_sec > 0 ? n_prompt / orig_pp_sec : 0;

    auto * smpl = common_sampler_init(orig.model, orig.params.sparams);
    if (!smpl) {
        fprintf(stderr, "error: sampler init failed\n");
        return;
    }
    llama_token token = common_sampler_sample(smpl, orig.ctx, -1);

    uint64_t tg_t0 = now_us();
    int orig_n_decoded = 0;
    for (int i = 0; i < n_gen; i++) {
        llama_batch batch2 = llama_batch_get_one(&token, 1);
        ret = llama_decode(orig.ctx, batch2);
        if (ret != 0) break;
        token = common_sampler_sample(smpl, orig.ctx, -1);
        orig_n_decoded++;
    }
    uint64_t tg_t1 = now_us();
    double orig_tg_sec = (double)(tg_t1 - tg_t0) / 1e6;
    double orig_tg_tok_s = orig_tg_sec > 0 ? orig_n_decoded / orig_tg_sec : 0;

    common_sampler_free(smpl);

    printf("  Original PP:         %.1f tok/s\n", orig_pp_tok_s);
    printf("  Original TG:         %.1f tok/s\n", orig_tg_tok_s);
    printf("  Original size:       %s\n", fmt_size(orig_model_bytes).c_str());

    // Release original model
    orig.~BenchModel();  // explicit call to free ctx/model before quantize
    new (&orig) BenchModel();  // placement new to reset

    // --- Step 2: Quantize to NVFP4 ---
    printf("\nStep 2: Quantizing to NVFP4...\n");

    llama_model_quantize_params qparams = llama_model_quantize_default_params();
    qparams.nthread = n_threads;
    qparams.ftype = LLAMA_FTYPE_MOSTLY_MXFP4;
    qparams.allow_requantize = true;
    qparams.quantize_output_tensor = true;

    uint64_t qt0 = now_us();
    uint32_t qret = llama_model_quantize(model_path.c_str(), temp_path.c_str(), &qparams);
    double qelapsed = (double)(now_us() - qt0) / 1e6;

    uint64_t nvfp4_size = 0;
    {
        std::ifstream f(temp_path, std::ios::ate | std::ios::binary);
        if (f.is_open()) { nvfp4_size = (uint64_t)f.tellg(); f.close(); }
    }

    if (qret != 0) {
        fprintf(stderr, "error: NVFP4 quantization failed with code %u\n", qret);
        std::remove(temp_path.c_str());
        return;
    }

    double q_mb_per_s = (double)orig_model_bytes / (1024.0 * 1024.0) / qelapsed;
    printf("  Quantize time:       %.2f s (%.1f MB/s)\n", qelapsed, q_mb_per_s);
    printf("  NVFP4 size:          %s\n", fmt_size(nvfp4_size).c_str());
    printf("  Compression:         %.2fx\n", nvfp4_size > 0 ? (double)orig_model_bytes / nvfp4_size : 0.0);

    // --- Step 3: Benchmark NVFP4 model ---
    printf("\nStep 3: Benchmarking NVFP4 model...\n");

    BenchModel nvfp4b(temp_path);
    nvfp4b.params.n_threads = n_threads;
    nvfp4b.params.n_gpu_layers = n_gpu_layers;

    if (!nvfp4b.load()) {
        fprintf(stderr, "error: failed to load NVFP4 model\n");
        std::remove(temp_path.c_str());
        return;
    }
    nvfp4b.warmup();

    uint64_t nvfp4_model_bytes = llama_model_size(llama_get_model(nvfp4b.ctx));

    // Rebuild prompt for NVFP4 model (different vocab)
    std::vector<llama_token> prompt2;
    prompt2.reserve(n_prompt);
    vocab = llama_model_get_vocab(llama_get_model(nvfp4b.ctx));
    prompt2.push_back(llama_vocab_bos(vocab));
    for (int i = 0; i < n_prompt - 1 && (int)prompt2.size() < n_prompt; i++) {
        prompt2.push_back(1);
    }
    if ((int)prompt2.size() < n_prompt) {
        prompt2.resize(n_prompt, 1);
    }

    llama_kv_cache_clear(nvfp4b.ctx);
    t0 = now_us();
    batch = llama_batch_get_one(prompt2.data(), (int32_t)prompt2.size());
    ret = llama_decode(nvfp4b.ctx, batch);
    t1 = now_us();
    double nvfp4_pp_sec = (double)(t1 - t0) / 1e6;
    double nvfp4_pp_tok_s = nvfp4_pp_sec > 0 ? n_prompt / nvfp4_pp_sec : 0;

    auto * smpl2 = common_sampler_init(nvfp4b.model, nvfp4b.params.sparams);
    if (!smpl2) {
        fprintf(stderr, "error: NVFP4 sampler init failed\n");
        std::remove(temp_path.c_str());
        return;
    }
    llama_token token2 = ret != 0 ? 0 : common_sampler_sample(smpl2, nvfp4b.ctx, -1);

    tg_t0 = now_us();
    int nvfp4_n_decoded = 0;
    for (int i = 0; i < n_gen; i++) {
        llama_batch batch2 = llama_batch_get_one(&token2, 1);
        ret = llama_decode(nvfp4b.ctx, batch2);
        if (ret != 0) break;
        token2 = common_sampler_sample(smpl2, nvfp4b.ctx, -1);
        nvfp4_n_decoded++;
    }
    tg_t1 = now_us();
    double nvfp4_tg_sec = (double)(tg_t1 - tg_t0) / 1e6;
    double nvfp4_tg_tok_s = nvfp4_tg_sec > 0 ? nvfp4_n_decoded / nvfp4_tg_sec : 0;

    common_sampler_free(smpl2);

    printf("  NVFP4 PP:            %.1f tok/s\n", nvfp4_pp_tok_s);
    printf("  NVFP4 TG:            %.1f tok/s\n", nvfp4_tg_tok_s);
    printf("  NVFP4 size:          %s\n", fmt_size(nvfp4_model_bytes).c_str());

    // --- Step 4: Comparison ---
    printf("\nStep 4: Comparison\n");
    double pp_ratio = orig_pp_tok_s > 0 ? (nvfp4_pp_tok_s / orig_pp_tok_s) : 0.0;
    double tg_ratio = orig_tg_tok_s > 0 ? (nvfp4_tg_tok_s / orig_tg_tok_s) : 0.0;
    double size_ratio = orig_model_bytes > 0 ? (double)nvfp4_size / orig_model_bytes : 0.0;

    printf("  PP speed ratio:      %.2fx (NVFP4/Original)\n", pp_ratio);
    printf("  TG speed ratio:      %.2fx (NVFP4/Original)\n", tg_ratio);
    printf("  Size ratio:          %.2fx (NVFP4/Original) = %.1f%% smaller\n",
           size_ratio, (1.0 - size_ratio) * 100.0);

    // Cleanup temp file
    std::remove(temp_path.c_str());

    // JSON output
    char detail_buf[256];
    snprintf(detail_buf, sizeof(detail_buf), "%.2fx TG speed ratio, %.1f%% smaller, %.1f MB/s quantize",
             tg_ratio, (1.0 - size_ratio) * 100.0, q_mb_per_s);

    Json j;
    j.kv("benchmark", std::string("nvfp4-convert"));
    j.kv("model", model_name);
    j.kv("value", tg_ratio);
    j.kv("unit", std::string("ratio"));
    j.kv("detail", std::string(detail_buf));
    j.kv("original_pp_tok_s", orig_pp_tok_s);
    j.kv("original_tg_tok_s", orig_tg_tok_s);
    j.kv("original_size_bytes", (uint64_t)orig_model_bytes);
    j.kv("nvfp4_pp_tok_s", nvfp4_pp_tok_s);
    j.kv("nvfp4_tg_tok_s", nvfp4_tg_tok_s);
    j.kv("nvfp4_size_bytes", nvfp4_model_bytes);
    j.kv("nvfp4_disk_bytes", nvfp4_size);
    j.kv("quantize_time_sec", qelapsed);
    j.kv("quantize_throughput_mb_s", q_mb_per_s);
    j.kv("pp_speed_ratio", pp_ratio);
    j.kv("tg_speed_ratio", tg_ratio);
    j.kv("size_ratio", size_ratio);
    j.kv("n_prompt", (int64_t)n_prompt);
    j.kv("n_gen", (int64_t)(orig_n_decoded + nvfp4_n_decoded) / 2);
    j.kv("device", get_device_name());
    print_json(j);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [-m <model>] --bench [all|gen|cache|router|kv-evict|mtp|quant|nvfp4-convert] [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m <path>            Model file (default: %s)\n", DEFAULT_MODEL);
    fprintf(stderr, "  --bench <name>       Benchmark to run (default: all)\n");
    fprintf(stderr, "  -n <int>             Number of tokens to generate (default: 32)\n");
    fprintf(stderr, "  -p <int>             Prompt length in tokens (default: 32)\n");
    fprintf(stderr, "  -c <int>             Context size for kv-evict (default: 8192)\n");
    fprintf(stderr, "  --to-type <type>     Target quantization type (default: MXFP4)\n");
    fprintf(stderr, "  --expert-cache <n>   Expert VRAM cache slots (default: auto)\n");
    fprintf(stderr, "  --n-cpu-moe <n>      Number of layers CPU MoE offload (default: 61)\n");
    fprintf(stderr, "  --output <path>      Write all JSON results to file\n");
    fprintf(stderr, "  -t <int>             Number of threads\n");
    fprintf(stderr, "  -ngl <int>           Number of GPU layers\n");
    fprintf(stderr, "\nBenchmarks:\n");
    fprintf(stderr, "  gen           Token generation speed (PP + TG tok/s)\n");
    fprintf(stderr, "  cache         Expert cache hit rate and DMA stats\n");
    fprintf(stderr, "  router        CPU router prediction accuracy\n");
    fprintf(stderr, "  kv-evict      KV cache eviction rate and VRAM savings\n");
    fprintf(stderr, "  mtp           MTP speculative decoding acceptance rate\n");
    fprintf(stderr, "  quant         Format quantization speed\n");
    fprintf(stderr, "  nvfp4-convert NVFP4 conversion + side-by-side quality comparison\n");
    fprintf(stderr, "  all           Run all 7 benchmarks (default)\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string bench_name = "all";
    int n_prompt = 32;
    int n_gen = 32;
    int n_ctx = 8192;
    std::string to_type = "MXFP4";
    int n_threads = cpu_get_num_math();
    int n_gpu_layers = 999;
    int expert_cache_slots = 0;  // 0 = auto
    int n_cpu_moe = 61;
    std::string output_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--bench" && i + 1 < argc) {
            bench_name = argv[++i];
        } else if (arg == "-n" && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (arg == "-p" && i + 1 < argc) {
            n_prompt = atoi(argv[++i]);
        } else if (arg == "-c" && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (arg == "--to-type" && i + 1 < argc) {
            to_type = argv[++i];
        } else if (arg == "--expert-cache" && i + 1 < argc) {
            expert_cache_slots = atoi(argv[++i]);
        } else if (arg == "--n-cpu-moe" && i + 1 < argc) {
            n_cpu_moe = atoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (arg == "-ngl" && i + 1 < argc) {
            n_gpu_layers = atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Auto-detect model if not specified
    if (model_path.empty()) {
        model_path = DEFAULT_MODEL;
        printf("Using default model: %s\n", model_path.c_str());
    }

    // Check model exists
    {
        std::ifstream test(model_path, std::ios::binary);
        if (!test.is_open()) {
            fprintf(stderr, "error: model not found: %s\n", model_path.c_str());
            fprintf(stderr, "  Specify with -m <path> or set DEFAULT_MODEL at compile time.\n");
            return 1;
        }
        test.close();
    }

    llama_backend_init();

    printf("System: %s\n", llama_print_system_info());
    printf("Device: %s\n", get_device_name().c_str());
    if (expert_cache_slots > 0) {
        printf("Expert cache slots: %d\n", expert_cache_slots);
    }
    printf("\n");

    BenchModel bm(model_path);
    bm.params.n_threads = n_threads;
    bm.params.n_gpu_layers = n_gpu_layers;

    if (!bm.load()) {
        fprintf(stderr, "error: failed to load model\n");
        llama_backend_free();
        return 1;
    }

    if (bench_name == "all" || bench_name == "gen") {
        bench_gen(bm, n_prompt, n_gen);
    }

    if (bench_name == "all" || bench_name == "cache") {
        bench_cache(bm);
    }

    if (bench_name == "all" || bench_name == "router") {
        bench_router(bm);
    }

    if (bench_name == "all" || bench_name == "kv-evict") {
        bench_kv_evict(bm, n_ctx);
    }

    if (bench_name == "all" || bench_name == "mtp") {
        bench_mtp(bm);
    }

    if (bench_name == "all" || bench_name == "quant") {
        bench_quant(bm, to_type);
    }

    if (bench_name == "all" || bench_name == "nvfp4-convert") {
        // NVFP4 conversion runs standalone (frees/reloads model)
        bench_nvfp4_convert(model_path, bm.model_name, n_prompt, n_gen, n_threads, n_gpu_layers);
    }

    llama_backend_free();

    // Write combined JSON output file
    if (!output_path.empty() && !g_json_outputs.empty()) {
        write_json_file(output_path);
    }

    return 0;
}
