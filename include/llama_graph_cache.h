#pragma once

#include "ggml.h"
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>

// Graph cache: eliminates redundant graph rebuilds during decode.
//
// Most decode steps produce identical graph structure (same operations,
// same tensor shapes).  Only input tensors change (token IDs, positions,
// KV cache views).  This cache stores the built graph and reuses it.
//
// Keyed by: architecture + n_layers + n_tokens + n_kv + flash_attn flag
// These parameters capture all structural variation in the graph.
//
// On cache hit: skip ggml_graph_alloc and graph building, update inputs.
// On cache miss: build graph, store in cache, return.
struct ggml_graph_cache_key {
    int32_t arch;        // model architecture enum
    int32_t n_layers;    // number of layers
    int32_t n_tokens;    // number of tokens in batch
    int32_t n_kv;        // KV cache size
    int32_t flash_attn;  // flash attention enabled
    int32_t causal;      // causal attention
    int32_t padding;     // padding for alignment (unused)

    bool operator==(const ggml_graph_cache_key & o) const {
        return arch == o.arch && n_layers == o.n_layers && n_tokens == o.n_tokens
            && n_kv == o.n_kv && flash_attn == o.flash_attn && causal == o.causal;
    }
};

struct ggml_graph_cache_entry {
    ggml_graph_cache_key key;
    ggml_cgraph * graph = nullptr;
    int64_t       timestamp = 0;  // LRU timestamp
    int32_t       hits = 0;       // how many times reused
};

// Thread-safe LRU graph cache
struct ggml_graph_cache {
    static constexpr int MAX_ENTRIES = 8;  // cache up to 8 graph variants

    ggml_graph_cache_entry entries[MAX_ENTRIES];
    int n_entries = 0;
    int64_t clock = 0;
    std::mutex mtx;

    // Lookup a cached graph.  Returns nullptr if not found.
    ggml_cgraph * lookup(const ggml_graph_cache_key & key) {
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < n_entries; i++) {
            if (entries[i].key == key) {
                entries[i].timestamp = ++clock;
                entries[i].hits++;
                return entries[i].graph;
            }
        }
        return nullptr;
    }

    // Insert a graph into the cache.  Evicts LRU if full.
    void insert(const ggml_graph_cache_key & key, ggml_cgraph * graph) {
        std::lock_guard<std::mutex> lock(mtx);
        // Find or create slot
        int slot = -1;
        int oldest = 0;
        for (int i = 0; i < n_entries; i++) {
            if (entries[i].key == key) { slot = i; break; }
            if (entries[i].timestamp < entries[oldest].timestamp) oldest = i;
        }
        if (slot < 0) {
            if (n_entries < MAX_ENTRIES) {
                slot = n_entries++;
            } else {
                slot = oldest;  // evict LRU
                // Evict old graph entry (ctx field removed from ggml_cgraph in this fork)
                if (entries[slot].graph) {
                    entries[slot].graph = nullptr;
                }
            }
        }
        entries[slot].key = key;
        entries[slot].graph = graph;
        entries[slot].timestamp = ++clock;
        entries[slot].hits = 0;
    }

    // Invalidate cache entries when the model changes
    void invalidate() {
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < n_entries; i++) {
            if (entries[i].graph) {
                entries[i].graph = nullptr;
            }
        }
        n_entries = 0;
    }

    ~ggml_graph_cache() { invalidate(); }
};
