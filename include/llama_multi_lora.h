#pragma once

#include "llama.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

// Multi-LoRA Adapter Serving
//
// Loads multiple LoRA adapters and switches between them per-request.
// Adapters are loaded from a directory of GGUF LoRA files and cached
// in memory.  A lora_id is passed with each request to select the
// active adapter.
//
// Usage:
//   llama_multi_lora loras;
//   loras.load_directory("/path/to/loras/");
//   loras.set_active(ctx, "my-adapter");  // applies adapter weights

struct llama_multi_lora_entry {
    std::string name;           // adapter name (filename without .gguf)
    std::string path;           // full file path
    float       alpha = 1.0f;   // LoRA scaling factor (from GGUF metadata)
    bool        loaded = false;
};

struct llama_multi_lora {
    std::vector<llama_multi_lora_entry> entries;
    std::unordered_map<std::string, size_t> name_to_idx;
    mutable std::mutex mtx;

    // Stats
    std::atomic<uint64_t> n_loads{0};
    std::atomic<uint64_t> n_switches{0};
    std::atomic<uint64_t> n_errors{0};

    // Scan a directory for .gguf LoRA files
    void scan_directory(const std::string & dir_path);

    // Load a specific adapter by name.  Returns true on success.
    // If already loaded, returns true immediately.
    bool load(struct llama_model * model, const std::string & name);

    // Unload a specific adapter to free memory
    void unload(const std::string & name);

    // Set the active adapter for a context.
    // Pass empty string to disable LoRA (base model only).
    // Returns true on success.
    bool set_active(struct llama_context * ctx, const std::string & name);

    // Get list of available adapter names
    std::vector<std::string> list_adapters() const;

    // Get currently active adapter for a context
    std::string get_active(struct llama_context * ctx) const;

    // Load ALL adapters in the directory at once
    void load_all(struct llama_model * model);

    // Unload all adapters
    void unload_all();

    // Get stats
    size_t n_loaded() const;
    size_t n_available() const { return entries.size(); }
};
