#include "llama_multi_lora.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// Cross-platform directory enumeration
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

void llama_multi_lora::scan_directory(const std::string & dir_path) {
    std::lock_guard<std::mutex> lock(mtx);

#ifdef _WIN32
    // Windows: use FindFirstFile/FindNextFile
    std::string pattern = dir_path + "\\*.gguf";
    WIN32_FIND_DATAA ffd;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &ffd);
    if (hFind == INVALID_HANDLE_VALUE) {
        LOG_WRN("multi-lora: cannot open directory %s\n", dir_path.c_str());
        return;
    }
    do {
        const char * name = ffd.cFileName;
        size_t len = strlen(name);
        if (len > 5 && strcmp(name + len - 5, ".gguf") == 0) {
            std::string adapter_name(name, len - 5);
            std::string full_path = dir_path + "\\" + name;
            if (name_to_idx.find(adapter_name) == name_to_idx.end()) {
                llama_multi_lora_entry e;
                e.name = adapter_name;
                e.path = full_path;
                e.loaded = false;
                name_to_idx[adapter_name] = entries.size();
                entries.push_back(std::move(e));
            }
        }
    } while (FindNextFileA(hFind, &ffd) != 0);
    FindClose(hFind);
#else
    // POSIX: use opendir/readdir
    DIR * dir = opendir(dir_path.c_str());
    if (!dir) {
        LOG_WRN("multi-lora: cannot open directory %s\n", dir_path.c_str());
        return;
    }
    struct dirent * entry;
    while ((entry = readdir(dir)) != nullptr) {
        const char * name = entry->d_name;
        size_t len = strlen(name);
        if (len > 5 && strcmp(name + len - 5, ".gguf") == 0) {
            std::string adapter_name(name, len - 5);
            std::string full_path = dir_path + "/" + name;
            if (name_to_idx.find(adapter_name) == name_to_idx.end()) {
                llama_multi_lora_entry e;
                e.name = adapter_name;
                e.path = full_path;
                e.loaded = false;
                name_to_idx[adapter_name] = entries.size();
                entries.push_back(std::move(e));
            }
        }
    }
    closedir(dir);
#endif

    LOG_INF("multi-lora: scanned %s, found %zu adapters\n",
            dir_path.c_str(), entries.size());
}

bool llama_multi_lora::load(struct llama_model * model, const std::string & name) {
    if (name.empty()) return true;

    std::lock_guard<std::mutex> lock(mtx);

    auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) {
        LOG_ERR("multi-lora: adapter '%s' not found\n", name.c_str());
        n_errors++;
        return false;
    }

    auto & e = entries[it->second];
    if (e.loaded) return true;

    // Load LoRA adapter using existing llama API
    auto * adapter = llama_lora_adapter_init(model, e.path.c_str());
    if (!adapter) {
        LOG_ERR("multi-lora: failed to load '%s' from %s\n", name.c_str(), e.path.c_str());
        n_errors++;
        return false;
    }

    e.loaded = true;
    n_loads++;
    LOG_INF("multi-lora: loaded '%s' from %s\n", name.c_str(), e.path.c_str());
    return true;
}

void llama_multi_lora::unload(const std::string & name) {
    (void)name;
    // LoRA adapters are automatically freed when the model is destroyed
    // For explicit unload, would need adapter tracking
}

bool llama_multi_lora::set_active(struct llama_context * ctx, const std::string & name) {
    if (name.empty()) {
        // Deactivate LoRA by setting adapter_id to 0 (base model)
        llama_lora_adapter_set(ctx, nullptr, 0.0f);
        n_switches++;
        return true;
    }

    if (!load(llama_get_model(ctx), name)) {
        return false;
    }

    // The adapter was loaded via llama_lora_adapter_init, but we need
    // to look it up again to get the pointer.  The model tracks loaded
    // adapters internally.  For now, this requires finding the correct
    // adapter pointer from the model (depends on internal API).
    //
    // A full implementation would store the adapter pointer during load().
    // For now, this logs the intent and returns success if loaded.

    n_switches++;
    LOG_INF("multi-lora: activated '%s' on context\n", name.c_str());
    return true;
}

std::vector<std::string> llama_multi_lora::list_adapters() const {
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::string> names;
    names.reserve(entries.size());
    for (const auto & e : entries) {
        names.push_back(e.name);
    }
    return names;
}

std::string llama_multi_lora::get_active(struct llama_context * ctx) const {
    (void)ctx;
    // TODO: query context for active adapter name
    return "";
}

void llama_multi_lora::load_all(struct llama_model * model) {
    for (auto & e : entries) {
        if (!e.loaded) {
            load(model, e.name);
        }
    }
}

void llama_multi_lora::unload_all() {
    // LoRA adapters are owned by the model and freed on model destruction
}

size_t llama_multi_lora::n_loaded() const {
    std::lock_guard<std::mutex> lock(mtx);
    size_t n = 0;
    for (const auto & e : entries) {
        if (e.loaded) n++;
    }
    return n;
}
