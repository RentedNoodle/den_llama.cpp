#ifndef LLAMA_KV_CACHE_H
#define LLAMA_KV_CACHE_H

#include "llama.h"

/**
 * llama_kv_cache_t handles the sharding of KV cache across VRAM and RAM.
 * This is the core "beeLlama" integration for high-context support.
 */
typedef struct llama_kv_cache {
    void* vram_shards;
    void* ram_shards;
    uint32_t shard_count;
    uint32_t current_shard;
    size_t shard_size;
} llama_kv_cache_t;

void llama_kv_cache_init(llama_kv_cache_t* cache, uint32_t shard_count, size_t shard_size);
void llama_kv_cache_push(llama_kv_cache_t* cache, const float* data);
void llama_kv_cache_pop(llama_kv_cache_t* cache, float* out_data);

#endif