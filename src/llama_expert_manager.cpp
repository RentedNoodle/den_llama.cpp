#include "llama-expert-io.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/**
 * Initializes the expert table and sets up the offloading memory pools.
 */
void llama_expert_manager_init(llama_expert_table_t* table, uint32_t num_experts) {
    table->num_experts = num_experts;
    table->experts = (llama_expert_entry_t*)malloc(sizeof(llama_expert_entry_t) * num_experts);
    
    for(uint32_t i = 0; i < num_experts; ++i) {
        table->experts[i].expert_id = i;
        table->experts[i].status = EXPERT_COLD;
        table->experts[i].heat_score = 0.0f;
        table->experts[i].priority_score = 0.0f;
        table->experts[i].vram_ptr = NULL;
        table->experts[i].ram_ptr = NULL;
        table->experts[i].last_used_tick = 0;
    }
    table->mtp_context = NULL;
}

/**
 * The core "Oracle" logic. Updates scores based on heat and MTP confidence.
 */
void llama_expert_manager_update_status(llama_expert_table_t* table) {
    static uint32_t global_tick = 0;
    global_tick++;

    for(uint32_t i = 0; i < table->num_experts; ++i) {
        // Decay heat over time
        table->experts[i].heat_score *= 0.95f;

        // Check for "Hot" status
        if (table->experts[i].heat_score > 0.8f && table->experts[i].status == EXPERT_COLD) {
            table->experts[i].status = EXPERT_WARM;
        }
        
        // If it's very hot, ensure it stays in VRAM
        if (table->experts[i].heat_score > 1.5f) {
            table->experts[i].status = EXPERT_HOT;
        }
    }
}

/**
 * The "Muscle": Asynchronously triggers loads for the next predicted experts.
 */
void llama_expert_manager_fetch_async(llama_expert_table_t* table, uint32_t* expert_ids, uint32_t count) {
    for(uint32_t i = 0; i < count; ++i) {
        uint32_t id = expert_ids[i];
        if (table->experts[id].status == EXPERT_COLD) {
            // Trigger background transfer from RAM to VRAM
            table->experts[id].status = EXPERT_WARM;
            // Placeholder for: cudaMemcpyAsync(table->experts[id].vram_ptr, table->experts[id].ram_ptr, size, stream);
        } else if (table->experts[id].status == EXPERT_HOT) {
            // Already in VRAM, ready for immediate execution
            table->experts[id].heat_score += 1.0f;
        }
    }
}

/**
 * Eviction policy to ensure VRAM doesn't overflow.
 */
void llama_expert_manager_evict_cold(llama_expert_table_t* table) {
    for(uint32_t i = 0; i < table->num_experts; ++i) {
        if (table->experts[i].status == EXPERT_HOT && table->experts[i].heat_score < 0.5f) {
            table->experts[i].status = EXPERT_COLD;
            // Free VRAM pointer here
        }
    }
}
