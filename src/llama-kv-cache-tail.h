#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

constexpr int32_t LLAMA_KV_TAIL_BODY_SLOT = -1;

// Structured KV-cache exact-tail coverage state. Defined llama-side here because
// den's llama.h does not yet carry beellama's tail enums; the KVarN foundation
// and base memory interface reference these types. If upstream llama.h later
// adds them, drop the #ifndef guards.
#ifndef LLAMA_KV_TAIL_COVERAGE_STATE_DEFINED
#define LLAMA_KV_TAIL_COVERAGE_STATE_DEFINED

enum llama_kv_tail_coverage_state {
    LLAMA_KV_TAIL_COVERAGE_NONE,
    LLAMA_KV_TAIL_COVERAGE_PARTIAL,
    LLAMA_KV_TAIL_COVERAGE_COMPLETE,
};

enum llama_kv_tail_degradation_flags {
    LLAMA_KV_TAIL_DEGRADED_NONE            = 0,
    LLAMA_KV_TAIL_DEGRADED_BODY_ONLY_STATE = 1 << 0,
    LLAMA_KV_TAIL_DEGRADED_HISTORICAL_OP   = 1 << 1,
    LLAMA_KV_TAIL_DEGRADED_STATE_RESTORE   = 1 << 2,
    LLAMA_KV_TAIL_DEGRADED_PAYLOAD_INVALID = 1 << 3,
};

struct llama_kv_tail_coverage_info {
    llama_kv_tail_coverage_state state;
    uint32_t requested;
    uint32_t exact;
    uint32_t degradation_flags;
};

#endif // LLAMA_KV_TAIL_COVERAGE_STATE_DEFINED

enum llama_kv_tail_storage_kind {
    LLAMA_KV_TAIL_STORAGE_DISABLED,
    LLAMA_KV_TAIL_STORAGE_OVERLAY,
    LLAMA_KV_TAIL_STORAGE_NATIVE_EXACT,
    LLAMA_KV_TAIL_STORAGE_COMPACT_OVERLAY,
    LLAMA_KV_TAIL_STORAGE_COMPACT_NATIVE_EXACT,
};

enum llama_kv_tail_route : int {
    LLAMA_KV_TAIL_ROUTE_NONE,
    LLAMA_KV_TAIL_ROUTE_NATIVE,
    LLAMA_KV_TAIL_ROUTE_GENERIC,
};

enum llama_kv_tail_operation {
    LLAMA_KV_TAIL_OP_NONE,
    LLAMA_KV_TAIL_OP_WRITE_K,
    LLAMA_KV_TAIL_OP_WRITE_V,
    LLAMA_KV_TAIL_OP_GATHER_K,
    LLAMA_KV_TAIL_OP_GATHER_V,
    LLAMA_KV_TAIL_OP_BODY_SCORE,
    LLAMA_KV_TAIL_OP_BODY_VALUE,
    LLAMA_KV_TAIL_OP_EXACT_SCORE,
    LLAMA_KV_TAIL_OP_EXACT_VALUE,
    LLAMA_KV_TAIL_OP_GENERIC_MERGE,
    LLAMA_KV_TAIL_OP_NATIVE_ATTENTION,
};

struct llama_kv_tail_route_requirements {
    bool write_k = true;
    bool write_v = true;
    bool gather_k = true;
    bool gather_v = true;
    bool body_score = true;
    bool body_value = true;
    bool exact_score = true;
    bool exact_value = true;
    bool generic_merge = true;
    bool native_attention = false;
};

struct llama_kv_tail_route_capability {
    bool supported;
    llama_kv_tail_route route;
    llama_kv_tail_operation missing_operation;
};

// Backend-neutral ownership contract for one persistent KV layer.
struct llama_kv_tail_layer_route {
    uint32_t layer_id;
    std::string backend;
    ggml_type body_type_k;
    ggml_type body_type_v;
    ggml_type exact_type_k;
    ggml_type exact_type_v;
    bool v_transposed;
    bool causal_attn;
    bool swa;
    bool explicit_bias;
    bool has_body;
    bool has_current;
    uint32_t body_execution_rows;
    ggml_backend_dev_t owner;
    llama_kv_tail_route_capability capability;
};

struct llama_kv_tail_layer_ownership {
    uint32_t layer_id;
    uintptr_t body_k_owner;
    uintptr_t body_v_owner;
    uintptr_t shadow_k_owner;
    uintptr_t shadow_v_owner;
    uintptr_t graph_write_owner;
    uintptr_t graph_read_owner;
    uintptr_t state_k_owner;
    uintptr_t state_v_owner;
    bool body_k_meta_split;
    bool body_v_meta_split;
};

enum llama_kv_tail_ownership_error {
    LLAMA_KV_TAIL_OWNERSHIP_OK,
    LLAMA_KV_TAIL_OWNERSHIP_META_SPLIT_K,
    LLAMA_KV_TAIL_OWNERSHIP_META_SPLIT_V,
    LLAMA_KV_TAIL_OWNERSHIP_SHADOW_K,
    LLAMA_KV_TAIL_OWNERSHIP_SHADOW_V,
    LLAMA_KV_TAIL_OWNERSHIP_GRAPH_WRITE,
    LLAMA_KV_TAIL_OWNERSHIP_GRAPH_READ,
    LLAMA_KV_TAIL_OWNERSHIP_STATE_K,
    LLAMA_KV_TAIL_OWNERSHIP_STATE_V,
};

// A contiguous run of payload slots (payload indices [payload_begin, payload_begin+length)
// mapped to physical slot indices [slot_begin, slot_begin+length)).
struct llama_kv_tail_slot_run {
    uint32_t payload_begin;
    int32_t  slot_begin;
    uint32_t length;
};

//
// Function surface used by the KVarN foundation. Implementations live in
// llama-kv-cache-tail.cpp.
//

llama_kv_tail_route_capability llama_kv_tail_select_route(
        const llama_kv_tail_route_requirements & requirements);

const char * llama_kv_tail_operation_name(llama_kv_tail_operation operation);

// The packed body is graph-local and may be padded to a backend execution
// alignment without changing the persistent compact-history capacity.
uint32_t llama_kv_tail_packed_body_stride(
        uint64_t logical_rows,
        uint32_t alignment);

llama_kv_tail_layer_ownership llama_kv_tail_plan_layer_ownership(
        uint32_t layer_id,
        uintptr_t body_k_owner,
        uintptr_t body_v_owner,
        bool shadow_k,
        bool shadow_v);

llama_kv_tail_ownership_error llama_kv_tail_validate_layer_ownership(
        const llama_kv_tail_layer_ownership & ownership);

std::vector<llama_kv_tail_slot_run> llama_kv_tail_contiguous_slot_runs(
        const std::vector<int32_t> & slots);
