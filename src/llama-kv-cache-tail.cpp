#include "llama-kv-cache-tail.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>

//
// Implementations for the KVarN-exact-tail function surface. Ported from
// beellama's llama-kv-cache-tail.cpp (only the subset den's KVarN foundation
// actually references).
//

std::vector<llama_kv_tail_slot_run> llama_kv_tail_contiguous_slot_runs(
        const std::vector<int32_t> & slots) {
    std::vector<llama_kv_tail_slot_run> runs;
    if (slots.empty()) {
        return runs;
    }

    runs.push_back({ 0, slots[0], 1 });
    for (size_t payload = 1; payload < slots.size(); ++payload) {
        auto & run = runs.back();
        if (int64_t(slots[payload]) == int64_t(slots[payload - 1]) + 1) {
            ++run.length;
        } else {
            runs.push_back({ uint32_t(payload), slots[payload], 1 });
        }
    }

#ifndef NDEBUG
    uint64_t n_payloads = 0;
    for (const auto & run : runs) {
        assert(run.length > 0);
        assert(run.payload_begin == n_payloads);
        n_payloads += run.length;
    }
    assert(n_payloads == slots.size());
#endif

    return runs;
}

uint32_t llama_kv_tail_packed_body_stride(uint64_t logical_rows, uint32_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("packed KV body alignment must be a non-zero power of two");
    }
    if (logical_rows > UINT64_MAX - (alignment - 1)) {
        throw std::overflow_error("packed KV body stride overflows uint64_t");
    }
    const uint64_t aligned = (logical_rows + alignment - 1)/alignment*alignment;
    if (aligned > uint64_t(UINT32_MAX)) {
        throw std::overflow_error("packed KV body stride overflows uint32_t");
    }
    return uint32_t(aligned);
}

llama_kv_tail_route_capability llama_kv_tail_select_route(
        const llama_kv_tail_route_requirements & requirements) {
    const auto missing_write = [&]() -> llama_kv_tail_operation {
        if (!requirements.write_k) return LLAMA_KV_TAIL_OP_WRITE_K;
        if (!requirements.write_v) return LLAMA_KV_TAIL_OP_WRITE_V;
        return LLAMA_KV_TAIL_OP_NONE;
    }();
    if (missing_write != LLAMA_KV_TAIL_OP_NONE) {
        return { false, LLAMA_KV_TAIL_ROUTE_NONE, missing_write };
    }
    if (requirements.native_attention) {
        return { true, LLAMA_KV_TAIL_ROUTE_NATIVE, LLAMA_KV_TAIL_OP_NONE };
    }
    const std::pair<bool, llama_kv_tail_operation> generic[] = {
        { requirements.gather_k,      LLAMA_KV_TAIL_OP_GATHER_K },
        { requirements.gather_v,      LLAMA_KV_TAIL_OP_GATHER_V },
        { requirements.body_score,    LLAMA_KV_TAIL_OP_BODY_SCORE },
        { requirements.body_value,    LLAMA_KV_TAIL_OP_BODY_VALUE },
        { requirements.exact_score,   LLAMA_KV_TAIL_OP_EXACT_SCORE },
        { requirements.exact_value,   LLAMA_KV_TAIL_OP_EXACT_VALUE },
        { requirements.generic_merge, LLAMA_KV_TAIL_OP_GENERIC_MERGE },
    };
    for (const auto & requirement : generic) {
        if (!requirement.first) {
            return { false, LLAMA_KV_TAIL_ROUTE_NONE, requirement.second };
        }
    }
    return { true, LLAMA_KV_TAIL_ROUTE_GENERIC, LLAMA_KV_TAIL_OP_NONE };
}

const char * llama_kv_tail_operation_name(llama_kv_tail_operation operation) {
    switch (operation) {
        case LLAMA_KV_TAIL_OP_NONE:             return "none";
        case LLAMA_KV_TAIL_OP_WRITE_K:          return "K body/exact write";
        case LLAMA_KV_TAIL_OP_WRITE_V:          return "V body/exact write";
        case LLAMA_KV_TAIL_OP_GATHER_K:         return "exact K gather";
        case LLAMA_KV_TAIL_OP_GATHER_V:         return "exact V gather";
        case LLAMA_KV_TAIL_OP_BODY_SCORE:       return "body K score";
        case LLAMA_KV_TAIL_OP_BODY_VALUE:       return "body V reduction";
        case LLAMA_KV_TAIL_OP_EXACT_SCORE:      return "exact K score";
        case LLAMA_KV_TAIL_OP_EXACT_VALUE:      return "exact V reduction";
        case LLAMA_KV_TAIL_OP_GENERIC_MERGE:    return "generic softmax merge";
        case LLAMA_KV_TAIL_OP_NATIVE_ATTENTION: return "native attached-tail attention";
    }
    return "unknown";
}

llama_kv_tail_ownership_error llama_kv_tail_validate_layer_ownership(
        const llama_kv_tail_layer_ownership & ownership) {
    if (ownership.shadow_k_owner != 0 && ownership.shadow_k_owner != ownership.body_k_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_SHADOW_K;
    }
    if (ownership.shadow_v_owner != 0 && ownership.shadow_v_owner != ownership.body_v_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_SHADOW_V;
    }
    if (ownership.graph_write_owner != ownership.body_k_owner ||
            (ownership.body_v_owner != 0 && ownership.graph_write_owner != ownership.body_v_owner)) {
        return LLAMA_KV_TAIL_OWNERSHIP_GRAPH_WRITE;
    }
    if (ownership.graph_read_owner != ownership.body_k_owner ||
            (ownership.body_v_owner != 0 && ownership.graph_read_owner != ownership.body_v_owner)) {
        return LLAMA_KV_TAIL_OWNERSHIP_GRAPH_READ;
    }
    if (ownership.state_k_owner != ownership.body_k_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_STATE_K;
    }
    if (ownership.body_v_owner != 0 && ownership.state_v_owner != ownership.body_v_owner) {
        return LLAMA_KV_TAIL_OWNERSHIP_STATE_V;
    }
    return LLAMA_KV_TAIL_OWNERSHIP_OK;
}

llama_kv_tail_layer_ownership llama_kv_tail_plan_layer_ownership(
        uint32_t layer_id,
        uintptr_t body_k_owner,
        uintptr_t body_v_owner,
        bool shadow_k,
        bool shadow_v) {
    return {
        layer_id,
        body_k_owner,
        body_v_owner,
        shadow_k ? body_k_owner : 0,
        shadow_v ? body_v_owner : 0,
        body_k_owner,
        body_k_owner,
        body_k_owner,
        body_v_owner,
        false,
        false,
    };
}
