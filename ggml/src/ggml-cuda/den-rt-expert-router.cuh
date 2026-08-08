// den-rt-expert-router.cuh — RT Core MoE Expert Router for den_llama.cpp
//
// Ported from Project Den dengine/include/den_rt_expert.h (2026-08-08)
//
// RT Cores (4th gen Blackwell, GB203) are dedicated BVH traversal units
// that run independently of SMs. During LLM inference they idle.
// This repurposes them for O(log N) expert routing instead of O(N) top-K sort.
//
// Concept:
//   1. Projected 2048D router weights → 3D via PCA (per layer, once)
//   2. Build BVH over 256 expert 3D points
//   3. Token hidden state projected → "ray" through BVH
//   4. RT Cores find top-16 candidates in O(log 256)
//   5. Exact 2048D dot products on 16 candidates → top-8
//
// Fallback (4 tiers):
//   Tier 0: OptiX RT Core (dynamic nvoptix.dll load, zero compile dep)
//   Tier 1: CUDA SW BVH (GPU, no OptiX)
//   Tier 2: Brute-force 3D NN (GPU, universal)
//   Tier 3: Exact GEMV (CPU, always correct)
//
// Quality gate: top-1 match + ≥80% overlap vs exact GEMV.
// Memory: ~640 KB total (40 layers × 511 nodes × 32B).
//
// Hardware: sm_120a GB203 (RTX 5070 Ti), 56 RT Cores @ 2.6 GHz.
// License: MIT (as part of den_llama.cpp)

#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define DEN_RT_EXPERT_MAX_LAYERS   64
#define DEN_RT_EXPERT_MAX_EXPERTS  256
#define DEN_RT_EXPERT_HIDDEN       2048
#define DEN_RT_EXPERT_TOP_K        8
#define DEN_RT_EXPERT_CANDIDATES   16
#define DEN_RT_EXPERT_LEAF_SIZE    4
#define DEN_RT_EXPERT_OVERLAP      0.80f

// BVH Node, 32 bytes, cache-line aligned
typedef struct {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    union {
        struct { int left_child, right_child; };
        struct { int prim_offset, prim_count; };
    };
} den_rt_expert_bvh_node_t;

// Per-layer state
typedef struct {
    float  projection[3 * DEN_RT_EXPERT_HIDDEN];
    void * d_expert_3d;
    float * h_expert_3d;
    void * d_bvh_nodes;
    int  * d_bvh_prim_indices;
    int  * d_bvh_scratch;
    int    n_bvh_nodes, bvh_capacity;
    int  * d_candidate_indices;
    float* d_candidate_dists;
    int  * h_candidate_indices;
    float* h_expert_weights;
    int    rt_core_accepted, rt_core_rejected;
    int    layer, n_experts, hidden_size, built;
} den_rt_expert_layer_t;

// Top-level router
typedef struct {
    den_rt_expert_layer_t layers[DEN_RT_EXPERT_MAX_LAYERS];
    int    n_layers;
    void * d_query_3d;
    void * d_gemv_check;
    void * optix_handle, * optix_context, * optix_module, * optix_pipeline;
    void * optix_accel[DEN_RT_EXPERT_MAX_LAYERS];
    int    optix_available;
    float  overlap_threshold;
    int    quality_gate_enabled;
    int    total_routed, total_fallback_gemv;
    int    initialized;
} den_rt_expert_router_t;

// Routing result
typedef struct {
    int   expert_ids[DEN_RT_EXPERT_TOP_K];
    float gate_weights[DEN_RT_EXPERT_TOP_K];
    int   n_active;
    int   rt_core_used;
    float overlap_with_gemv;
} den_rt_expert_route_result_t;

// API
int  den_rt_expert_router_init (den_rt_expert_router_t * router, int n_layers);
int  den_rt_expert_router_build (den_rt_expert_router_t * router, int layer,
                                  const uint16_t * router_weights, int n_experts, int hidden_size);
int  den_rt_expert_router_trace (den_rt_expert_router_t * router, int layer,
                                  const float * hidden, den_rt_expert_route_result_t * result);
void den_rt_expert_router_disable_quality_gate(den_rt_expert_router_t * router);
void den_rt_expert_router_get_stats(const den_rt_expert_router_t * router,
                                     int * total_routed, int * total_fallback,
                                     int * per_layer_accepted, int * per_layer_rejected);
void den_rt_expert_router_free  (den_rt_expert_router_t * router);

#ifdef __cplusplus
}
#endif
