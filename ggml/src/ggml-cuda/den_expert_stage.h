// den_expert_stage.h - L3-resident CPU staging tier for expert offload (Den)
//
// When MoE experts are host-resident (--cpu-moe / --n-cpu-moe), the GPU's H2D
// copies for the active expert slices are served from a small pinned
// (cudaMallocHost) double-buffered staging area that is pre-filled by a
// dedicated CPU staging thread.  This gives the DMA engine a hot, pinned
// source (CPU L3 / low-latency pinned memory) instead of bouncing 4KB pages
// out of the large unpinned host expert buffer.
//
// The staging thread runs a per-layer Markov/expert-locality predictor:
//   - it learns the split key sequence (gate_up L -> down L -> gate_up L+1 ...)
//   - after each ids read-back it stages the NEXT split's active experts,
//     predicting their ids from the previous occurrence of that key (temporal
//     locality) or the current layer's ids (first-cut heuristic).
//
// This header exposes two things:
//   1. the interface struct registered into ggml-base (ggml-backend.cpp) so the
//      scheduler can call the staging functions without a link dependency on the
//      CUDA backend, and
//   2. the extern "C" implementation API (rule 7.9: no OMMA/tensor-core asm here).
#ifndef DEN_EXPERT_STAGE_H
#define DEN_EXPERT_STAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// GGML_API may not be defined when this header is included from the CUDA
// backend (which does not define GGML_BUILD).  Fall back to plain extern.
#ifndef GGML_API
#define GGML_API extern
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Interface installed by the CUDA backend into ggml-base via
// ggml_backend_den_stage_register().  ggml-backend.cpp calls these through the
// registered pointers, so ggml-base has no link dependency on ggml-cuda and the
// staging tier is a pure no-op when the CUDA backend does not register it.
typedef struct ggml_den_stage_iface {
    // Enable/disable (and lazily init) the staging tier.  Must be called before
    // the first submit.  disable() tears the thread + buffers down.
    void (*set_enabled)(bool enabled);
    // Stage the NEXT split's predicted active experts for `key`.
    //   key        - expert tensor name, e.g. "blk.0.ffn_gate_up_exps.weight"
    //   layer      - parsed layer index (informational; predictor re-parses key)
    //   ids        - active expert ids observed for the current split
    //   n_ids      - number of ids
    //   host_src   - host pointer of the full expert tensor (per-expert stride)
    //   expert_size- per-expert stride in host_src (== tensor nb[2])
    // Returns 0 on success, -1 if staging is disabled/not inited.
    int (*submit)(const char * key, int layer, const int32_t * ids, int n_ids,
                  const void * host_src, size_t expert_size);
    // Return a pointer to the staged copy of expert `first_id` if every expert in
    // [first_id, last_id] is resident in the current staging buffer for `key`
    // (contiguous, in ascending id order), else NULL.  The caller then copies
    // (last_id - first_id + 1) * expert_size bytes from the returned pointer.
    const void * (*find_span)(const char * key, int32_t first_id, int32_t last_id);
    // Block until the in-flight staging job completes (bounded).
    void (*wait_layer)(void);
    // L3-residency probe: timed re-read of a filled staging buffer.  Returns GB/s.
    double (*probe)(void);
    // Fetch hit/miss/bytes counters.
    void (*stats)(size_t * hits, size_t * misses, size_t * bytes_staged);
    // Stop the staging thread and free the pinned buffers.
    void (*shutdown)(void);
} ggml_den_stage_iface;

// Called by the CUDA backend (den_expert_stage.cu) to install its implementation.
GGML_API void ggml_backend_den_stage_register(const ggml_den_stage_iface * iface);

// ---- extern "C" implementation API (implemented in den_expert_stage.cu) ----

void   den_expert_stage_set_enabled(bool enabled);
void   den_expert_stage_init(size_t bytes_total, int n_buffers);
int    den_expert_stage_submit(const char * key, int layer, const int32_t * ids, int n_ids,
                               const void * host_src, size_t expert_size);
const void * den_expert_stage_find(const char * key, int32_t expert_id);
const void * den_expert_stage_find_span(const char * key, int32_t first_id, int32_t last_id);
void   den_expert_stage_wait_layer(void);
double den_expert_stage_probe(void);
void   den_expert_stage_stats(size_t * hits, size_t * misses, size_t * bytes_staged);
void   den_expert_stage_shutdown(void);

// Installs the implementation into ggml-base.  Called by the CUDA backend
// registration (ggml_backend_cuda_reg).
void   den_expert_stage_register_ggml(void);

#ifdef __cplusplus
}
#endif

#endif // DEN_EXPERT_STAGE_H
