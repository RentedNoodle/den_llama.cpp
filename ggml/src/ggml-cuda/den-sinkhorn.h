// den-sinkhorn.h — Sinkhorn rerank cache warmth (C-compatible)
// Include from both CUDA and C++ code.
// Bias expert logits toward previously-selected experts.
// Ratio: 0.7 current_logit + 0.3 previous_logit.
#pragma once
#include <cstring>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

#define DEN_SINKHORN_MAX_LAYERS 64
#define DEN_SINKHORN_MAX_EXPERTS 256
#define DEN_SINKHORN_TOP_K 8
#define DEN_SINKHORN_BIAS_FACTOR 0.30f

typedef struct {
    float prev_logits[DEN_SINKHORN_MAX_LAYERS][DEN_SINKHORN_MAX_EXPERTS];
    int   prev_active[DEN_SINKHORN_MAX_LAYERS][DEN_SINKHORN_TOP_K];
    int   n_layers, n_experts, initialized;
} den_sinkhorn_state_t;

static den_sinkhorn_state_t g_sinkhorn;

static inline int den_sinkhorn_enabled(void) {
    static int checked = 0, enabled = 1; // default on
    if (!checked) {
        const char * env = getenv("DEN_SINKHORN_RERANK");
        if (env && env[0] == '0') enabled = 0;
        checked = 1;
    }
    return enabled;
}

static inline void den_sinkhorn_init(int n_layers, int n_experts) {
    if (n_layers > DEN_SINKHORN_MAX_LAYERS || n_experts > DEN_SINKHORN_MAX_EXPERTS) return;
    memset(&g_sinkhorn, 0, sizeof(g_sinkhorn));
    g_sinkhorn.n_layers = n_layers;
    g_sinkhorn.n_experts = n_experts;
    g_sinkhorn.initialized = 1;
}

// Apply warmth bias to logits in-place. Call during graph build before softmax.
static inline void den_sinkhorn_bias_logits(float * logits, int layer, int n_experts) {
    if (!g_sinkhorn.initialized || layer >= g_sinkhorn.n_layers) return;
    const float * prev = g_sinkhorn.prev_logits[layer];
    for (int e = 0; e < n_experts; e++) {
        if (prev[e] > 0.0f)
            logits[e] += DEN_SINKHORN_BIAS_FACTOR * (prev[e] - logits[e]);
    }
}

// Store current logits after selection. Call after graph compute.
static inline void den_sinkhorn_store(float * logits, int * expert_ids, int layer, int n_experts, int top_k) {
    if (!g_sinkhorn.initialized || layer >= g_sinkhorn.n_layers) return;
    memcpy(g_sinkhorn.prev_logits[layer], logits, (size_t)n_experts * sizeof(float));
    for (int k = 0; k < top_k && k < DEN_SINKHORN_TOP_K; k++)
        g_sinkhorn.prev_active[layer][k] = expert_ids[k];
}

#ifdef __cplusplus
}
#endif
