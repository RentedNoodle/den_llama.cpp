// den_persistent_work.h — Work queue builder for persistent kernel
// Generates work items for a complete transformer forward pass.
#pragma once
#include "den_persistent_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEN_SLOT_TOKEN_EMBD   0
#define DEN_SLOT_OUTPUT_NORM  1
#define DEN_SLOT_OUTPUT       2

#define DEN_SUB_INPUT_LN     20
#define DEN_SUB_POST_ATTN_LN 21
#define DEN_SUB_PRE_MLP_LN   22
#define DEN_SUB_POST_MLP_LN  23
#define DEN_SUB_ATTN_Q        1
#define DEN_SUB_ATTN_K        2
#define DEN_SUB_ATTN_V        3
#define DEN_SUB_ATTN_O        4
#define DEN_SUB_ATTN_Q_NORM   5
#define DEN_SUB_ATTN_K_NORM   6
#define DEN_SUB_GDN_QKV      24
#define DEN_SUB_GDN_A        12
#define DEN_SUB_GDN_B        25
#define DEN_SUB_GDN_Z        13
#define DEN_SUB_GDN_OUT      14
#define DEN_SUB_GDN_NORM     19
#define DEN_SUB_GDN_A_LOG    15
#define DEN_SUB_GDN_DT       16
#define DEN_SUB_GDN_CONV     17
#define DEN_SUB_GDN_N1       20
#define DEN_SUB_GDN_N2       21
#define DEN_SUB_MOE_ROUTER    7
#define DEN_SUB_MOE_GATE_UP  11
#define DEN_SUB_MOE_DOWN     26
#define DEN_SUB_SHARED_GATE  27
#define DEN_SUB_SHARED_UP    28
#define DEN_SUB_SHARED_DOWN  29
#define DEN_SUB_SHARED_GATE_W 30
#define DEN_SUB_MLP_GATE      8
#define DEN_SUB_MLP_UP        9
#define DEN_SUB_MLP_DOWN     10
#define DEN_LAYER_STRIDE     32

static inline int layer_base(int layer) {
    return 3 + layer * 32;
}

static inline int pk_find_tensor(
    const uint32_t* slots, int n_tensors, uint32_t target_slot)
{
    for (int i = 0; i < n_tensors; i++) {
        if (slots[i] == target_slot)
            return i;
    }
    return -1;
}

// Look up GPU weight or tile pointer by slot
static inline const void* pk_lookup_weight(
    const uint32_t* slots, int n,
    const void** d_weights, const void** d_tiles,
    uint32_t target_slot)
{
    int ti = pk_find_tensor(slots, n, target_slot);
    if (ti < 0) return NULL;
    if (d_tiles && d_tiles[ti]) return d_tiles[ti];
    if (d_weights && d_weights[ti]) return d_weights[ti];
    return NULL;
}

// Look up BF16 norm weight pointer by slot, sets is_bf16 flag
static inline const void* pk_lookup_norm(
    const uint32_t* slots, int n,
    const void** d_weights,
    uint32_t target_slot, unsigned* is_bf16)
{
    int ti = pk_find_tensor(slots, n, target_slot);
    if (ti < 0 || !d_weights || !d_weights[ti]) return NULL;
    *is_bf16 = 1;
    return d_weights[ti];
}

// ── Work queue builder ────────────────────────────────────────────────
inline int pk_build_forward_work(
    pk_work_queue_t* queue,
    const void** d_weights, const void** d_tiles,
    const uint32_t* tensor_slot, const int* tensor_N, const int* tensor_K,
    int n_tensors,
    int token_id,
    const void* d_embedding,
    float* d_hidden, float* d_scratch, float* d_logits,
    float* d_gdn_state, float* d_k_cache, float* d_v_cache,
    int* d_seq_lens,
    int H, int V, int L,
    float eps, float theta,
    int arch, int fai,
    int seq_pos,
    int nh, int nkv, int hd, int nr,
    int nvh, int kd, int vd)
{
    if (!queue || !d_hidden || !d_scratch || !d_logits) return -1;

    void* q_items = queue->items;

    // Helper: find tensor slot quick
    #define SLOT_OF(l, s) ((uint32_t)(layer_base(l) + s))

    // Enqueue one work item via mapped memory (direct write, no cudaMemcpy)
    // NOTE: macro params prefixed with w_ to avoid MSVC macro/struct member conflicts
    #define ENQ(w_type, w_token, w_layer, w_in, w_out, w_w, w_n, w_k, w_nrm, w_eps, w_fl) do { \
        uint32_t _t = queue->tail.load(cuda::std::memory_order_acquire); \
        if (_t >= PK_MAX_WORK_ITEMS) return -1; \
        pk_work_item_t _wi = {0}; \
        _wi.type = (uint32_t)(w_type); \
        _wi.token_id = (uint32_t)(w_token); \
        _wi.layer = (uint32_t)(w_layer); \
        _wi.flags = (uint32_t)(w_fl); \
        _wi.in_ptr = (uint64_t)(uintptr_t)(w_in); \
        _wi.out_ptr = (uint64_t)(uintptr_t)(w_out); \
        _wi.weight_ptr = (uint64_t)(uintptr_t)(w_w); \
        _wi.norm_ptr = (uint64_t)(uintptr_t)(w_nrm); \
        _wi.N = (uint32_t)(w_n); \
        _wi.K = (uint32_t)(w_k); \
        _wi.eps = (float)(w_eps); \
        ((pk_work_item_t*)q_items)[_t] = _wi; \
        queue->tail.store(_t + 1, cuda::std::memory_order_release); \
    } while(0)

    // ── 1. Embedding ─────────────────────────────────────────────────
    ENQ(PK_WORK_EMBED, token_id, 0,
        d_embedding, d_hidden, NULL, H, 0, NULL, 0.0f, 0);

    // ── 2. Process each layer ───────────────────────────────────────
    int gdn_layer_idx = 0;

    for (int layer = 0; layer < L; layer++) {
        int is_attn = (fai > 0) && (layer % fai == fai - 1);
        int lb = layer_base(layer);

        // Pre-norm weight lookup
        unsigned nflags = 0;
        const void* n1 = pk_lookup_norm(tensor_slot, n_tensors, d_weights,
            SLOT_OF(layer, DEN_SUB_INPUT_LN), &nflags);

        if (is_attn) {
            // ═══ ATTENTION LAYER ═══
            ENQ(PK_WORK_RMS_NORM, token_id, layer,
                d_hidden, d_scratch, NULL, H, 0, n1, eps, nflags);

            const void* wq = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_ATTN_Q));
            if (wq) ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                d_scratch, d_scratch, wq, nh * hd * 2, H, NULL, 0.0f, 0);

            float* k_buf = d_scratch + (size_t)nh * hd * 2;
            const void* wk = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_ATTN_K));
            if (wk) ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                d_scratch, k_buf, wk, nkv * hd, H, NULL, 0.0f, 0);

            float* v_buf = k_buf + (size_t)nkv * hd;
            const void* wv = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_ATTN_V));
            if (wv) ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                d_scratch, v_buf, wv, nkv * hd, H, NULL, 0.0f, 0);

            // RoPE
            unsigned rope_flags = (unsigned)nkv | ((unsigned)seq_pos << 16) | ((unsigned)nr << 24);
            ENQ(PK_WORK_ROPE, token_id, layer,
                d_scratch, NULL, k_buf, nh, hd, NULL, theta, rope_flags);

            // Attention
            float* lkc = d_k_cache + (size_t)layer * 2048 * nkv * hd;
            float* lvc = d_v_cache + (size_t)layer * 2048 * nkv * hd;
            float* attn_out = d_scratch + (size_t)H * 2;
            int sl = d_seq_lens ? d_seq_lens[layer] : 1;
            unsigned attn_flags = (unsigned)nkv | ((unsigned)(sl > 0 ? sl : 1) << 16);
            float ascale = 1.0f / sqrtf((float)hd);
            ENQ(PK_WORK_ATTN, token_id, layer,
                d_scratch, attn_out, lkc, nh, hd, lvc, ascale, attn_flags);

            // O projection
            const void* wo = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_ATTN_O));
            if (wo) ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                attn_out, d_scratch, wo, H, H, NULL, 0.0f, 0);

            // Residual
            ENQ(PK_WORK_ADD, token_id, layer,
                d_scratch, d_hidden, NULL, H, 0, NULL, 0.0f, 0);

        } else if (arch == 2) {
            // ═══ MoE LAYER ═══
            ENQ(PK_WORK_RMS_NORM, token_id, layer,
                d_hidden, d_scratch, NULL, H, 0, n1, eps, nflags);

            const void* router_w = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_MOE_ROUTER));
            if (router_w) {
                ENQ(PK_WORK_MOE_ROUTE, token_id, layer,
                    d_scratch, d_logits, router_w, 256, H, NULL, 0.0f, 0);
            }

            // MoE expert FFN: gate+up projection → SiLU → down projection
            const void* w_gate_up = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_MOE_GATE_UP));
            const void* w_down = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_MOE_DOWN));
            if (w_gate_up && w_down) {
                int ffn_size = H * 4; // approximate — lookup from tensor dims
                int gu_ti = pk_find_tensor(tensor_slot, n_tensors,
                    SLOT_OF(layer, DEN_SUB_MOE_GATE_UP));
                if (gu_ti >= 0) {
                    int n_experts = 256;
                    int total_ffn_2 = tensor_N[gu_ti];
                    if (total_ffn_2 > 0 && n_experts > 0)
                        ffn_size = total_ffn_2 / (n_experts * 2);
                }
                int n_exp = 256; // default — MoE models typically have 256 experts
                // NOTE: top experts are determined at runtime by MoE router.
                // For Phase 1, use placeholder expert 0 and 1.
                (void)n_exp;
                // Expert 0
                pk_work_item_t _wi_exp0 = {0};
                _wi_exp0.type = PK_WORK_MOE_EXPERT;
                _wi_exp0.token_id = (uint32_t)token_id;
                _wi_exp0.layer = (uint32_t)layer;
                _wi_exp0.in_ptr = (uint64_t)(uintptr_t)d_scratch;
                _wi_exp0.out_ptr = (uint64_t)(uintptr_t)d_scratch;
                _wi_exp0.weight_ptr = (uint64_t)(uintptr_t)w_gate_up;
                _wi_exp0.norm_ptr = (uint64_t)(uintptr_t)w_down;
                _wi_exp0.N = (uint32_t)ffn_size;
                _wi_exp0.K = (uint32_t)H;
                _wi_exp0.expert_ids[0] = 0; // expert 0
                _wi_exp0.expert_weights[0] = 0.5f; // weight
                {
                    uint32_t _t = queue->tail.load(cuda::std::memory_order_acquire);
                    if (_t >= PK_MAX_WORK_ITEMS) return -1;
                    ((pk_work_item_t*)q_items)[_t] = _wi_exp0;
                    queue->tail.store(_t + 1, cuda::std::memory_order_release);
                }
                // Expert 1
                pk_work_item_t _wi_exp1 = _wi_exp0;
                _wi_exp1.expert_ids[0] = 1;
                _wi_exp1.expert_weights[0] = 0.5f;
                {
                    uint32_t _t = queue->tail.load(cuda::std::memory_order_acquire);
                    if (_t >= PK_MAX_WORK_ITEMS) return -1;
                    ((pk_work_item_t*)q_items)[_t] = _wi_exp1;
                    queue->tail.store(_t + 1, cuda::std::memory_order_release);
                }
                // Weighted sum (simple: average the two expert outputs)
                // stored in d_scratch by the expert items, then add to hidden
            }

            // Shared expert FFN (if present)
            const void* w_shared_gate = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_SHARED_GATE));
            const void* w_shared_up = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_SHARED_UP));
            const void* w_shared_down = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_SHARED_DOWN));
            if (w_shared_gate && w_shared_up && w_shared_down) {
                int shared_ffn = H * 4;
                ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                    d_scratch, d_scratch, w_shared_gate, shared_ffn, H, NULL, 0.0f, 0);
                ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                    d_scratch, d_scratch, w_shared_up, shared_ffn, H, NULL, 0.0f, 0);
                ENQ(PK_WORK_SILU, token_id, layer,
                    d_scratch, d_scratch, NULL, shared_ffn, 0, NULL, 0.0f, 0);
                ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                    d_scratch, d_scratch, w_shared_down, H, shared_ffn, NULL, 0.0f, 0);
            }

            // Residual (d_scratch has expert output after last FFN)
            ENQ(PK_WORK_ADD, token_id, layer,
                d_scratch, d_hidden, NULL, H, 0, NULL, 0.0f, 0);

        } else if (arch == 3) {
            // ═══ DENSE MLP LAYER (non-GDN, non-attention) ═══
            ENQ(PK_WORK_RMS_NORM, token_id, layer,
                d_hidden, d_scratch, NULL, H, 0, n1, eps, nflags);

            int ffn = H * 4; // default 4× hidden
            const void* w_gate = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_MLP_GATE));
            const void* w_up = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_MLP_UP));
            const void* w_down = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_MLP_DOWN));
            if (w_gate) ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                d_scratch, d_scratch, w_gate, ffn, H, NULL, 0.0f, 0);
            if (w_up)   ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                d_scratch, d_scratch, w_up, ffn, H, NULL, 0.0f, 0);
            ENQ(PK_WORK_SILU, token_id, layer,
                d_scratch, d_scratch, NULL, ffn, 0, NULL, 0.0f, 0);
            if (w_down) ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                d_scratch, d_scratch, w_down, H, ffn, NULL, 0.0f, 0);
            ENQ(PK_WORK_ADD, token_id, layer,
                d_scratch, d_hidden, NULL, H, 0, NULL, 0.0f, 0);

        } else {
            // ═══ GDN LAYER ═══
            ENQ(PK_WORK_RMS_NORM, token_id, layer,
                d_hidden, d_scratch, NULL, H, 0, n1, eps, nflags);

            // GDN SSM: find weight pointers
            const void* w_qkv = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_GDN_QKV));
            const void* w_a = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_GDN_A));
            const void* w_b = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_GDN_B));
            const void* w_z = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_GDN_Z));
            // a_log and dt may be 1D — lookup fails unless uploaded as 2D.
            const void* w_a_log = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_GDN_A_LOG));
            const void* w_dt = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_GDN_DT));

            float* q_buf = d_scratch + H;
            float* lgdn = d_gdn_state + (size_t)gdn_layer_idx * nvh * kd * vd;
            unsigned gdn_flags = (unsigned)vd | ((unsigned)gdn_layer_idx << 16);

            // Pack all pointers into GDN SSM work item
            uint64_t s_ptr = (uint64_t)(uintptr_t)(lgdn ? lgdn : d_scratch);
            uint64_t wa_ptr = (uint64_t)(uintptr_t)(w_a ? w_a : d_scratch);
            uint64_t wb_ptr = (uint64_t)(uintptr_t)(w_b ? w_b : d_scratch);
            uint64_t wz_ptr = (uint64_t)(uintptr_t)(w_z ? w_z : d_scratch);
            uint64_t al_ptr = (uint64_t)(uintptr_t)(w_a_log);
            uint64_t dt_ptr = (uint64_t)(uintptr_t)(w_dt);

            {
                uint32_t _t = queue->tail.load(cuda::std::memory_order_acquire);
                if (_t >= PK_MAX_WORK_ITEMS) return -1;
                pk_work_item_t _wi = {0};
                _wi.type = PK_WORK_GDN_SSM;
                _wi.token_id = (uint32_t)token_id;
                _wi.layer = (uint32_t)layer;
                _wi.flags = gdn_flags;
                _wi.in_ptr = (uint64_t)(uintptr_t)d_scratch;    // normed input after RMSNorm
                _wi.out_ptr = (uint64_t)(uintptr_t)d_scratch;   // SSM output (reuse scratch)
                _wi.weight_ptr = (uint64_t)(uintptr_t)w_qkv;    // W_qkv
                _wi.norm_ptr = (uint64_t)(uintptr_t)q_buf;      // scratch buffer after H
                _wi.N = (uint32_t)nvh;
                _wi.K = (uint32_t)kd;
                memcpy(&_wi.expert_weights[0], &s_ptr, sizeof(uint64_t));
                memcpy(&_wi.expert_weights[2], &wa_ptr, sizeof(uint64_t));
                memcpy(&_wi.expert_weights[4], &wb_ptr, sizeof(uint64_t));
                memcpy(&_wi.expert_weights[6], &wz_ptr, sizeof(uint64_t));
                memcpy(&_wi.expert_ids[0], &al_ptr, sizeof(uint64_t));
                memcpy(&_wi.expert_ids[2], &dt_ptr, sizeof(uint64_t));
                ((pk_work_item_t*)q_items)[_t] = _wi;
                queue->tail.store(_t + 1, cuda::std::memory_order_release);
            }

            // Output projection (note: q_buf now has SSM output)
            const void* w_out = pk_lookup_weight(tensor_slot, n_tensors,
                d_weights, d_tiles, SLOT_OF(layer, DEN_SUB_GDN_OUT));
            if (w_out) ENQ(PK_WORK_GEMV_BF16, token_id, layer,
                (const float*)q_buf, q_buf, w_out, H, vd * nvh, NULL, 0.0f, 0);

            // Residual
            ENQ(PK_WORK_ADD, token_id, layer,
                q_buf, d_hidden, NULL, H, 0, NULL, 0.0f, 0);

            gdn_layer_idx++;
        }
    }

    // ── 3. Final RMSNorm + LM Head ───────────────────────────────
    unsigned on_flags = 0;
    const void* on_w = pk_lookup_norm(tensor_slot, n_tensors, d_weights,
        (uint32_t)DEN_SLOT_OUTPUT_NORM, &on_flags);
    ENQ(PK_WORK_RMS_NORM, token_id, L,
        d_hidden, d_scratch, NULL, H, 0, on_w, eps, on_flags);

    const void* lm_w = pk_lookup_weight(tensor_slot, n_tensors,
        d_weights, d_tiles, (uint32_t)DEN_SLOT_OUTPUT);
    if (!lm_w) lm_w = d_embedding;
    ENQ(PK_WORK_LM_HEAD, token_id, L,
        d_scratch, d_logits, lm_w, V, H, NULL, 0.0f, 0);

    return 0;

    #undef ENQ
    #undef SLOT_OF
}

#ifdef __cplusplus
}
#endif
