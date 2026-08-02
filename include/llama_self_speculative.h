#pragma once

#include "llama.h"
#include <vector>
#include <cstdint>

// ── Self-Speculative Decoding ───────────────────────────────────────────────
//
// Uses the first N layers of the main model as a draft model (no separate
// model file required).  The draft runs ~3x faster than the full model and
// produces candidate tokens.  The full model then verifies all candidates
// in a single forward pass.
//
// Architecture:
//   1. Draft phase:  run layers [0, draft_layers) to get draft_logits
//   2. Sample N candidate tokens from draft_logits
//   3. Verify phase: run all layers with KV cache to get target_logits
//   4. Rejection sampling: accept/reject each candidate against target
//   5. Reroll: for the first rejected token, re-sample from correct dist
//
// Adaptive: automatically adjusts draft length based on acceptance rate.
// Target: 60-80% acceptance rate → 2-3x throughput gain.

struct llama_self_speculative_params {
    int32_t   draft_layers  = -1;   // number of draft layers (negative = auto)
    int32_t   max_draft     = 5;    // max tokens to draft per step
    int32_t   min_draft     = 1;    // min tokens to draft per step
    float     target_accept = 0.7f; // target acceptance rate
    int32_t   window        = 100;  // adaptive window for acceptance tracking
    bool      enabled       = true;
};

struct llama_self_speculative {
    // Parameters
    llama_self_speculative_params params;

    // State
    int32_t current_draft_len = 3;
    std::vector<float> recent_acceptance; // acceptance rates over last `window` steps

    // KV cache state for rollback on rejection
    std::vector<uint32_t> kv_checkpoints;

    // Stats
    uint64_t n_draft_tokens  = 0;
    uint64_t n_accepted      = 0;
    uint64_t n_rejected      = 0;
    uint64_t n_rerolled      = 0;

    // ── Init ─────────────────────────────────────────────────────────────
    void init(const struct llama_model * /*model*/, int32_t n_layers) {
        if (params.draft_layers < 0) {
            // Auto: draft = min(first 1/3 of layers, 16 layers)
            params.draft_layers = std::max(1, std::min(n_layers / 3, 16));
        }
        current_draft_len = std::max(params.min_draft,
            std::min(params.max_draft, params.draft_layers / 2));
        recent_acceptance.reserve(params.window);
    }

    // ── Adaptive draft length ────────────────────────────────────────────
    void update_acceptance(float rate) {
        if (recent_acceptance.size() >= (size_t)params.window) {
            recent_acceptance.erase(recent_acceptance.begin());
        }
        recent_acceptance.push_back(rate);

        float avg = 0.0f;
        for (float r : recent_acceptance) avg += r;
        avg /= (float)recent_acceptance.size();

        // Adjust draft length to hit target acceptance rate
        if (avg > params.target_accept + 0.1f) {
            current_draft_len = std::min(params.max_draft, current_draft_len + 1);
        } else if (avg < params.target_accept - 0.1f) {
            current_draft_len = std::max(params.min_draft, current_draft_len - 1);
        }
    }

    // ── Stats ────────────────────────────────────────────────────────────
    float acceptance_rate() const {
        uint64_t total = n_accepted + n_rejected;
        return total > 0 ? (float)n_accepted / (float)total : 0.0f;
    }

    float speedup() const {
        // Speedup = (draft time + verify time) / (full model time)
        // Approximation: draft runs at (1/draft_ratio) speed
        // Verify runs at ~full speed but processes multiple tokens at once
        float ar = acceptance_rate();
        if (ar <= 0.0f) return 1.0f;
        // Expected accepted tokens per step = current_draft_len * ar
        float expected = (float)current_draft_len * ar;
        return expected > 0.5f ? expected : 1.0f;
    }
};

// ── Self-Speculative Decode ──────────────────────────────────────────────
//
// Runs a single self-speculative decode step.
// Returns the number of tokens accepted (written to out_tokens).
// Returns 0 on failure.
//
int32_t llama_self_speculative_decode(
    struct llama_context * ctx,
    struct llama_self_speculative * spec,
    const llama_token * input_tokens,
    int32_t n_input,
    llama_token * out_tokens,
    int32_t max_out);
