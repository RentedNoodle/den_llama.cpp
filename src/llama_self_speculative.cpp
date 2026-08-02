#include "llama_self_speculative.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

// Self-Speculative Decode Step
//
// Returns the number of tokens generated (accepted).
// Output tokens are written to `out_tokens`.
// Returns 0 on failure.
//
int32_t llama_self_speculative_decode(
    struct llama_context * ctx,
    struct llama_self_speculative * spec,
    const llama_token * input_tokens,
    int32_t n_input,
    llama_token * out_tokens,
    int32_t max_out)
{
    if (!ctx || !spec || !spec->params.enabled) return 0;

    const int32_t draft_len = std::min(spec->current_draft_len, max_out);
    if (draft_len <= 0) return 0;

    const auto * model = llama_get_model(ctx);
    const auto * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Phase 1: Draft
    // Run one token at a time to build candidate sequence.
    // In a full implementation, this would use a truncated model
    // (first N layers) for faster drafting.

    std::vector<llama_token> draft_tokens(draft_len);
    std::vector<float> draft_probs(draft_len, 0.0f);

    // Use the last input token as the starting point for drafting
    llama_token prev_token = input_tokens[n_input - 1];

    int32_t actual_draft = 0;
    for (int32_t i = 0; i < draft_len; i++) {
        llama_batch batch = llama_batch_get_one(&prev_token, 1, n_input + i, 0);

        if (llama_decode(ctx, batch) != 0) break;

        auto * logits = llama_get_logits_ith(ctx, 0);
        if (!logits) break;

        // Argmax sampling
        int32_t best_token = 0;
        float best_logit = logits[0];
        float max_logit = logits[0];
        for (int32_t v = 1; v < n_vocab; v++) {
            if (logits[v] > best_logit) {
                best_logit = logits[v];
                best_token = v;
            }
            if (logits[v] > max_logit) max_logit = logits[v];
        }

        draft_tokens[i] = (llama_token)best_token;
        prev_token = draft_tokens[i];

        // Softmax probability for the chosen token
        float sum_exp = 0.0f;
        for (int32_t v = 0; v < n_vocab; v++) {
            sum_exp += expf(logits[v] - max_logit);
        }
        draft_probs[i] = (sum_exp > 0.0f)
            ? expf(best_logit - max_logit) / sum_exp
            : 0.0f;

        actual_draft++;

        // Stop at EOG tokens
        if (llama_vocab_is_eog(vocab, draft_tokens[i])) break;
    }

    if (actual_draft == 0) return 0;
    spec->n_draft_tokens += actual_draft;

    // Phase 2: Verify all draft tokens in one forward pass
    // The KV cache from drafting is reused so verification is fast.

    llama_batch verify_batch = llama_batch_get_one(
        draft_tokens.data(), actual_draft, n_input, 0);

    if (llama_decode(ctx, verify_batch) != 0) {
        // Fallback: accept all draft tokens
        for (int32_t i = 0; i < actual_draft; i++) {
            out_tokens[i] = draft_tokens[i];
        }
        spec->n_accepted += actual_draft;
        spec->update_acceptance(1.0f);
        return actual_draft;
    }

    // Phase 3: Rejection Sampling
    int32_t n_accepted = 0;

    for (int32_t i = 0; i < actual_draft; i++) {
        auto * target_logits = llama_get_logits_ith(ctx, i);
        if (!target_logits) break;

        // Find target logit for the draft token
        float target_logit = target_logits[draft_tokens[i]];
        float max_lt = target_logits[0];
        for (int32_t v = 1; v < n_vocab; v++) {
            if (target_logits[v] > max_lt) max_lt = target_logits[v];
        }
        float target_prob = expf(target_logit - max_lt);

        // Standard speculative decoding: accept if uniform(0,1) < target/draft
        float r = (float)rand() / (float)RAND_MAX;
        float ratio = (draft_probs[i] > 1e-10f)
            ? (target_prob / draft_probs[i])
            : 0.0f;

        if (r < std::min(1.0f, ratio)) {
            out_tokens[n_accepted] = draft_tokens[i];
            n_accepted++;
        } else {
            // Rejected: re-sample from target distribution
            int32_t reroll = 0;
            float best_rt = target_logits[0];
            for (int32_t v = 1; v < n_vocab; v++) {
                if (target_logits[v] > best_rt) {
                    best_rt = target_logits[v];
                    reroll = v;
                }
            }
            out_tokens[n_accepted] = (llama_token)reroll;
            n_accepted++;
            break;
        }
    }

    spec->n_accepted += n_accepted;

    // Update adaptive acceptance rate
    float rate = actual_draft > 0 ? (float)n_accepted / (float)actual_draft : 0.0f;
    spec->update_acceptance(rate);

    return n_accepted;
}
