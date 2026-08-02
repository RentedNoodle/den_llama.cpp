#pragma once

// Ported from BeeLlama: adaptive draft-max (DM) controller.
// Adjusts draft token budget based on acceptance rate and throughput.
// Requires DFlash or model-based drafter to be meaningful.

#include "common.h"
#include "log.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

static constexpr int SERVER_ADAPTIVE_DM_PROFIT_POSITIONS  = 128;
static constexpr int SERVER_ADAPTIVE_DM_PROFIT_DEPTHS     = SERVER_ADAPTIVE_DM_PROFIT_POSITIONS + 1;
static constexpr int SERVER_ADAPTIVE_DM_PROFIT_CANDIDATES = SERVER_ADAPTIVE_DM_PROFIT_DEPTHS + 1;

static inline int server_adaptive_dm_probe_n_max(int base_n_max, float probe_fraction) {
    if (base_n_max <= 0) return 0;
    return std::min(std::max(2, (int)(base_n_max * probe_fraction)), base_n_max);
}

static inline float server_adaptive_dm_required_fringe_for_n_max(
        int target_n_max, int base_n_max, float fringe_min, float fringe_max) {
    if (base_n_max <= 2 || target_n_max <= 2 || fringe_max <= fringe_min) return fringe_min;
    float t = (float)(std::clamp(target_n_max, 2, base_n_max) - 2) / (float)(base_n_max - 2);
    return std::clamp(fringe_min + (fringe_max - fringe_min) * t, fringe_min, fringe_max);
}

static inline bool server_adaptive_dm_uses_fringe_controller(common_speculative_dm_controller c) {
    return c == COMMON_SPECULATIVE_DM_CONTROLLER_FRINGE;
}

static inline bool server_adaptive_dm_uses_profit_controller(common_speculative_dm_controller c) {
    return c == COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT;
}

static inline const char * server_adaptive_dm_controller_name(common_speculative_dm_controller c) {
    switch (c) {
        case COMMON_SPECULATIVE_DM_CONTROLLER_FRINGE: return "fringe";
        case COMMON_SPECULATIVE_DM_CONTROLLER_PROFIT: return "profit";
    }
    return "unknown";
}

static inline void server_adaptive_dm_ewma_init_or_update(float & dst, float sample, float alpha, int32_t seen) {
    if (!std::isfinite(sample)) return;
    alpha = std::clamp(alpha, 0.01f, 1.0f);
    if (seen == 0) dst = sample;
    else dst += alpha * (sample - dst);
}

static inline int server_adaptive_dm_build_candidates(int base_n_max, int * out, int out_cap) {
    if (out_cap <= 0) return 0;
    int n = 0;
    auto add = [&](int c) {
        if (c < 0 || c > base_n_max || n >= out_cap) return;
        for (int i = 0; i < n; ++i) if (out[i] == c) return;
        out[n++] = c;
    };
    add(0);
    for (int c = 1; c <= std::min(base_n_max, SERVER_ADAPTIVE_DM_PROFIT_POSITIONS); ++c) add(c);
    std::sort(out, out + n);
    return n;
}

static inline float server_adaptive_dm_survival_expected_accept(
        const float * pos_ewma, const int32_t * pos_samples,
        int n_positions, int depth, int min_samples, bool * ready) {
    bool is_ready = true;
    float expected = 0.0f;
    if (depth <= 0) { if (ready) *ready = true; return 0.0f; }
    for (int pos = 0; pos < depth; ++pos) {
        if (pos >= n_positions || pos_samples[pos] < min_samples) { is_ready = false; break; }
        if (!std::isfinite(pos_ewma[pos])) { is_ready = false; break; }
        expected += std::clamp(pos_ewma[pos], 0.0f, 1.0f);
    }
    if (ready) *ready = is_ready;
    return is_ready ? expected : 0.0f;
}

static inline float server_adaptive_dm_score(float expected_out_tokens, float cycle_ms) {
    if (cycle_ms <= 0.0f || expected_out_tokens <= 0.0f
            || !std::isfinite(cycle_ms) || !std::isfinite(expected_out_tokens)) return 0.0f;
    return expected_out_tokens * 1000.0f / cycle_ms;
}
