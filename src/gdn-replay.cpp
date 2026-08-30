#include "gdn-replay.h"
#include <vector>
#include <cstring>
#include <cstdlib>

struct GdnReplayStore {
    std::vector<float> q_buf;
    std::vector<float> k_buf;
    int layers = 0;
    int seq_len = 0;
    int head_dim = 0;
};

static GdnReplayStore g_store;

void gdn_replay_record_prefill(GdnReplayRecord & rec, const float * q, const float * k) {
    // Record q/k for O(1) replay — store per-layer head state
    if (!q || !k) return;
    size_t bytes = rec.bytes;
    if (bytes == 0 || !rec.payload) return;
    // Stub: copy first tile as proof of plumbing
    size_t copy = std::min(bytes, size_t(4096));
    std::memcpy(rec.payload, q, copy);
    g_store.layers = rec.spec.layers;
}

void gdn_replay_replay_decode(GdnReplayRecord & rec, float * out) {
    if (!rec.ready() || !out) return;
    // Replay: copy recorded state back
    size_t copy = std::min(rec.bytes, size_t(4096));
    std::memcpy(out, rec.payload, copy);
}
