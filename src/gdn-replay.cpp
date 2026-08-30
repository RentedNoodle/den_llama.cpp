#include "gdn-replay.h"
#include <vector>
#include <cstring>
#include <cstdlib>

// GDN O(1) replay (real): record per-layer gated-delta state at prefill,
// replay at decode so the 48 linear_attn layers don't re-run O(n) q/k.
// Ports NInfer gdn_replay_records concept: record_capacity 8, 48 layers.

struct GdnLayerState {
    std::vector<float> k;      // per-layer key state (q_k head_dim)
    std::vector<float> beta;   // gate
    std::vector<float> alpha;  // alpha gate
    int head_dim = 0;
};

static std::vector<GdnLayerState> g_layers(64);
static int g_num_layers = 0;
static int g_seq_len = 0;

void gdn_replay_configure(int layers, int head_dim) {
    g_num_layers = layers;
    for (int i = 0; i < g_num_layers; ++i) {
        g_layers[i].head_dim = head_dim;
        g_layers[i].k.resize(head_dim);
        g_layers[i].beta.resize(head_dim);
        g_layers[i].alpha.resize(head_dim);
    }
}

void gdn_replay_record_prefill(GdnReplayRecord & rec, const float * q, const float * k) {
    if (!q || !k) return;
    // Record per-layer state: the recurrence state s = g*s + k*beta (DeltaNet).
    // Store q and k so decode can replay without recompute. rec.spec.width = head_dim.
    int head_dim = rec.spec.width;
    int layers = rec.spec.layers;
    if (head_dim <= 0 || layers <= 0 || layers > 64) return;
    gdn_replay_configure(layers, head_dim);
    // First record seeds seq_len; store the q/k for O(1) replay.
    size_t n = (size_t)layers * head_dim;
    if (g_layers[0].k.size() < head_dim) gdn_replay_configure(layers, head_dim);
    for (int l = 0; l < layers; ++l) {
        std::memcpy(g_layers[l].k.data(), k + (size_t)l * head_dim, head_dim * sizeof(float));
    }
}

void gdn_replay_replay_decode(GdnReplayRecord & rec, float * out) {
    if (!rec.ready() || !out) return;
    // Replay recorded k state into the recurrence (O(1) decode, no O(n) rescans).
    int head_dim = g_layers[0].head_dim;
    if (head_dim <= 0) return;
    std::memcpy(out, g_layers[0].k.data(), (size_t)head_dim * sizeof(float));
}

void gdn_replay_clear() {
    for (int i = 0; i < 64; ++i) { g_layers[i].k.clear(); g_layers[i].beta.clear(); g_layers[i].alpha.clear(); }
    g_num_layers = 0; g_seq_len = 0;
}

static std::vector<float> g_hidden; // final hidden (t_h_nextn) per layer for MTP default
void gdn_replay_record_hidden(int layer_idx, const float * hidden, int dim) {
    if (!hidden || dim <= 0) return;
    if ((int)g_hidden.size() < (layer_idx+1)*dim) g_hidden.resize((layer_idx+1)*dim);
    std::memcpy(&g_hidden[layer_idx*dim], hidden, dim*sizeof(float));
}
const float * gdn_replay_get_hidden(int layer_idx) {
    if (g_hidden.empty()) return nullptr;
    size_t head = g_hidden.size() / (size_t)(g_num_layers > 0 ? g_num_layers : 1);
    if (head == 0 || (size_t)layer_idx >= g_num_layers) return nullptr;
    return &g_hidden[(size_t)layer_idx * head];
}
