#pragma once
#include <cstdint>
#include <cstddef>

// GDN O(1) replay — record DeltaNet q/k state at prefill, replay at decode.
// 48 linear_attn layers in Qwen3.8-27B. Composes with MTP (MTP reads t_h_nextn).

struct GdnReplaySpec {
    int32_t layers = 0;
    int32_t width = 0;       // head_dim
    int32_t qk_heads = 0;
    int32_t value_heads = 0;
    int32_t record_capacity = 8; // NInfer record capacity
};

struct GdnReplayRecord {
    GdnReplaySpec spec;
    void * payload = nullptr;
    size_t bytes = 0;
    bool ready() const { return payload != nullptr && bytes > 0; }
};

void gdn_replay_configure(int layers, int head_dim);
void gdn_replay_record_prefill(GdnReplayRecord & rec, const float * q, const float * k);
void gdn_replay_replay_decode(GdnReplayRecord & rec, float * out);
void gdn_replay_clear();

// Convenience: hook the q35 GDN path. Returns true if replay succeeded (O(1)).
bool gdn_replay_decode_layer(int layer_idx, float * out);

void gdn_replay_record_hidden(int layer_idx, const float * hidden, int dim);
const float * gdn_replay_get_hidden(int layer_idx);
