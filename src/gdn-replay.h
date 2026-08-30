#pragma once
#include <cstdint>
#include <cstddef>
// Minimal GDN replay stub — O(1) DeltaNet via prefill record / decode replay
// Ported from NInfer gdn_replay_records.h — will be fleshed incrementally
struct GdnReplaySpec {
    int32_t layers = 0;
    int32_t width = 0;
    int32_t qk_heads = 0;
    int32_t value_heads = 0;
};

struct GdnReplayRecord {
    GdnReplaySpec spec;
    void * payload = nullptr;
    size_t bytes = 0;
    bool ready() const { return payload != nullptr && bytes > 0; }
};

void gdn_replay_record_prefill(GdnReplayRecord & rec, const float * q, const float * k);
void gdn_replay_replay_decode(GdnReplayRecord & rec, float * out);
