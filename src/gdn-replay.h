#pragma once
#include <cstdint>
#include <cstddef>
struct GdnReplaySpec { int32_t layers=0; int32_t width=0; int32_t qk_heads=0; int32_t value_heads=0; };
struct GdnReplayRecord { GdnReplaySpec spec; void * payload=nullptr; size_t bytes=0; bool ready() const { return payload && bytes; } };
void gdn_replay_record_prefill(GdnReplayRecord & rec, const float * q, const float * k);
void gdn_replay_replay_decode(GdnReplayRecord & rec, float * out);
