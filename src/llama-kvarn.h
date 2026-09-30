#pragma once
#include <cstdint>
// HybridKV per-head policy (ACL 2026): static heads resident (q8), dynamic streamed (NVFP4).
void hybridkv_configure(int total_heads, int num_static_heads);
bool hybridkv_head_static(int head_idx, bool is_sink, bool is_recent);
bool hybridkv_head_resident(int head_idx);
int  hybridkv_head_bits(int head_idx);
void hybridkv_pin_sink(int sink_tokens);
int  hybridkv_sink_tokens();
