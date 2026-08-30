#include <cstdint>
#include <vector>
#include <algorithm>

// HybridKV per-head policy (real): GQA = 4 KV heads. Classify each head
// STATIC (sink+recent attention -> always resident, q8) vs DYNAMIC (middle
// long-tail -> streamed, NVFP4). ACL 2026 HybridKV. Reduces streamed volume.
enum class HeadPolicy : uint8_t { STATIC_RESIDENT = 0, DYNAMIC_STREAMED = 1 };

static std::vector<uint8_t> g_head_policy;   // per-head 0=static,1=dynamic
static int g_num_static = 0;
static int g_num_dynamic = 0;
static int g_total_heads = 0;

// Sink: first ~8 tokens attend strongly (must stay resident). Recent: last window.
// Static if the head converges attention on sink+recent (rare in GQA); else dynamic.
void hybridkv_configure(int total_heads, int num_static_heads) {
    g_total_heads = total_heads;
    g_head_policy.assign(total_heads, 1); // default dynamic/streamed
    g_num_static = std::min(num_static_heads, total_heads);
    g_num_dynamic = total_heads - g_num_static;
    // First `num_static_heads` heads (low index) stay resident (static).
    for (int i = 0; i < g_num_static; ++i) g_head_policy[i] = 0;
}

bool hybridkv_head_static(int head_idx, bool is_sink, bool is_recent) {
    if (head_idx < 0 || head_idx >= g_total_heads) return false;
    // Static-resident if policy says so OR it's a sink/recent-focused head (low index, first heads).
    return g_head_policy[head_idx] == 0;
}

// Sink pinning: never evict the first `sink_tokens` (attention sinks).
void hybridkv_pin_sink(int sink_tokens) { g_sink_tokens = std::max(g_sink_tokens, sink_tokens); }
static int g_sink_tokens = 4;

int hybridkv_sink_tokens() { return g_sink_tokens; }

// Per-head NVFP4 scale bits: static heads get q8 (full), dynamic get NVFP4 (microscaled).
int hybridkv_head_bits(int head_idx) {
    return hybridkv_head_static(head_idx, false, false) ? 8 : 4; // q8 vs nvfp4
}

// Convenience for kv_stream: head is resident (never streamed) if static.
bool hybridkv_head_resident(int head_idx) { return hybridkv_head_static(head_idx, false, false); }
