#include <cstdint>
#include <vector>
// HybridKV head-aware policy: static heads (sink+recent) stay kvarn4 resident,
// dynamic heads (middle) get streamed. AC 2026 HybridKV.
enum class HeadPolicy : uint8_t { STATIC_RESIDENT=0, DYNAMIC_STREAMED=1 };

struct HybridKVPolicy {
    std::vector<uint8_t> per_head; // 0 static, 1 dynamic
    int num_static = 0;
    int num_dynamic = 0;
    bool classify(int head_idx, bool is_sink, bool is_recent, int total_heads) {
        // Static: sink (first ~8 tokens) or recent (last 16?) attend uniformly — keep resident.
        // Dynamic: attend to long-tail middle — stream.
        // For a 27B qwen35 with GQA (4 kv heads, 24 q heads), classify by head.
        bool static_head = is_sink || is_recent;
        per_head.push_back(static_head ? 0 : 1);
        if (static_head) ++num_static; else ++num_dynamic;
        return static_head;
    }
};

static HybridKVPolicy g_policy;
void hybridkv_configure(int total_heads) { g_policy.per_head.assign(total_heads, 1); g_policy.num_static=0; g_policy.num_dynamic=total_heads; }
bool hybridkv_head_static(int head_idx, bool is_sink, bool is_recent) { return g_policy.classify(head_idx, is_sink, is_recent, g_policy.per_head.size()); }
