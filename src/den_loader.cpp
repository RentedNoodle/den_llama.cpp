// den_loader.cpp -- .den native format parser (port of dengine den_core.c)
#include "den_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace denrt {

static uint32_t rd_u32(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd_u64(const uint8_t * p) {
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}
static int64_t rd_i64(const uint8_t * p) { return (int64_t)rd_u64(p); }
static float rd_f32(const uint8_t * p) {
    uint32_t u = rd_u32(p); float f; memcpy(&f, &u, 4); return f;
}

bool parse_header(const uint8_t * b, den_header * h) {
    h->magic   = rd_u32(b + 0);
    h->version = rd_u32(b + 4);
    h->arch    = rd_u32(b + 8);
    h->flags   = rd_u32(b + 12);
    h->n_layers = rd_u32(b + 16);
    h->n_heads  = rd_u32(b + 20);
    h->n_kv_heads = rd_u32(b + 24);
    h->hidden_size = rd_u32(b + 28);
    h->ffn_size = rd_u32(b + 32);
    h->vocab_size = rd_u32(b + 36);
    h->max_seq_len = rd_u32(b + 40);
    h->n_rot    = rd_u32(b + 44);
    h->n_experts = rd_u32(b + 48);
    h->n_experts_used = rd_u32(b + 52);
    h->rope_theta = rd_f32(b + 56);
    h->rms_norm_eps = rd_f32(b + 60);
    h->ssm_state_size = rd_u32(b + 64);
    h->ssm_conv_kernel = rd_u32(b + 68);
    h->ssm_inner_size = rd_u32(b + 72);
    h->ssm_group_count = rd_u32(b + 76);
    h->ssm_time_step_rank = rd_u32(b + 80);
    h->full_attention_interval = rd_u32(b + 84);
    h->mtp_layer_count = rd_u32(b + 88);
    h->ssm_value_size = rd_u32(b + 92);
    h->tensor_count = rd_u32(b + 104);
    h->index_offset = rd_u32(b + 108);
    h->data_offset = rd_u64(b + 112);
    h->total_data_size = rd_u64(b + 120);

    if (h->magic != DEN_MAGIC) {
        return false;
    }
    // v1 (0x00010000): compact prefix, tensor_count in flags slot, index at 16.
    if (h->version == 0x00010000) {
        h->tensor_count = h->flags;
        h->index_offset = 16;
        uint64_t data_off = 16 + (uint64_t)h->tensor_count * DEN_ENTRY_SIZE;
        data_off = ((data_off + 4095) / 4096) * 4096;
        h->data_offset = data_off;
    } else if (h->version != 0x00050000 && h->version != 0x00000001) {
        return false;
    }
    return h->tensor_count > 0 && h->tensor_count <= 4096;
}

std::vector<den_tensor> parse_entries(const uint8_t * b, size_t n) {
    std::vector<den_tensor> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t * e = b + i * DEN_ENTRY_SIZE;
        den_tensor t;
        t.slot      = rd_u32(e + 0);
        t.hw_target = rd_u32(e + 4);
        t.ndim      = rd_u32(e + 8);
        t.flags     = rd_u32(e + 12);
        t.dims[0]   = rd_i64(e + 16);
        t.dims[1]   = rd_i64(e + 24);
        t.dims[2]   = rd_i64(e + 32);
        t.dims[3]   = rd_i64(e + 40);
        t.numel     = rd_u64(e + 48);
        t.data_offset = rd_u64(e + 56);
        t.data_size   = rd_u64(e + 64);
        if (t.ndim > 4) t.ndim = 4;
        out.push_back(t);
    }
    return out;
}

bool is_den_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    uint32_t magic = 0;
    size_t rd = fread(&magic, 1, 4, f);
    fclose(f);
    return rd == 4 && magic == DEN_MAGIC;
}

bool load(const char * path, den_header * hdr, std::vector<den_tensor> * entries) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    uint8_t hdrb[DEN_HEADER_SIZE];
    if (fread(hdrb, 1, DEN_HEADER_SIZE, f) != DEN_HEADER_SIZE) { fclose(f); return false; }
    if (!parse_header(hdrb, hdr)) { fclose(f); return false; }

    const size_t index_bytes = (size_t)hdr->tensor_count * DEN_ENTRY_SIZE;
    std::vector<uint8_t> idx(index_bytes);
    if (fread(idx.data(), 1, index_bytes, f) != index_bytes) { fclose(f); return false; }
    fclose(f);

    *entries = parse_entries(idx.data(), hdr->tensor_count);
    return true;
}

// Slot -> llama.cpp qwen35 tensor name.
// Mirrors the HF->GGUF renaming in convert_hf_to_gguf.py (_ssm_tensor_map)
// for the Qwen3.5 hybrid architecture. Sub-slot ids per denrt_format.h.
std::string slot_to_name(uint32_t slot, uint32_t n_layers) {
    static const uint32_t LAYER_BASE = 3;
    if (slot == 0) return "token_embd.weight";
    if (slot == 1) return "output_norm.weight";
    if (slot == 2) return "output.weight";

    if (slot < LAYER_BASE) return std::string();
    const uint32_t layer = (slot - LAYER_BASE) / DEN_LAYER_STRIDE;
    if (layer >= n_layers) return std::string();
    const uint32_t sub = (slot - LAYER_BASE) % DEN_LAYER_STRIDE;
    char buf[128];
    switch (sub) {
        case 1:  snprintf(buf, sizeof buf, "blk.%u.attn_q.weight",           layer); return buf;
        case 2:  snprintf(buf, sizeof buf, "blk.%u.attn_k.weight",           layer); return buf;
        case 3:  snprintf(buf, sizeof buf, "blk.%u.attn_v.weight",           layer); return buf;
        case 4:  snprintf(buf, sizeof buf, "blk.%u.attn_output.weight",      layer); return buf;
        case 5:  snprintf(buf, sizeof buf, "blk.%u.attn_q_norm.weight",      layer); return buf;
        case 6:  snprintf(buf, sizeof buf, "blk.%u.attn_k_norm.weight",      layer); return buf;
        case 7:  snprintf(buf, sizeof buf, "blk.%u.ffn_gate_inp.weight",     layer); return buf; // MoE router
        case 8:  snprintf(buf, sizeof buf, "blk.%u.ffn_gate.weight",         layer); return buf;
        case 9:  snprintf(buf, sizeof buf, "blk.%u.ffn_up.weight",           layer); return buf;
        case 10: snprintf(buf, sizeof buf, "blk.%u.ffn_down.weight",         layer); return buf;
        case 11: snprintf(buf, sizeof buf, "blk.%u.ffn_gate_up_exps.weight", layer); return buf; // EXPERT_GATE_UP
        case 12: snprintf(buf, sizeof buf, "blk.%u.ssm_alpha.weight",        layer); return buf; // in_proj_a
        case 13: snprintf(buf, sizeof buf, "blk.%u.attn_gate.weight",        layer); return buf; // in_proj_z (z gate)
        case 14: snprintf(buf, sizeof buf, "blk.%u.ssm_out.weight",          layer); return buf; // out_proj
        case 15: snprintf(buf, sizeof buf, "blk.%u.ssm_a",                   layer); return buf; // A_log (transformed)
        case 16: snprintf(buf, sizeof buf, "blk.%u.ssm_dt.bias",             layer); return buf; // dt_bias
        case 17: snprintf(buf, sizeof buf, "blk.%u.ssm_conv1d.weight",       layer); return buf; // conv1d
        case 19: snprintf(buf, sizeof buf, "blk.%u.ssm_norm.weight",         layer); return buf; // norm
        case 20: snprintf(buf, sizeof buf, "blk.%u.attn_norm.weight",        layer); return buf; // input_layernorm
        case 21: snprintf(buf, sizeof buf, "blk.%u.post_attention_norm.weight", layer); return buf;
        case 24: snprintf(buf, sizeof buf, "blk.%u.attn_qkv.weight",         layer); return buf; // in_proj_qkv
        case 25: snprintf(buf, sizeof buf, "blk.%u.ssm_beta.weight",         layer); return buf; // in_proj_b
        case 26: snprintf(buf, sizeof buf, "blk.%u.ffn_down_exps.weight",    layer); return buf; // EXPERT_DOWN
        case 27: snprintf(buf, sizeof buf, "blk.%u.ffn_gate_shexp.weight",   layer); return buf;
        case 28: snprintf(buf, sizeof buf, "blk.%u.ffn_up_shexp.weight",     layer); return buf;
        case 29: snprintf(buf, sizeof buf, "blk.%u.ffn_down_shexp.weight",   layer); return buf;
        case 30: snprintf(buf, sizeof buf, "blk.%u.ffn_gate_inp_shexp.weight", layer); return buf; // SHARED_GATE_W
        default: return std::string();
    }
}

// hparam derivation.
// The .den header stores ssm.inner_size = key_dim and ssm.time_step_rank =
// num-key-heads, but llama.cpp needs ssm_d_inner = value_dim and ssm_dt_rank =
// num-value-heads. Derive from the GDN tensor shapes when present:
//   value_dim   = ssm_out/out_proj dims[0]
//   num_v_heads = ssm_dt.dims[0] (dt_bias/A_log) or in_proj_a dims[0]
//   head_dim    = attn_q dims[0] / (2 * n_heads)
void derive_hparams(const den_header & hdr, const std::vector<den_tensor> & entries,
                    uint32_t * ssm_dt_rank, uint32_t * ssm_d_inner, uint32_t * head_dim) {
    uint32_t dt_rank = hdr.ssm_time_step_rank;
    uint32_t d_inner = hdr.ssm_inner_size;
    uint32_t hdim    = hdr.n_heads ? hdr.hidden_size / hdr.n_heads : 0;

    for (const auto & t : entries) {
        if (t.slot < 3) continue;
        const uint32_t layer = (t.slot - 3) / DEN_LAYER_STRIDE;
        const uint32_t sub   = (t.slot - 3) % DEN_LAYER_STRIDE;
        if (layer >= hdr.n_layers) continue;
        switch (sub) {
            case 14: // ssm_out / out_proj: [hidden, value_dim] -> value_dim = larger dim
                if (t.ndim >= 2) d_inner = (uint32_t)std::max(t.dims[0], t.dims[1]);
                break;
            case 13: // attn_gate / in_proj_z: [value_dim, hidden] -> value_dim = larger dim (fallback)
                if (t.ndim >= 2 && d_inner == hdr.ssm_inner_size) {
                    d_inner = (uint32_t)std::max(t.dims[0], t.dims[1]);
                }
                break;
            case 15: // ssm_a / A_log: dims[0] = num value heads
                if (t.ndim >= 1 && t.dims[0] > 0) dt_rank = (uint32_t)t.dims[0];
                break;
            case 16: // ssm_dt / dt_bias: dims[0] = num value heads (fallback)
                if (t.ndim >= 1 && t.dims[0] > 0 && dt_rank == hdr.ssm_time_step_rank) {
                    dt_rank = (uint32_t)t.dims[0];
                }
                break;
            case 1: // attn_q: dims[0] = n_heads * head_dim * 2
                if (t.ndim >= 2 && t.dims[0] > 0 && hdr.n_heads) {
                    const uint32_t per = (uint32_t)t.dims[0] / (hdr.n_heads * 2);
                    if (per > 0) hdim = per;
                }
                break;
            default: break;
        }
    }
    // sanity: value_dim must be a multiple of dt_rank
    if (dt_rank && d_inner && (d_inner % dt_rank) != 0) {
        dt_rank = hdr.ssm_group_count;
        if (dt_rank && (d_inner % dt_rank) != 0) dt_rank = 0;
    }
    *ssm_dt_rank = dt_rank;
    *ssm_d_inner = d_inner;
    *head_dim    = hdim;
}

} // namespace denrt
