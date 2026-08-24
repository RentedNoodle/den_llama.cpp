#include "models.h"
#include "llama-memory-recurrent.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

LLAMA_API bool llama_qwen35_escha_companions_complete(
        bool code,
        bool config,
        bool rin,
        bool rout,
        bool s_in,
        bool s_out,
        bool bias) {
    GGML_UNUSED(bias);
    return code && config && rin && rout && s_in && s_out;
}

LLAMA_API bool llama_qwen35_escha_shared_complete(
        bool lut,
        bool dep_k2,
        bool dep_k3,
        bool needs_dep_k3) {
    return lut && dep_k2 && (!needs_dep_k3 || dep_k3);
}

LLAMA_API bool llama_qwen35_escha_inputs_complete(
        const ggml_tensor * dep,
        const ggml_tensor * x) {
    return dep != nullptr && x != nullptr &&
        (dep->type == GGML_TYPE_I16 || dep->type == GGML_TYPE_I32) &&
        dep->ne[0] == 16 && dep->ne[1] == 256 && x->type == GGML_TYPE_F32;
}

static void qwen35_validate_scaled_i8_rows(
        const ggml_tensor * values,
        const ggml_tensor * row_scales,
        const ggml_tensor * rows) {
    if (values == nullptr || row_scales == nullptr || rows == nullptr ||
            values->type != GGML_TYPE_I8 || row_scales->type != GGML_TYPE_F16 ||
            rows->type != GGML_TYPE_I32 || ggml_n_dims(values) != 2 ||
            ggml_n_dims(row_scales) != 1 || values->ne[1] != row_scales->ne[0]) {
        throw std::runtime_error("qwen35 scaled-I8 embedding: invalid weight/scale/row shape or type");
    }
}

static void qwen35_validate_scaled_i8_mat(
        const ggml_tensor * weights,
        const ggml_tensor * activations,
        const ggml_tensor * row_scales) {
    if (weights == nullptr || activations == nullptr || row_scales == nullptr ||
            weights->type != GGML_TYPE_I8 || activations->type != GGML_TYPE_F32 ||
            row_scales->type != GGML_TYPE_F16 || ggml_n_dims(weights) != 2 ||
            ggml_n_dims(row_scales) != 1 || weights->ne[0] != activations->ne[0] ||
            weights->ne[1] != row_scales->ne[0]) {
        throw std::runtime_error(format(
            "qwen35 scaled-I8 lm-head: invalid tensors: w=%s[%lld,%lld] x=%s[%lld,%lld] s=%s[%lld]",
            weights ? ggml_type_name(weights->type) : "null",
            (long long) (weights ? weights->ne[0] : -1), (long long) (weights ? weights->ne[1] : -1),
            activations ? ggml_type_name(activations->type) : "null",
            (long long) (activations ? activations->ne[0] : -1), (long long) (activations ? activations->ne[1] : -1),
            row_scales ? ggml_type_name(row_scales->type) : "null",
            (long long) (row_scales ? row_scales->ne[0] : -1)));
    }
}

LLAMA_API ggml_tensor * llama_qwen35_get_rows_endpoint(
        ggml_context * ctx,
        ggml_tensor  * values,
        ggml_tensor  * row_scales,
        ggml_tensor  * rows) {
    if (row_scales == nullptr) {
        if (values == nullptr || values->type == GGML_TYPE_I8) {
            throw std::runtime_error("qwen35 embedding: I8 weight requires F16 row scales");
        }
        return ggml_get_rows(ctx, values, rows);
    }
    qwen35_validate_scaled_i8_rows(values, row_scales, rows);
    return ggml_get_rows_scaled_i8(ctx, values, rows, row_scales);
}

LLAMA_API ggml_tensor * llama_qwen35_mul_mat_endpoint(
        ggml_context * ctx,
        ggml_tensor  * weights,
        ggml_tensor  * activations,
        ggml_tensor  * row_scales) {
    if (row_scales == nullptr) {
        if (weights == nullptr || weights->type == GGML_TYPE_I8) {
            throw std::runtime_error("qwen35 lm-head: I8 weight requires F16 row scales");
        }
        return ggml_mul_mat(ctx, weights, activations);
    }
    qwen35_validate_scaled_i8_mat(weights, activations, row_scales);
    return ggml_mul_mat_scaled_i8(ctx, weights, activations, row_scales);
}

static ggml_tensor * qwen35_build_inp_embd(
        llm_graph_context & graph,
        ggml_tensor        * tok_embd,
        ggml_tensor        * tok_embd_s) {
    const int64_t n_embd_inp = graph.hparams.n_embd_inp();
    const int64_t n_embd     = graph.hparams.n_embd;

    GGML_ASSERT(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);
    inp->tokens = ggml_new_tensor_1d(graph.ctx0, GGML_TYPE_I32, graph.ubatch.n_tokens);
    graph.cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    graph.res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(graph.ctx0, GGML_TYPE_F32, n_embd_inp, graph.ubatch.n_tokens);
    graph.cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    std::array<ggml_tensor *, 2> inps;
    ggml_tensor * & token_input = inps[0];
    token_input = llama_qwen35_get_rows_endpoint(graph.ctx0, tok_embd, tok_embd_s, inp->tokens);

    for (const auto & lora : *graph.loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);
        ggml_tensor * inpL_delta = ggml_scale(graph.ctx0, ggml_mul_mat(
                graph.ctx0, lw->b, ggml_get_rows(graph.ctx0, lw->a, inp->tokens)), scale);
        token_input = ggml_add(graph.ctx0, token_input, inpL_delta);
    }

    if (n_embd_inp != n_embd) {
        token_input = ggml_pad(graph.ctx0, token_input, n_embd_inp - n_embd, 0, 0, 0);
    }
    inps[1] = inp->embd;

    GGML_ASSERT(ggml_are_same_shape(inps[0], inps[1]));
    GGML_ASSERT(ggml_are_same_stride(inps[0], inps[1]));
    ggml_tensor * cur = ggml_build_forward_select(graph.gf, inps.data(), inps.size(), graph.ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(graph.ctx0, cur, n_embd, graph.n_tokens, cur->nb[1], 0);
    }

    graph.res->t_inp_embd = cur;
    graph.cb(cur, "embd", -1);
    graph.res->add_input(std::move(inp));
    ggml_build_forward_expand(graph.gf, cur);
    return cur;
}


void llama_model_qwen35::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // NextN/MTP (Qwen3.5/3.6): extra decoder block appended beyond the main stack
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < n_layer_impl");

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_8B : LLM_TYPE_2B; break;
        case 32: type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_9B; break;
        case 64: type = LLM_TYPE_27B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const bool mtp_only = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    // Dual-hybrid (Cold-Fusion): every layer carries BOTH the linear-attn and
    // the full-attention tensor sets. Detect once - only then create the
    // mirrored optional sets below (plain qwen35 checkpoints must be inert).
    const bool dual_hybrid = !mtp_only
        && (ml.get_tensor_meta("blk.0.ssm_conv1d.weight") != nullptr)
        && (ml.get_tensor_meta("blk.3.attn_q.weight") != nullptr
            || ml.get_tensor_meta("blk.3.attn_q.weight.escha_code") != nullptr);
    hparams.is_dual_hybrid = dual_hybrid;
    const int trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    int64_t n_vocab_out = n_vocab;
    const ggml_tensor * d2t_meta = ml.get_tensor_meta("d2t");
    if (mtp_only && d2t_meta) {
        n_vocab_out = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_out }, 0);
        LLAMA_LOG_INFO("%s: QWEN35 MTP using d2t draft-vocab trim (n_vocab_out = %lld)\n",
                __func__, (long long) n_vocab_out);
    }

    auto load_scaled_endpoint = [&](llm_tensor tensor, int il, int64_t n_in, int64_t n_out,
                                    int flags, ggml_tensor * & row_scales) -> ggml_tensor * {
        const llm_tensor scaled_tensor = tensor == LLM_TENSOR_TOKEN_EMBD
            ? LLM_TENSOR_QWEN35_TOKEN_EMBD_SCALED
            : LLM_TENSOR_QWEN35_OUTPUT_SCALED;
        const std::string weight_name = tn(scaled_tensor, "weight", il).str();
        const std::string scale_name  = tn(scaled_tensor, "weight_scale", il).str();
        const ggml_tensor * weight_meta = ml.get_tensor_meta(weight_name.c_str());
        const ggml_tensor * scale_meta  = ml.get_tensor_meta(scale_name.c_str());

        if (weight_meta == nullptr && scale_meta == nullptr) {
            row_scales = nullptr;
            return nullptr;
        }
        if (scale_meta == nullptr && weight_meta != nullptr && weight_meta->type != GGML_TYPE_I8) {
            row_scales = nullptr;
            return nullptr;
        }
        if (weight_meta == nullptr || scale_meta == nullptr) {
            throw std::runtime_error("qwen35 scaled-I8 endpoint requires both I8 weight and weight_scale");
        }
        if (weight_meta->type != GGML_TYPE_I8 || scale_meta->type != GGML_TYPE_F16 ||
                ggml_n_dims(weight_meta) != 2 || ggml_n_dims(scale_meta) != 1 ||
                weight_meta->ne[0] != n_in || weight_meta->ne[1] != n_out ||
                scale_meta->ne[0] != n_out) {
            throw std::runtime_error("qwen35 scaled-I8 endpoint has invalid weight/scale type or shape");
        }

        row_scales = create_tensor(tn(scaled_tensor, "weight_scale", il), { n_out }, flags);
        return create_tensor(tn(scaled_tensor, "weight", il), { n_in, n_out }, flags);
    };

    tok_embd = load_scaled_endpoint(LLM_TENSOR_TOKEN_EMBD, -1, n_embd, n_vocab, 0, tok_embd_s);
    if (tok_embd == nullptr) {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    }

    // Escha W2 shared codec (official dense decode)
    if (ml.get_tensor_meta("escha_lut") != nullptr) {
        escha_lut = create_tensor(tn(LLM_TENSOR_ESCHA_LUT), { 65536 }, 0);
    }
    if (ml.get_tensor_meta("escha_dep_k2") != nullptr) {
        escha_dep = create_tensor(tn(LLM_TENSOR_ESCHA_DEP_K2), { 16, 256 }, TENSOR_NOT_REQUIRED);
    }
    if (ml.get_tensor_meta("escha_dep_k3") != nullptr) {
        escha_dep3 = create_tensor(tn(LLM_TENSOR_ESCHA_DEP_K3), { 16, 256 }, TENSOR_NOT_REQUIRED);
    }


    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = load_scaled_endpoint(LLM_TENSOR_OUTPUT, -1, n_embd, n_vocab_out, TENSOR_NOT_REQUIRED, output_s);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab_out }, TENSOR_NOT_REQUIRED);
    }

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        GGML_ASSERT(!d2t && "d2t draft-vocab trim requires output.weight");
        if (tok_embd_s != nullptr) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
            output_s = tok_embd_s;
        } else {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
        }
    }


    // Escha W2 (official dense decode): load the 6 sidecar sets per projection
    // when the {name}.weight.escha_code tensor is present. Shapes from the code
    // tensor (ne[1]*16 = O, ne[2]*16 = I).
    auto load_escha_proj = [&](llama_escha_proj & e, llm_tensor t, llm_tensor escha_t, int il,
                               int64_t ic, int64_t oc, int flags) -> bool {
        GGML_UNUSED(ic);
        GGML_UNUSED(oc);
        const std::string cn = tn(t, "weight.escha_code", il).str();
        const auto * code_meta   = ml.get_tensor_meta(cn.c_str());
        const auto * config_meta = ml.get_tensor_meta(tn(t, "weight.escha_config", il).str().c_str());
        const auto * rin_meta    = ml.get_tensor_meta(tn(t, "weight.escha_rin", il).str().c_str());
        const auto * rout_meta   = ml.get_tensor_meta(tn(t, "weight.escha_rout", il).str().c_str());
        const auto * s_in_meta   = ml.get_tensor_meta(tn(t, "weight.escha_s_in", il).str().c_str());
        const auto * s_out_meta  = ml.get_tensor_meta(tn(t, "weight.escha_s_out", il).str().c_str());
        const auto * bias_meta   = ml.get_tensor_meta(tn(t, "weight.escha_bias", il).str().c_str());
        const bool any_companion = code_meta || config_meta || rin_meta || rout_meta || s_in_meta || s_out_meta || bias_meta;
        if (!any_companion) {
            return false;
        }
        if (!llama_qwen35_escha_companions_complete(code_meta != nullptr, config_meta != nullptr,
                rin_meta != nullptr, rout_meta != nullptr, s_in_meta != nullptr, s_out_meta != nullptr,
                bias_meta != nullptr)) {
            throw std::runtime_error(format("incomplete ESCHA companion set for %s", cn.c_str()));
        }

        const bool needs_dep_k3 = code_meta->ne[0] == 48;
        if (!llama_qwen35_escha_shared_complete(
                ml.get_tensor_meta("escha_lut") != nullptr,
                ml.get_tensor_meta("escha_dep_k2") != nullptr,
                ml.get_tensor_meta("escha_dep_k3") != nullptr,
                needs_dep_k3)) {
            throw std::runtime_error(format("incomplete ESCHA shared set for %s", cn.c_str()));
        }

        // Dimensions come from the packed code tensor: ne[1]*16 = O, ne[2]*16 = I.
        const int64_t real_oc = code_meta->ne[1]*16;
        const int64_t real_ic = code_meta->ne[2]*16;
        e.code  = create_tensor(tn(escha_t, "weight.escha_code",  il), { code_meta->ne[0], code_meta->ne[1], code_meta->ne[2] }, flags);
        e.rin   = create_tensor(tn(escha_t, "weight.escha_rin",   il), { real_ic }, flags);
        e.rout  = create_tensor(tn(escha_t, "weight.escha_rout",  il), { real_oc }, flags);
        e.s_in  = create_tensor(tn(escha_t, "weight.escha_s_in",  il), { real_ic }, flags);
        e.s_out = create_tensor(tn(escha_t, "weight.escha_s_out", il), { real_oc }, flags);
        e.bias  = create_tensor(tn(escha_t, "weight.escha_bias",  il), { real_oc }, flags | TENSOR_NOT_REQUIRED);
        // Explicit capability check, not a broad qwen35 mutation. This is the
        // Qwen3.8-27B dense W2 geometry: 64 layers, H=5120, 48 value heads.
        e.qwen38_w2_fast_path = hparams.n_layer_all == 64 && n_embd == 5120 &&
            hparams.ssm_dt_rank == 48 && real_ic % 128 == 0 && real_oc % 128 == 0;
        create_tensor(tn(escha_t, "weight.escha_config", il), { 6 }, flags);
        return true;
    };

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        // Escha W2 sidecars (dense): attn_qkv + attn_gate live on the recurrent
        // layers; the FFN sets on all layers; ssm_out on the recurrent layers.
        load_escha_proj(layer.escha_ffn_gate, LLM_TENSOR_FFN_GATE, LLM_TENSOR_QWEN35_ESCHA_FFN_GATE, il, n_embd, n_ff,      flags);
        load_escha_proj(layer.escha_ffn_up,   LLM_TENSOR_FFN_UP,   LLM_TENSOR_QWEN35_ESCHA_FFN_UP,   il, n_embd, n_ff,      flags);
        load_escha_proj(layer.escha_ffn_down, LLM_TENSOR_FFN_DOWN, LLM_TENSOR_QWEN35_ESCHA_FFN_DOWN, il, n_ff,   n_embd,    flags);
        if (hparams.is_recr(il)) {
            // recurrent layers carry the linear-attn escha projections
            load_escha_proj(layer.escha_wqkv,    LLM_TENSOR_ATTN_QKV,  LLM_TENSOR_QWEN35_ESCHA_ATTN_QKV,  il, n_embd, key_dim * 2 + value_dim, flags);
            load_escha_proj(layer.escha_gate,    LLM_TENSOR_ATTN_GATE, LLM_TENSOR_QWEN35_ESCHA_ATTN_GATE, il, n_embd, value_dim, flags);
            load_escha_proj(layer.escha_ssm_out, LLM_TENSOR_SSM_OUT,   LLM_TENSOR_QWEN35_ESCHA_SSM_OUT,   il, value_dim, n_embd, flags);
        }

        if (!hparams.is_recr(il)) {
            // Attention layers
            const bool e_q = load_escha_proj(layer.escha_attn_q, LLM_TENSOR_ATTN_Q, LLM_TENSOR_QWEN35_ESCHA_ATTN_Q, il, n_embd, n_embd_head_k * n_head, flags);
            const bool e_k = load_escha_proj(layer.escha_attn_k, LLM_TENSOR_ATTN_K, LLM_TENSOR_QWEN35_ESCHA_ATTN_K, il, n_embd, n_embd_k_gqa,        flags);
            const bool e_v = load_escha_proj(layer.escha_attn_v, LLM_TENSOR_ATTN_V, LLM_TENSOR_QWEN35_ESCHA_ATTN_V, il, n_embd, n_embd_v_gqa,        flags);
            load_escha_proj(layer.escha_attn_out, LLM_TENSOR_ATTN_OUT, LLM_TENSOR_QWEN35_ESCHA_ATTN_OUT, il, n_embd_head_k * n_head, n_embd, flags);
            const int aflags = (e_q || e_k || e_v) ? TENSOR_NOT_REQUIRED : flags;
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, aflags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, layer.escha_attn_out.code ? TENSOR_NOT_REQUIRED : flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            const int raw_flags = flags & ~(TENSOR_SKIP | TENSOR_NOT_REQUIRED);
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, layer.escha_wqkv.code ? TENSOR_NOT_REQUIRED : raw_flags);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, layer.escha_gate.code ? TENSOR_NOT_REQUIRED : raw_flags);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, layer.escha_ssm_out.code ? TENSOR_NOT_REQUIRED : flags);
            if (dual_hybrid) {
                create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, TENSOR_NOT_REQUIRED);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, TENSOR_NOT_REQUIRED);
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, TENSOR_NOT_REQUIRED);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, TENSOR_NOT_REQUIRED);
            }
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, layer.escha_ffn_gate.code ? TENSOR_NOT_REQUIRED : flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, layer.escha_ffn_down.code ? TENSOR_NOT_REQUIRED : flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, layer.escha_ffn_up.code ? TENSOR_NOT_REQUIRED : flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        // MTP block looks like a full-attention Qwen3.5 decoder block.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, mtp_flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {  n_ff, n_embd}, mtp_flags);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen35::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = qwen35_build_inp_embd(*this, model.tok_embd, model.tok_embd_s);
    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = model.output->type == GGML_TYPE_I8
        ? llama_qwen35_mul_mat_endpoint(ctx0, model.output, cur, model.output_s)
        : build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}


static ggml_tensor * escha_mm_dense(ggml_context * ctx0,
        const llama_escha_proj & e, ggml_tensor * x,
        ggml_tensor * dep, ggml_tensor * dep3) {
    if (!llama_qwen35_escha_inputs_complete(dep, x) || e.code == nullptr || e.rin == nullptr ||
            e.rout == nullptr || e.s_in == nullptr || e.s_out == nullptr) {
        throw std::runtime_error("Qwen3.5 dense ESCHA graph requires code, rin, rout, s_in, s_out, dep, and x");
    }
    // The K=3 projections (the MLP) need the K3 dependency table.
    ggml_tensor * dep_sel = (e.code && e.code->ne[0] == 48) ? dep3 : dep;
    if (dep_sel == nullptr) {
        throw std::runtime_error("Qwen3.5 ESCHA graph requires the matching dependency table");
    }
    return e.qwen38_w2_fast_path
        ? ggml_escha_linear_qwen38_w2(ctx0, e.code, e.rin, e.rout, e.s_in, e.s_out, dep_sel, x)
        : ggml_escha_linear(ctx0, e.code, e.rin, e.rout, e.s_in, e.s_out, dep_sel, x);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = nullptr;
    const llama_escha_proj & e_wqkv = model.layers[il].escha_wqkv;
    if (e_wqkv.code != nullptr) {
        qkv_mixed = escha_mm_dense(ctx0, e_wqkv, input, model.escha_dep, model.escha_dep3);
        if (e_wqkv.bias) {
            qkv_mixed = ggml_add(ctx0, qkv_mixed, ggml_cast(ctx0, e_wqkv.bias, GGML_TYPE_F32));
        }
    } else {
        qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    }
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = nullptr;
    const llama_escha_proj & e_z = model.layers[il].escha_gate;
    if (e_z.code != nullptr) {
        z = escha_mm_dense(ctx0, e_z, input, model.escha_dep, model.escha_dep3);
        if (e_z.bias) {
            z = ggml_add(ctx0, z, ggml_cast(ctx0, e_z.bias, GGML_TYPE_F32));
        }
    } else {
        z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    }
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    const llama_escha_proj & e_q = model.layers[il].escha_attn_q;
    ggml_tensor * Qcur_full = e_q.code
        ? escha_mm_dense(ctx0, e_q, cur, model.escha_dep, model.escha_dep3)
        : build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
    if (e_q.code && e_q.bias) {
        Qcur_full = ggml_add(ctx0, Qcur_full, ggml_cast(ctx0, e_q.bias, GGML_TYPE_F32));
    }
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    const llama_escha_proj & e_k = model.layers[il].escha_attn_k;
    ggml_tensor * Kcur = e_k.code
        ? escha_mm_dense(ctx0, e_k, cur, model.escha_dep, model.escha_dep3)
        : build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    if (e_k.code && e_k.bias) {
        Kcur = ggml_add(ctx0, Kcur, ggml_cast(ctx0, e_k.bias, GGML_TYPE_F32));
    }
    cb(Kcur, "Kcur", il);

    const llama_escha_proj & e_v = model.layers[il].escha_attn_v;
    ggml_tensor * Vcur = e_v.code
        ? escha_mm_dense(ctx0, e_v, cur, model.escha_dep, model.escha_dep3)
        : build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    if (e_v.code && e_v.bias) {
        Vcur = ggml_add(ctx0, Vcur, ggml_cast(ctx0, e_v.bias, GGML_TYPE_F32));
    }
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    const llama_escha_proj & e_wo = model.layers[il].escha_attn_out;
    cur = e_wo.code
        ? escha_mm_dense(ctx0, e_wo, cur, model.escha_dep, model.escha_dep3)
        : build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    if (e_wo.code && e_wo.bias) {
        cur = ggml_add(ctx0, cur, ggml_cast(ctx0, e_wo.bias, GGML_TYPE_F32));
    }
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    const llama_escha_proj & e_out = model.layers[il].escha_ssm_out;
    if (e_out.code != nullptr) {
        cur = escha_mm_dense(ctx0, e_out, final_output, model.escha_dep, model.escha_dep3);
        if (e_out.bias) {
            cur = ggml_add(ctx0, cur, ggml_cast(ctx0, e_out.bias, GGML_TYPE_F32));
        }
    } else {
        cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    }
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    const llama_escha_proj & e_g = model.layers[il].escha_ffn_gate;
    const llama_escha_proj & e_u = model.layers[il].escha_ffn_up;
    const llama_escha_proj & e_d = model.layers[il].escha_ffn_down;
    if (e_g.code != nullptr && e_u.code != nullptr && e_d.code != nullptr) {
        ggml_tensor * g = escha_mm_dense(ctx0, e_g, cur, model.escha_dep, model.escha_dep3);
        ggml_tensor * u = escha_mm_dense(ctx0, e_u, cur, model.escha_dep, model.escha_dep3);
        if (e_g.bias) { g = ggml_add(ctx0, g, ggml_cast(ctx0, e_g.bias, GGML_TYPE_F32)); }
        if (e_u.bias) { u = ggml_add(ctx0, u, ggml_cast(ctx0, e_u.bias, GGML_TYPE_F32)); }
        cur = ggml_mul(ctx0, ggml_silu(ctx0, g), u);
        cur = escha_mm_dense(ctx0, e_d, cur, model.escha_dep, model.escha_dep3);
        if (e_d.bias) { cur = ggml_add(ctx0, cur, ggml_cast(ctx0, e_d.bias, GGML_TYPE_F32)); }
    } else {
        cur = build_ffn(cur,
            model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
            model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
            model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    }
    cb(cur, "ffn_out", il);

    return cur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
llama_model_qwen35::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // hparams.n_layer includes both main model layers and MTP layers. The MTP
    // layer is stored immediately after the main layers in model.layers[].
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm   && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm   && "MTP block missing nextn.hnorm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
        ggml_tensor * tok_embd_s = tok_embd_w == model.tok_embd ? model.tok_embd_s : nullptr;

        tok_embd = llama_qwen35_get_rows_endpoint(ctx0, tok_embd_w, tok_embd_s, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    cur = build_ffn(cur,
            layer.ffn_up,   nullptr, layer.ffn_up_s,
            layer.ffn_gate, nullptr, layer.ffn_gate_s,
            layer.ffn_down, nullptr, layer.ffn_down_s,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
    cur = head_w->type == GGML_TYPE_I8
        ? llama_qwen35_mul_mat_endpoint(ctx0, head_w, cur, head_s)
        : build_lora_mm(head_w, cur, head_s);
    cb(cur, "result_output", -1);

    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab_full  = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0,
                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab_full, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab_full, n_outputs);
        cb(cur, "result_output_d2t", -1);
    }

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
