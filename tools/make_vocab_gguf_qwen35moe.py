#!/usr/bin/env python3
"""make_vocab_gguf_qwen35moe.py -- companion vocab GGUF for a MoE .den.

The .den loader (load_den_model) hardcodes LLM_ARCH_QWEN35 + qwen35.* hparams
KV, then grafts ALL KV from the companion "<model>.vocab.gguf" (gguf_set_kv
overwrites). To make a MoE slot-.den load as LLM_ARCH_QWEN35MOE we carry:
  * general.architecture = "qwen35moe"          (overrides the loader default)
  * qwen35moe.* hparams KV                       (the MoE hparams loader reads)
  * tokenizer.ggml.* + tokenizer.chat_template   (llama.cpp requires a tokenizer)

Values are taken from the source GGUF (Ornith-1.0-35B-NVFP4) and the HF
tokenizer at the model dir.
"""
import argparse
import json
import struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
T_U32, T_I32, T_F32, T_STR, T_ARR, T_BOOL = 4, 5, 6, 8, 9, 7
T_INT32 = 5
TOKEN_NORMAL, TOKEN_CONTROL, TOKEN_UNUSED = 1, 3, 5


class GGUFWriter:
    def __init__(self, path):
        self.path = path
        self.buf = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 0))
        self.n_kv = 0

    def _wstr(self, s):
        b = s.encode("utf-8")
        self.buf += struct.pack("<Q", len(b)) + b

    def kv(self, key, vtype, payload):
        self._wstr(key)
        self.buf += struct.pack("<I", vtype) + payload
        self.n_kv += 1

    def u32(self, k, v):
        self.kv(k, T_U32, struct.pack("<I", int(v)))

    def f32(self, k, v):
        self.kv(k, T_F32, struct.pack("<f", float(v)))

    def str(self, k, v):
        b = v.encode("utf-8")
        self.kv(k, T_STR, struct.pack("<Q", len(b)) + b)

    def arr_i32(self, k, vals):
        p = struct.pack("<IQ", T_INT32, len(vals)) + b"".join(struct.pack("<i", int(x)) for x in vals)
        self.kv(k, T_ARR, p)

    def arr_str(self, k, vals):
        p = struct.pack("<IQ", T_STR, len(vals))
        for s in vals:
            b = s.encode("utf-8")
            p += struct.pack("<Q", len(b)) + b
        self.kv(k, T_ARR, p)

    def save(self):
        self.buf[16:24] = struct.pack("<Q", self.n_kv)
        with open(self.path, "wb") as f:
            f.write(bytes(self.buf))


def read_gguf_hparams(gguf_path):
    """Read the qwen35moe.* hparams KV from the source GGUF (exact values)."""
    from gguf import GGUFReader
    r = GGUFReader(gguf_path)

    def u32(suffix, default=0):
        k = "qwen35moe." + suffix
        if k in r.fields:
            v = r.fields[k].contents()
            return int(v[0]) if hasattr(v, "__len__") and not isinstance(v, str) else int(v)
        return default

    def f32(suffix, default=0.0):
        k = "qwen35moe." + suffix
        if k in r.fields:
            v = r.fields[k].contents()
            return float(v[0]) if hasattr(v, "__len__") and not isinstance(v, str) else float(v)
        return default

    def arr_i32(suffix):
        k = "qwen35moe." + suffix
        if k in r.fields:
            return [int(x) for x in r.fields[k].contents()]
        return []

    hp = {}
    hp["block_count"] = u32("block_count")
    hp["context_length"] = u32("context_length")
    hp["embedding_length"] = u32("embedding_length")
    hp["vocab_size"] = u32("vocab_size", 0)
    hp["head_count"] = u32("attention.head_count")
    hp["head_count_kv"] = u32("attention.head_count_kv")
    hp["rms_eps"] = f32("attention.layer_norm_rms_epsilon", 1e-6)
    hp["key_length"] = u32("attention.key_length")
    hp["value_length"] = u32("attention.value_length")
    hp["expert_count"] = u32("expert_count")
    hp["expert_used_count"] = u32("expert_used_count")
    hp["expert_ffn_length"] = u32("expert_feed_forward_length")
    hp["expert_shared_ffn_length"] = u32("expert_shared_feed_forward_length")
    hp["ffn_length"] = u32("feed_forward_length")
    hp["ssm_state"] = u32("ssm.state_size")
    hp["ssm_conv"] = u32("ssm.conv_kernel")
    hp["ssm_inner"] = u32("ssm.inner_size")
    hp["ssm_tstep"] = u32("ssm.time_step_rank")
    hp["ssm_group"] = u32("ssm.group_count")
    hp["full_attn"] = u32("full_attention_interval", 4)
    hp["rope_sections"] = arr_i32("rope.dimension_sections")
    hp["rope_freq"] = f32("rope.freq_base", 1e7)
    hp["rope_dim"] = u32("rope.dimension_count")
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gguf", default="",
                    help="source NVFP4 GGUF to copy qwen35moe hparams from (exact values)")
    ap.add_argument("--chat-template", default="")
    args = ap.parse_args()
    tdir = args.tokenizer_dir
    tok = json.load(open("%s/tokenizer.json" % tdir))
    cfg = json.load(open("%s/config.json" % tdir))
    tcfg = json.load(open("%s/tokenizer_config.json" % tdir))
    tc = cfg.get("text_config", cfg)

    vocab_size = tc.get("vocab_size") or cfg.get("vocab_size") or len(tok["model"]["vocab"])
    model_vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    if merges and isinstance(merges[0], (list, tuple)):
        merges = ["%s %s" % (m[0], m[1]) for m in merges]
    added = {a["content"]: a["id"] for a in tok.get("added_tokens", [])}
    tokens = [""] * vocab_size
    toktypes = [TOKEN_NORMAL] * vocab_size
    for tkn, idx in model_vocab.items():
        if idx < vocab_size:
            tokens[idx] = tkn
    for content, idx in added.items():
        if idx < vocab_size:
            tokens[idx] = content
            if content in ("<|endoftext|>", "<|im_end|>", "<|im_start|>"):
                toktypes[idx] = TOKEN_CONTROL
    for i in range(vocab_size):
        if not tokens[i]:
            tokens[i] = "[PAD%d]" % i
            toktypes[i] = TOKEN_UNUSED

    eos = tc.get("eos_token_id")
    bos = tc.get("bos_token_id")
    pad = tc.get("pad_token_id")
    if eos is None and tcfg.get("eos_token"):
        eos = added.get(tcfg["eos_token"])
    if bos is None and tcfg.get("bos_token"):
        bos = added.get(tcfg["bos_token"])
    if pad is None and tcfg.get("pad_token"):
        pad = added.get(tcfg["pad_token"])

    w = GGUFWriter(args.out)
    w.str("general.architecture", "qwen35moe")
    w.str("general.type", "model")

    # ---- qwen35moe hparams (copied verbatim from the source NVFP4 GGUF) ----
    hp = read_gguf_hparams(args.gguf) if args.gguf else {}
    w.u32("qwen35moe.block_count", hp.get("block_count", tc.get("num_hidden_layers", 0)))
    w.u32("qwen35moe.context_length", hp.get("context_length", tc.get("max_position_embeddings", 262144)))
    w.u32("qwen35moe.embedding_length", hp.get("embedding_length", tc.get("hidden_size", 0)))
    w.u32("qwen35moe.vocab_size", hp.get("vocab_size") or vocab_size)
    w.u32("qwen35moe.attention.head_count", hp.get("head_count", tc.get("num_attention_heads", 0)))
    w.u32("qwen35moe.attention.head_count_kv", hp.get("head_count_kv", tc.get("num_key_value_heads", 0)))
    w.f32("qwen35moe.attention.layer_norm_rms_epsilon", hp.get("rms_eps", 1e-6))
    w.u32("qwen35moe.attention.key_length", hp.get("key_length", 256))
    w.u32("qwen35moe.attention.value_length", hp.get("value_length", 256))
    w.u32("qwen35moe.expert_count", hp.get("expert_count", tc.get("num_experts", 0)))
    w.u32("qwen35moe.expert_used_count", hp.get("expert_used_count", tc.get("num_experts_per_tok", 0)))
    w.u32("qwen35moe.expert_feed_forward_length", hp.get("expert_ffn_length", 0))
    w.u32("qwen35moe.expert_shared_feed_forward_length", hp.get("expert_shared_ffn_length", 0))
    w.u32("qwen35moe.feed_forward_length", hp.get("ffn_length", 0))
    w.u32("qwen35moe.ssm.state_size", hp.get("ssm_state", 128))
    w.u32("qwen35moe.ssm.conv_kernel", hp.get("ssm_conv", 4))
    w.u32("qwen35moe.ssm.inner_size", hp.get("ssm_inner", 0))
    w.u32("qwen35moe.ssm.time_step_rank", hp.get("ssm_tstep", 0))
    w.u32("qwen35moe.ssm.group_count", hp.get("ssm_group", 0))
    w.u32("qwen35moe.full_attention_interval", hp.get("full_attn", 4))
    w.arr_i32("qwen35moe.rope.dimension_sections", hp.get("rope_sections", [11, 11, 10, 0]))
    w.f32("qwen35moe.rope.freq_base", hp.get("rope_freq", 1e7))
    w.u32("qwen35moe.rope.dimension_count", hp.get("rope_dim", 64))

    # ---- tokenizer ----
    w.str("tokenizer.ggml.model", "gpt2")
    w.str("tokenizer.ggml.pre", "qwen35")
    w.arr_str("tokenizer.ggml.tokens", tokens)
    w.arr_i32("tokenizer.ggml.token_type", toktypes)
    w.arr_str("tokenizer.ggml.merges", merges)
    if eos is not None:
        w.u32("tokenizer.ggml.eos_token_id", int(eos))
    if bos is not None:
        w.u32("tokenizer.ggml.bos_token_id", int(bos))
    if pad is not None:
        w.u32("tokenizer.ggml.padding_token_id", int(pad))
    template = args.chat_template or tcfg.get("chat_template", "")
    if template:
        w.str("tokenizer.chat_template", template)
    w.save()
    print("Wrote %s: arch=qwen35moe vocab=%d merges=%d eos=%s bos=%s pad=%s"
          % (args.out, vocab_size, len(merges), eos, bos, pad))


if __name__ == "__main__":
    main()
