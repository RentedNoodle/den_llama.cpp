#!/usr/bin/env python3
"""make_vocab_gguf.py -- build a tokenizer-only GGUF from an HF Qwen3.5 tokenizer.

The .den format stores weights only (no tokenizer). llama.cpp requires the
tokenizer KV (tokenizer.ggml.*) to be present in the model gguf context, so the
.den loader grafts these KV pairs from a small companion "vocab GGUF".
"""
import argparse, json, struct

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)
TOKEN_NORMAL,TOKEN_UNKNOWN,TOKEN_CONTROL,TOKEN_USER_DEFINED,TOKEN_UNUSED,TOKEN_BYTE = range(1,7)

class GGUFWriter:
    def __init__(self, path):
        self.path = path
        self.buf = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, 0, 0))
        self.n_kv = 0
    def _wstr(self, s):
        b = s.encode("utf-8"); self.buf += struct.pack("<Q", len(b)); self.buf += b
    def kv(self, key, vtype, payload):
        self._wstr(key); self.buf += struct.pack("<I", vtype); self.buf += payload; self.n_kv += 1
    def u32(self,k,v): self.kv(k, T_U32, struct.pack("<I", v))
    def i32(self,k,v): self.kv(k, T_I32, struct.pack("<i", v))
    def str(self,k,v):
        b = v.encode("utf-8"); self.kv(k, T_STR, struct.pack("<Q", len(b)) + b)
    def arr_i32(self,k,vals):
        p = struct.pack("<IQ", T_I32, len(vals)) + b"".join(struct.pack("<i",x) for x in vals)
        self.kv(k, T_ARR, p)
    def arr_str(self,k,vals):
        p = struct.pack("<IQ", T_STR, len(vals))
        for s in vals:
            b = s.encode("utf-8"); p += struct.pack("<Q", len(b)) + b
        self.kv(k, T_ARR, p)
    def save(self):
        self.buf[16:24] = struct.pack("<Q", self.n_kv)
        with open(self.path, "wb") as f: f.write(bytes(self.buf))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat-template", default="")
    args = ap.parse_args()
    tdir = args.tokenizer_dir
    tok = json.load(open(f"{tdir}/tokenizer.json"))
    cfg = json.load(open(f"{tdir}/config.json"))
    tcfg = json.load(open(f"{tdir}/tokenizer_config.json"))
    tc = cfg.get("text_config", cfg)
    vocab_size = tc.get("vocab_size") or cfg.get("vocab_size") or len(tok["model"]["vocab"])
    model_vocab = tok["model"]["vocab"]; merges = tok["model"]["merges"]
    if merges and isinstance(merges[0], (list, tuple)):
        merges = ["%s %s" % (m[0], m[1]) for m in merges]
    added = {a["content"]: a["id"] for a in tok.get("added_tokens", [])}
    tokens = [""]*vocab_size; toktypes = [TOKEN_NORMAL]*vocab_size
    for tkn, idx in model_vocab.items():
        if idx < vocab_size: tokens[idx] = tkn
    for content, idx in added.items():
        if idx < vocab_size:
            tokens[idx] = content
            if content in ("<|endoftext|>","<|im_end|>","<|im_start|>"):
                toktypes[idx] = TOKEN_CONTROL
    for i in range(vocab_size):
        if not tokens[i]:
            tokens[i] = f"[PAD{i}]"; toktypes[i] = TOKEN_UNUSED
    eos = tc.get("eos_token_id"); bos = tc.get("bos_token_id"); pad = tc.get("pad_token_id")
    if eos is None and tcfg.get("eos_token"): eos = added.get(tcfg["eos_token"])
    if bos is None and tcfg.get("bos_token"): bos = added.get(tcfg["bos_token"])
    if pad is None and tcfg.get("pad_token"): pad = added.get(tcfg["pad_token"])
    w = GGUFWriter(args.out)
    w.str("general.architecture", "qwen35")
    w.str("general.type", "model")
    w.str("tokenizer.ggml.model", "gpt2")
    w.str("tokenizer.ggml.pre", "qwen35")
    w.arr_str("tokenizer.ggml.tokens", tokens)
    w.arr_i32("tokenizer.ggml.token_type", toktypes)
    w.arr_str("tokenizer.ggml.merges", merges)
    if eos is not None: w.u32("tokenizer.ggml.eos_token_id", int(eos))
    if bos is not None: w.u32("tokenizer.ggml.bos_token_id", int(bos))
    if pad is not None: w.u32("tokenizer.ggml.padding_token_id", int(pad))
    template = args.chat_template or tcfg.get("chat_template", "")
    if template: w.str("tokenizer.chat_template", template)
    w.save()
    print(f"Wrote {args.out}: vocab={vocab_size} merges={len(merges)} eos={eos} bos={bos} pad={pad}")
if __name__ == "__main__":
    main()
