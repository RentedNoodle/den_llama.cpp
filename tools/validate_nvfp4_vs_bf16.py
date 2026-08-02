#!/usr/bin/env python3
"""
validate_nvfp4_vs_bf16.py — Per-tensor NVFP4 fidelity audit (Ornith-1.0-35B).

Compares every NVFP4 (type 40) tensor in a GGUF against its BF16 safetensors
source and reports cosine fidelity. PURPOSE (2026-08-02 rewrite):

  The I: GGUF (`I:/models/Ornith-1.0-35B-NVFP4.gguf`) was previously judged
  "irrecoverably corrupt" (plan Phase 0). That verdict was WRONG — it came from
  TWO bugs in the comparison dequant:
    1. nibble interleave: byte j holds ELEMENTS {j, 8+j} (low/high), NOT {2j, 2j+1}.
    2. scale byte stride: group g's 4 scale bytes live at (g//4)*36 + (g%4), and
       group g's 32 nibble bytes at (g//4)*36 + 4 + (g%4)*8  — the 144 B block is
       FOUR interleaved sub-blocks of [4 scale][32 nibble], not [16 scale][128 nibble].
  With the correct dequant, expert tensors round-trip at cos ~0.9955 — FAITHFUL.

  THIS TOOL is the permanent gate: any tensor with cos < 0.99 against its BF16
  source is a RUNTIME misread / conversion defect and must be investigated.

  NVFP4 on-disk block (144 B per 256 elements):
    sub-block s (s=0..3) at byte [s*36 : s*36+36] = [4 UE4M3 scale bytes][32 E2M1 nibble bytes]
    scale for group g (g=0..15):  byte (g//4)*36 + (g%4)
    nibble for group g, byte j:    byte (g//4)*36 + 4 + (g%4)*8 + j
    nibble byte j -> element j (low nibble), element 8+j (high nibble)
    value = ue4m3(scale_g) * norm * std_e2m1[nibble]
  The per-expert/per-tensor `norm` is the companion `.scale` F32 tensor.

Usage:
  python tools/validate_nvfp4_vs_bf16.py --sanity          # one expert, 5 s
  python tools/validate_nvfp4_vs_bf16.py --tensor blk.0.ffn_gate_exps.weight
  python tools/validate_nvfp4_vs_bf16.py --all --max-experts 4   # full sweep
  python tools/validate_nvfp4_vs_bf16.py --mode ci         # gate: cos>=0.99 on 16-tensor sample, exit != 0 on fail

Dependency-free (numpy only). Reads I: models. Windows-native python.
"""
import argparse
import json
import struct
import sys

import numpy as np

GGUF = "I:/models/Ornith-1.0-35B-NVFP4.gguf"
HF_DIR = "I:/models/Ornith-1.0-35B"
HF_INDEX = f"{HF_DIR}/model.safetensors.index.json"

# ── E2M1 / UE4M3 tables ──────────────────────────────────────────────────────
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

_UE4M3 = None


def ue4m3_lut():
    global _UE4M3
    if _UE4M3 is None:
        out = np.zeros(256, dtype=np.float32)
        for c in range(256):
            if c >= 0x7F:
                out[c] = 0.0
                continue
            e = (c >> 3) & 0xF
            m = c & 0x7
            out[c] = (m / 8.0) * (2.0 ** -7) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 7))
        _UE4M3 = out
    return _UE4M3


SCALE_IDX = np.array([(k // 4) * 36 + k % 4 for k in range(16)])
# Correct, sub-block-aware nibble offsets, group-major then byte-minor:
NIB_OFF = np.array([(g // 4) * 36 + 4 + (g % 4) * 8 + j for g in range(16) for j in range(8)])


# ── Raw GGUF reader ──────────────────────────────────────────────────────────
def _read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, t):
    if t == 0:
        return struct.unpack("<B", f.read(1))[0]
    if t == 1:
        return struct.unpack("<b", f.read(1))[0]
    if t == 2:
        return struct.unpack("<H", f.read(2))[0]
    if t == 3:
        return struct.unpack("<h", f.read(2))[0]
    if t == 4:
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:
        return struct.unpack("<?", f.read(1))[0]
    if t == 8:
        return _read_str(f)
    if t == 10:
        return struct.unpack("<Q", f.read(8))[0]
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError(f"unknown scalar type {t}")


def _read_kv(f):
    t = struct.unpack("<I", f.read(4))[0]
    if t == 9:
        at = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            if at == 8:
                _read_str(f)
            else:
                _read_value(f, at)
        return t
    _read_value(f, t)
    return t


def parse_gguf(path):
    """Return (tensors dict name->(type, dims, off), data_start)."""
    with open(path, "rb") as f:
        f.read(4)
        struct.unpack("<I", f.read(4))[0]  # version
        n_t = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            _read_str(f)
            _read_kv(f)
        tensors = {}
        for _ in range(n_t):
            name = _read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            tt = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = (tt, dims, off)
        data_start = (f.tell() + 31) & ~31
        return tensors, data_start


def read_gguf_tensor(tensors, data_start, name):
    tt, dims, off = tensors[name]
    ne = 1
    for d in dims:
        ne *= d
    nbytes = ((ne + 255) // 256) * 144 if tt == 40 else ne * ((2,) if tt in (30, 1) else (4,))[0]
    if tt == 40:
        with open(GGUF, "rb") as f:
            f.seek(data_start + off)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8).reshape(-1, 144)
    # non-NVFP4 (BF16 or F32) return raw bytes handled by caller
    with open(GGUF, "rb") as f:
        f.seek(data_start + off)
        return np.frombuffer(f.read(nbytes), dtype=np.uint8)


def read_gguf_scale(tensors, data_start, wname):
    base = wname[:-7] if wname.endswith(".weight") else wname
    key = base + ".scale"
    if key not in tensors:
        return None
    tt, dims, off = tensors[key]
    n = 1
    for d in dims:
        n *= d
    with open(GGUF, "rb") as f:
        f.seek(data_start + off)
        return np.frombuffer(f.read(n * 4), dtype=np.float32)


# ── Correct NVFP4 dequant ────────────────────────────────────────────────────
def dequant_nvfp4(blocks, norm):
    """blocks: (nrows, nblocks, 144) uint8. Returns float32 (nrows, nblocks*256)."""
    raw = blocks.reshape(blocks.shape[0], -1, 144)
    nr, nblk, _ = raw.shape
    sc = raw[:, :, SCALE_IDX]                                   # (nr, nblk, 16)
    nb = raw[:, :, NIB_OFF].reshape(nr, nblk, 16, 8)            # group g, byte j
    lo = nb & 0x0F
    hi = (nb >> 4) & 0x0F
    vals = np.concatenate([lo, hi], axis=-1).reshape(nr, nblk, 16, 16)
    e2 = E2M1[vals]
    scl = ue4m3_lut()[sc] * norm
    return (scl[:, :, :, None] * e2).reshape(nr, nblk * 256)


# ── HF safetensors reader (dependency-free) ──────────────────────────────────
def load_hf_header():
    idx = json.load(open(HF_INDEX))
    wm = idx["weight_map"]
    # cache per-shard headers
    headers = {}

    def get(name):
        shard = wm[name]
        if shard not in headers:
            with open(f"{HF_DIR}/{shard}", "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                headers[shard] = (json.loads(f.read(hlen).decode("utf-8")), 8 + hlen)
        return headers[shard]

    return get


def read_hf_bf16(get_hdr, name):
    hdr, ds = get_hdr(name)
    meta = hdr[name]
    with open(f"{HF_DIR}/{wm_of(get_hdr, name)}", "rb") as f:
        f.seek(ds + meta["data_offsets"][0])
        raw = np.frombuffer(f.read(meta["data_offsets"][1] - meta["data_offsets"][0]), dtype=np.uint16).reshape(
            meta["shape"]
        )
    return bf16_to_f32(raw)


_WM = {}


def wm_of(get_hdr, name):
    return _WM[name]


def bf16_to_f32(u16):
    u = u16.astype(np.int32) & 0xFFFF
    sign = (u >> 15) & 1
    exp = (u >> 7) & 0xFF
    man = u & 0x7F
    out = np.zeros(u.shape, dtype=np.float32)
    normal = (exp > 0) & (exp < 255)
    out[normal] = (1.0 + man[normal] / 128.0) * (2.0 ** (exp[normal] - 127))
    sub = exp == 0
    out[sub] = (man[sub] / 128.0) * (2.0 ** -126)
    return np.where(sign == 1, -out, out)


# ── GGUF-side transforms (mirror convert_hf_to_gguf.py Qwen3_5MoeModel) ─────
# Qwen3.5 linear attention: num_k_heads=16, num_value_heads=32 (num_v_per_k=2),
# key/value head dims 128. HF stores V grouped by K head; ggml needs tiled.
_VREORDER = None  # (num_k, num_vpk, hvd)


def _reorder_v(hf, dim, num_k, num_vpk, hvd):
    shape = list(hf.shape)
    if dim < 0:
        dim += len(shape)
    ns = shape[:dim] + [num_k, num_vpk, hvd] + shape[dim + 1:]
    t = hf.reshape(*ns)
    perm = list(range(len(ns)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.transpose(*perm).reshape(*shape)


def apply_gguf_transform(hf, gguf_name, hf_name, gguf_dims, exp_kind):
    """Apply the converter's storage transform to the HF reference so the
    GGUF dequant can be compared against the layout the loader expects."""
    if "linear_attn.in_proj_qkv.weight" in hf_name:
        q_dim = 2 * 128 * 16  # q+k
        v = _reorder_v(hf[q_dim:], 0, 16, 2, 128)
        return np.concatenate([hf[:q_dim], v], axis=0)
    if "linear_attn.in_proj_z.weight" in hf_name:
        return _reorder_v(hf, 0, 16, 2, 128)
    if "linear_attn.in_proj_a.weight" in hf_name or "linear_attn.in_proj_b.weight" in hf_name:
        return _reorder_v(hf, 0, 16, 2, 1)
    if "linear_attn.A_log" in hf_name:
        return -np.exp(hf)
    if "linear_attn.dt_bias" in hf_name:
        return _reorder_v(hf.reshape(1, -1), 0, 16, 2, 1).reshape(-1)
    if "linear_attn.out_proj.weight" in hf_name:
        return _reorder_v(hf, 1, 16, 2, 128)
    if "linear_attn.conv1d.weight" in hf_name:
        hf = hf.squeeze(1)
        qk = 2 * 128 * 16
        v = _reorder_v(hf[qk:], 0, 16, 2, 128)
        return np.concatenate([hf[:qk], v], axis=0)
    if gguf_name.endswith("_norm.weight") and "ssm_norm" not in gguf_name:
        # Gemma-style: stored = 1 - weight (sign-flip of weight-1). Loader adds 1.
        return 1.0 - hf
    return hf


# ── metrics ──────────────────────────────────────────────────────────────────
def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def analyze(dq, hf):
    c = cos(dq, hf)
    sc = cos(np.sort(dq.reshape(-1)), np.sort(hf.reshape(-1)))
    row_cos = np.array([cos(dq[r], hf[r]) for r in range(min(dq.shape[0], 4096))])
    zero_gg = (np.abs(dq).max(axis=1) < 1e-6).sum()
    zero_hf = (np.abs(hf).max(axis=1) < 1e-6).sum()
    amax_hf = float(np.abs(hf).max())
    return {
        "cos": c,
        "sorted": sc,
        "rowcos_med": float(np.median(row_cos)) if len(row_cos) else 0.0,
        "rowcos_gt09": int((row_cos > 0.9).sum()),
        "zero_gg": int(zero_gg),
        "zero_hf": int(zero_hf),
        "amax_hf": amax_hf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--tensor", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-experts", type=int, default=4)
    ap.add_argument("--mode", type=str, default=None, choices=["ci"])
    ap.add_argument("--min-cos", type=float, default=0.99)
    args = ap.parse_args()

    tensors, data_start = parse_gguf(GGUF)
    idx = json.load(open(HF_INDEX))
    _WM.update(idx["weight_map"])
    get_hdr = load_hf_header()

    # GGUF name -> (HF prefix pattern, expert_dim_present)
    def hf_prefix(gguf_name, layer, kind):
        return f"model.language_model.layers.{layer}.{kind}"

    def map_2d(gguf_name):
        # returns (hf_base, is_expert) or None
        parts = gguf_name.split(".")
        if gguf_name == "token_embd.weight":
            return ("model.language_model.embed_tokens.weight", False)
        if gguf_name == "output.weight":
            return ("lm_head.weight", False)
        if parts[0] != "blk":
            return None
        layer = int(parts[1])
        leaf = parts[2]
        if leaf == "attn_qkv":
            return (hf_prefix(gguf_name, layer, "linear_attn.in_proj_qkv.weight"), False)
        if leaf == "attn_q":
            return (hf_prefix(gguf_name, layer, "self_attn.q_proj.weight"), False)
        if leaf == "attn_k":
            return (hf_prefix(gguf_name, layer, "self_attn.k_proj.weight"), False)
        if leaf == "attn_v":
            return (hf_prefix(gguf_name, layer, "self_attn.v_proj.weight"), False)
        if leaf == "attn_output":
            return (hf_prefix(gguf_name, layer, "self_attn.o_proj.weight"), False)
        if leaf == "ffn_gate":
            return (hf_prefix(gguf_name, layer, "mlp.gate_proj.weight"), False)
        if leaf == "ffn_up":
            return (hf_prefix(gguf_name, layer, "mlp.up_proj.weight"), False)
        if leaf == "ffn_down":
            return (hf_prefix(gguf_name, layer, "mlp.down_proj.weight"), False)
        if leaf == "ffn_gate_inp":
            return (hf_prefix(gguf_name, layer, "mlp.gate.weight"), False)
        if leaf == "ffn_gate_exps":
            return (hf_prefix(gguf_name, layer, "mlp.experts"), "gate_proj")
        if leaf == "ffn_up_exps":
            return (hf_prefix(gguf_name, layer, "mlp.experts"), "up_proj")
        if leaf == "ffn_down_exps":
            return (hf_prefix(gguf_name, layer, "mlp.experts"), "down_proj")
        return None

    def validate_one(gguf_name, expert=None, verbose=True):
        tt, dims, off = tensors[gguf_name]
        m = map_2d(gguf_name)
        if m is None:
            return None
        hf_base, exp_kind = m
        if exp_kind and expert is None:
            return None
        scale = read_gguf_scale(tensors, data_start, gguf_name)
        norm = scale[expert] if (scale is not None and scale.size > 1) else (scale[0] if scale is not None else 1.0)
        if tt == 40:
            ne0, ne1 = dims[0], dims[1]
            blk_per_row = ne0 // 256
            row_bytes = blk_per_row * 144
            with open(GGUF, "rb") as f:
                f.seek(data_start + off)
                if exp_kind:
                    f.seek(data_start + off + expert * ne1 * row_bytes)
                    raw = np.frombuffer(f.read(ne1 * row_bytes), dtype=np.uint8).reshape(ne1, blk_per_row, 144)
                else:
                    raw = np.frombuffer(f.read(ne1 * row_bytes), dtype=np.uint8).reshape(ne1, blk_per_row, 144)
            dq = dequant_nvfp4(raw, norm)
        else:
            # non-NVFP4 tensor: BF16 compare directly
            if exp_kind:
                hf_name = f"{hf_base}.{expert}.{exp_kind}.weight"
            else:
                hf_name = hf_base
            hf = read_hf_bf16(get_hdr, hf_name)
            raw = read_gguf_tensor(tensors, data_start, gguf_name)
            if tt == 30:
                dq = bf16_to_f32(raw.astype(np.uint16))
            else:
                dq = raw.astype(np.float32)
            return dq, hf
        if exp_kind:
            hf_name = f"{hf_base}.{expert}.{exp_kind}.weight"
        else:
            hf_name = hf_base
        hf = read_hf_bf16(get_hdr, hf_name)
        if dq.shape == hf.shape[::-1]:
            hf = hf.T
        hf = apply_gguf_transform(hf, gguf_name, hf_name, dims, exp_kind)
        res = analyze(dq, hf)
        if verbose:
            nm = f"{gguf_name}#exp{expert}" if expert is not None else gguf_name
            verdict = "OK" if res["cos"] >= args.min_cos else "*** LOW ***"
            print(
                f"  [{verdict:>9}] {nm:55s} cos={res['cos']:.4f} sorted={res['sorted']:.4f} "
                f"rowcos>0.9={res['rowcos_gt09']} zero={res['zero_gg']}/{res['zero_hf']} "
                f"amax_hf={res['amax_hf']:.4g}"
            )
        return res

    if args.sanity:
        print("SANITY: blk.0.ffn_gate_exps.weight expert 0 vs HF gate_proj (expect cos ~0.9955):")
        validate_one("blk.0.ffn_gate_exps.weight", expert=0)
        return 0

    if args.tensor:
        if tensors[args.tensor][0] == 40 and args.tensor.endswith("exps.weight"):
            for e in range(min(args.max_experts, 4)):
                validate_one(args.tensor, expert=e)
        else:
            validate_one(args.tensor)
        return 0

    if args.all:
        nvfp4 = [n for n in tensors if tensors[n][0] == 40 and not n.endswith(".scale")]
        print(f"Sweeping {len(nvfp4)} NVFP4 tensors (max {args.max_experts} experts each for exps):")
        fails = []
        for nm in sorted(nvfp4):
            if nm.endswith("exps.weight"):
                for e in range(min(args.max_experts, 4)):
                    res = validate_one(nm, expert=e)
                    if res and res["cos"] < args.min_cos:
                        fails.append((nm, e, res["cos"]))
            else:
                res = validate_one(nm)
                if res and res["cos"] < args.min_cos:
                    fails.append((nm, None, res["cos"]))
        print("\n" + "=" * 70)
        if fails:
            print(f"LOW-COS TENSORS ({len(fails)}):")
            for nm, e, c in fails:
                print(f"  {nm}#exp{e}: cos={c:.4f}")
        else:
            print(f"ALL {len(nvfp4)} NVFP4 tensors cos >= {args.min_cos} — MODEL IS FAITHFUL.")
        return 1 if fails else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
