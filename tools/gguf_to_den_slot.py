#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gguf_to_den_slot.py -- GGUF (NVFP4) -> slot-.den repack for den_llama.cpp.

The thesis Blocker-5 format: NVFP4 weights live in NULLGLASS 160B tiles, keyed
by (3 + layer*32 + subslot), behind a 4096B header + 128B slot entries.

NO re-quantization: the GGUF's NVFP4 blocks and their `.scale` companions are
reused verbatim. For NVFP4 tensors this replicates the load-time 144->160B
expansion in llama-model-loader.cpp exactly:
  * de-interleave 4x [4 scales][32 nibbles] per on-disk block
  * zero the 16B NULLGLASS header (bytes 144:160)
  * fold the companion `.scale` (per-tensor or per-expert) into bytes 152:155
  * set blk[148] = 0x10 (GEMV dispatch)

Slot map (denrt_format.h / den_loader.cpp slot_to_name):
  global: 0 token_embd, 1 output_norm, 2 output
  per-layer (base = 3 + layer*32):
    1 attn_q, 2 attn_k, 3 attn_v, 4 attn_output, 5 attn_q_norm, 6 attn_k_norm,
    7 ffn_gate_inp (MoE router), 8/9/10 dense ffn gate/up/down,
    11 EXPERT_GATE_UP (fused ffn_gate_up_exps = ffn_gate_exps + ffn_up_exps),
    12 ssm_alpha, 13 attn_gate, 14 ssm_out, 15 ssm_a, 16 ssm_dt.bias,
    17 ssm_conv1d (stored BF16, loader expands to F32), 19 ssm_norm,
    20 attn_norm, 21 post_attention_norm, 24 attn_qkv, 25 ssm_beta,
    26 EXPERT_DOWN (ffn_down_exps),
    27 ffn_gate_shexp, 28 ffn_up_shexp, 29 ffn_down_shexp, 30 ffn_gate_inp_shexp
"""
import argparse
import os
import re
import struct
import sys
import time

import numpy as np

try:
    from gguf import GGUFReader
except ImportError:
    print("need gguf python package (pip install gguf)", file=sys.stderr)
    sys.exit(2)

# ---- .den format constants (denrt_format.h / den_loader.cpp) ----
DEN_MAGIC         = 0x4E454400
DEN_VERSION       = 0x00050000
DEN_HEADER_SIZE   = 4096
DEN_ENTRY_SIZE    = 128
DEN_LAYER_STRIDE  = 32
DEN_ARCH_QWEN35   = 1
DEN_ARCH_QWEN36_MOE = 2

DEN_HW_NVFP4 = 1
DEN_HW_BF16  = 2
DEN_HW_F32   = 3

# ---- NVFP4 block geometry (ggml-common.h / llama-model-loader.cpp) ----
QK_NVFP4      = 256        # elements per NULLGLASS tile
NVFP4_ON_DISK = 144        # GGUF on-disk bytes per 256-elem block
NVFP4_MEM     = 160        # in-memory NULLGLASS bytes per block
NVFP4_DISPATCH_GEMV = 0x10 # tile[148]

# ---- ggml type ids in this fork's enum (ggml.h) ----
GGML_F32   = 0
GGML_BF16  = 30
GGML_NVFP4 = 40

# ---- GGUF tail name -> sub-slot (matches den_loader.cpp slot_to_name) ----
SUB = {
    "attn_q.weight": 1, "attn_k.weight": 2, "attn_v.weight": 3, "attn_output.weight": 4,
    "attn_q_norm.weight": 5, "attn_k_norm.weight": 6,
    "ffn_gate_inp.weight": 7,
    "ffn_gate.weight": 8, "ffn_up.weight": 9, "ffn_down.weight": 10,
    "ssm_alpha.weight": 12, "attn_gate.weight": 13, "ssm_out.weight": 14,
    "ssm_a": 15, "ssm_dt.bias": 16, "ssm_conv1d.weight": 17,
    "ssm_norm.weight": 19, "attn_norm.weight": 20, "post_attention_norm.weight": 21,
    "attn_qkv.weight": 24, "ssm_beta.weight": 25,
    "ffn_gate_shexp.weight": 27, "ffn_up_shexp.weight": 28,
    "ffn_down_shexp.weight": 29, "ffn_gate_inp_shexp.weight": 30,
}
GATE_HALF   = "ffn_gate_exps.weight"
UP_HALF     = "ffn_up_exps.weight"
EXPERT_DOWN = "ffn_down_exps.weight"


def slot_of(name, n_layers):
    """Map a GGUF tensor name to a .den slot. Returns None for companions/skip."""
    if name == "token_embd.weight":
        return 0
    if name == "output_norm.weight":
        return 1
    if name == "output.weight":
        return 2
    if name.endswith(".scale") or name.endswith(".input_scale") or name.endswith("_n"):
        return None
    m = re.match(r"blk\.(\d+)\.(.*)$", name)
    if not m:
        return None
    layer = int(m.group(1))
    if layer >= n_layers:
        return None
    tail = m.group(2)
    base = 3 + layer * DEN_LAYER_STRIDE
    if tail in (GATE_HALF, UP_HALF):
        return base + 11          # EXPERT_GATE_UP (fused gate+up)
    if tail == EXPERT_DOWN:
        return base + 26          # EXPERT_DOWN
    if tail in SUB:
        return base + SUB[tail]
    return None


def scale_name_of(weight_name):
    """The companion scale tensor for a weight (llama-model-loader convention)."""
    if weight_name.endswith(".weight"):
        return weight_name[:-len(".weight")] + ".scale"
    return None


def expand_nvfp4_blocks(raw, nblocks):
    """Replicate llama-model-loader.cpp 144->160 expansion (vectorized)."""
    b = np.asarray(raw, dtype=np.uint8).reshape(nblocks, 4, 36)
    out = np.zeros((nblocks, NVFP4_MEM), dtype=np.uint8)
    out[:, 0:16] = b[:, :, 0:4].reshape(nblocks, 16)      # scales -> [0:16]
    out[:, 16:144] = b[:, :, 4:36].reshape(nblocks, 128)  # nibbles -> [16:144]
    # bytes 144:160 stay zero (NULLGLASS header)
    return out


def fold_scale(out, gval):
    """Fold per-tensor/per-expert .scale into bytes 152:155, set dispatch byte.
    Replicates the loader: fold only when scale != 1.0; blk[148]=0x10 always.
    gval: 1-D float32 numpy (len 1 = global, len n_experts = per-expert) or None."""
    nblocks = out.shape[0]
    if gval is None or len(gval) == 0:
        scales = np.ones(nblocks, dtype=np.float32)
    elif len(gval) == 1:
        scales = np.full(nblocks, gval[0], dtype=np.float32)
    else:
        n_exp = len(gval)
        assert nblocks % n_exp == 0, "blocks %d not divisible by experts %d" % (nblocks, n_exp)
        scales = np.repeat(gval, nblocks // n_exp)
    out[:, 148] = NVFP4_DISPATCH_GEMV
    idx = np.nonzero(scales != 1.0)[0]
    if len(idx):
        out[idx, 152:156] = scales[idx].astype("<f4").view(np.uint8).reshape(len(idx), 4)


def f32_to_bf16(data):
    """Convert raw F32 bytes to BF16 (truncate mantissa)."""
    u32 = np.frombuffer(data, dtype=np.uint32)
    return (u32 >> 16).astype(np.uint16).tobytes()


def _scale_of(g, name):
    """Return float32 1-D numpy of the companion .scale tensor, or None."""
    if name is None:
        return None
    t = g.get(name)
    if t is None:
        return None
    return np.asarray(t.data, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="input NVFP4 GGUF")
    ap.add_argument("--output", "-o", required=True, help="output slot-.den")
    ap.add_argument("--skip-data", action="store_true",
                    help="write header+entries only (dry run, for sizing)")
    ap.add_argument("--n-layers", type=int, default=0, help="override layer count")
    args = ap.parse_args()

    t0 = time.time()
    r = GGUFReader(args.input)
    g = {t.name: t for t in r.tensors}

    # ---- hparams from KV ----
    def kv_u32(suffix, default=0):
        for k, f in r.fields.items():
            if k.endswith(suffix):
                v = f.contents()
                if hasattr(v, "__len__") and not isinstance(v, str):
                    return int(v[0])
                return int(v)
        return default

    def kv_f32(suffix, default=0.0):
        for k, f in r.fields.items():
            if k.endswith(suffix):
                v = f.contents()
                if hasattr(v, "__len__") and not isinstance(v, str):
                    return float(v[0])
                return float(v)
        return default

    n_layers  = args.n_layers or kv_u32("block_count")
    n_heads   = kv_u32("attention.head_count")
    n_kv      = kv_u32("attention.head_count_kv")
    hidden    = kv_u32("embedding_length")
    ffn_expert = kv_u32("expert_feed_forward_length")
    ffn_shared = kv_u32("expert_shared_feed_forward_length")
    n_used    = kv_u32("expert_used_count")
    n_experts = kv_u32("expert_count")
    vocab     = kv_u32("vocab_size", 0) or g["token_embd.weight"].shape.tolist()[1]
    max_seq   = kv_u32("context_length", 262144)
    n_rot     = kv_u32("rope.dimension_count", 64)
    rope_theta = kv_f32("rope.freq_base", 1e7)
    rms_eps   = kv_f32("attention.layer_norm_rms_epsilon", 1e-6)
    ssm_state = kv_u32("ssm.state_size", 128)
    ssm_conv  = kv_u32("ssm.conv_kernel", 4)
    ssm_inner = kv_u32("ssm.inner_size", 0)
    ssm_grp   = kv_u32("ssm.group_count", 0)
    ssm_tstep = kv_u32("ssm.time_step_rank", 0)
    full_attn = kv_u32("full_attention_interval", 4)

    ffn = (ffn_expert * n_used + ffn_shared) if n_experts > 0 else (hidden * 4)
    ssm_vd = ssm_inner // ssm_grp if (ssm_inner and ssm_grp) else 0

    print("[hparams] layers=%d heads=%d kv=%d hidden=%d ffn=%d vocab=%d experts=%d/%d "
          "ssm(state=%d conv=%d inner=%d grp=%d tstep=%d) full_attn=%d"
          % (n_layers, n_heads, n_kv, hidden, ffn, vocab, n_experts, n_used,
             ssm_state, ssm_conv, ssm_inner, ssm_grp, ssm_tstep, full_attn))

    # ---- Pass A: map tensors -> slots ----
    slot_tensors = {}   # slot -> list of (name, role); role in {"w","gate","up"}
    for t in r.tensors:
        slot = slot_of(t.name, n_layers)
        if slot is None:
            continue
        tail = re.match(r"blk\.\d+\.(.*)$", t.name)
        role = "w"
        if tail:
            if tail.group(1) == GATE_HALF:
                role = "gate"
            elif tail.group(1) == UP_HALF:
                role = "up"
        slot_tensors.setdefault(slot, []).append((t.name, role))

    for slot, lst in list(slot_tensors.items()):
        roles = [rr for _, rr in lst]
        sub = (slot - 3) % DEN_LAYER_STRIDE if slot >= 3 else slot
        if sub == 11 and roles != ["gate", "up"]:
            print("  [ERROR] slot %d has unexpected fused sources: %s" % (slot, roles))
            sys.exit(1)

    slots = sorted(slot_tensors.keys())
    n_entries = len(slots)
    index_end = DEN_HEADER_SIZE + n_entries * DEN_ENTRY_SIZE
    data_offset = ((index_end + 4095) // 4096) * 4096

    slot_size = {}
    slot_hw = {}
    slot_shape = {}
    slot_numel = {}

    def entry_meta(slot, lst):
        sub = (slot - 3) % DEN_LAYER_STRIDE if slot >= 3 else slot
        if slot == 0:
            t = g[lst[0][0]]; ne = t.shape.tolist()
            n = int(np.prod(ne))
            return DEN_HW_BF16, n, list(reversed(ne)), n * 2
        if sub == 17:  # ssm_conv1d -> BF16 (loader expands to F32)
            t = g[lst[0][0]]; ne = t.shape.tolist()
            n = int(np.prod(ne))
            return DEN_HW_BF16, n, list(reversed(ne)), n * 2
        if sub == 11:  # fused gate+up
            gate = g[lst[0][0]]; up = g[lst[1][0]]
            ne_g = gate.shape.tolist(); ne_u = up.shape.tolist()
            fused_ne = [ne_g[0], ne_g[1] + ne_u[1], ne_g[2]]
            n = int(np.prod(fused_ne))
            nblocks = n // QK_NVFP4
            return DEN_HW_NVFP4, n, list(reversed(fused_ne)), nblocks * NVFP4_MEM
        t = g[lst[0][0]]
        tt = t.tensor_type
        ne = t.shape.tolist()
        n = int(np.prod(ne))
        if tt.name == "NVFP4":
            nblocks = n // QK_NVFP4
            return DEN_HW_NVFP4, n, list(reversed(ne)), nblocks * NVFP4_MEM
        if tt.name == "BF16":
            return DEN_HW_BF16, n, list(reversed(ne)), n * 2
        return DEN_HW_F32, n, list(reversed(ne)), n * 4

    pos = data_offset
    for slot in slots:
        hw, numel, shape, size = entry_meta(slot, slot_tensors[slot])
        slot_hw[slot] = hw; slot_numel[slot] = numel; slot_shape[slot] = shape
        slot_size[slot] = size
        pos += size
    total_data_size = pos - data_offset

    print("[plan] %d slots, data_offset=%d, total_data=%d (%.2f GB)"
          % (n_entries, data_offset, total_data_size, total_data_size / 1e9))

    # ---- Write header + entries ----
    arch = DEN_ARCH_QWEN36_MOE if n_experts > 0 else DEN_ARCH_QWEN35
    hdr = struct.pack("<IIII", DEN_MAGIC, DEN_VERSION, arch, 0)
    hdr += struct.pack("<IIIIIIII", n_layers, n_heads, n_kv, hidden, ffn, vocab, max_seq, n_rot)
    hdr += struct.pack("<II", n_experts, n_used)
    hdr += struct.pack("<ff", rope_theta, rms_eps)
    hdr += struct.pack("<IIIIII", ssm_state, ssm_conv, ssm_inner, ssm_grp, ssm_tstep, full_attn)
    hdr += struct.pack("<II", 0, ssm_vd)
    hdr += struct.pack("<II", 0, 0)
    hdr += struct.pack("<IIQ", n_entries, DEN_HEADER_SIZE, data_offset)
    hdr += struct.pack("<Q", total_data_size)
    hdr += struct.pack("<III", 0, 0, 0)
    hdr += struct.pack("<I", 0)
    hdr += struct.pack("<QQQ", 0, 0, 0)
    assert len(hdr) <= DEN_HEADER_SIZE
    hdr = hdr.ljust(DEN_HEADER_SIZE, b"\x00")

    off = 0
    entry_bytes = b""
    for slot in slots:
        hw = slot_hw[slot]; numel = slot_numel[slot]; shape = slot_shape[slot]
        ndim = min(len(shape), 4)
        dims = (list(shape[:4]) + [0] * 4)[:4]
        e = struct.pack("<IIII", slot, hw, ndim, 0)
        e += struct.pack("<qqqq", dims[0], dims[1], dims[2], dims[3])
        e += struct.pack("<Q", numel)
        e += struct.pack("<QQ", off, slot_size[slot])
        e += struct.pack("<QQ", 0, 0)
        e += struct.pack("<IIII", 0, 0, 0, 0)
        e += struct.pack("<QI", 0, 0)
        e += struct.pack("<II", 0, 0)
        e += struct.pack("<I", 0)
        assert len(e) == DEN_ENTRY_SIZE
        entry_bytes += e
        off += slot_size[slot]

    with open(args.output, "wb") as f:
        f.write(hdr)
        f.write(entry_bytes)
        pad = data_offset - f.tell()
        if pad > 0:
            f.write(b"\x00" * pad)

    if args.skip_data:
        print("[dry-run] header+entries written to %s (%d bytes)"
              % (args.output, os.path.getsize(args.output)))
        return

    # ---- Pass C: write data ----
    n_nvfp4 = n_bf16 = n_f32 = 0
    cur = data_offset
    with open(args.output, "r+b") as f:
        for i, slot in enumerate(slots):
            lst = slot_tensors[slot]
            sub = (slot - 3) % DEN_LAYER_STRIDE if slot >= 3 else slot
            name0 = lst[0][0]
            src_t = g[name0]
            tt = src_t.tensor_type

            if sub == 17:  # conv1d: F32 -> BF16 (data is float32 native bytes)
                raw = src_t.data.tobytes()
                out = f32_to_bf16(raw)
                f.seek(cur); f.write(out)
                n_bf16 += 1
            elif sub == 11:  # fused gate+up
                gate_t = g[lst[0][0]]; up_t = g[lst[1][0]]
                g_ne = gate_t.shape.tolist(); u_ne = up_t.shape.tolist()
                n_exp = g_ne[2]
                nblocks_g = int(np.prod(g_ne)) // QK_NVFP4
                nblocks_u = int(np.prod(u_ne)) // QK_NVFP4
                assert nblocks_g == nblocks_u
                gb = expand_nvfp4_blocks(np.asarray(gate_t.data, dtype=np.uint8), nblocks_g)
                ub = expand_nvfp4_blocks(np.asarray(up_t.data, dtype=np.uint8), nblocks_u)
                gs = _scale_of(g, scale_name_of(name0))
                us = _scale_of(g, scale_name_of(lst[1][0]))
                fold_scale(gb, gs)
                fold_scale(ub, us)
                bpe = int(g_ne[0]) // QK_NVFP4
                gb = gb.reshape(n_exp, int(g_ne[1]), bpe, NVFP4_MEM)
                ub = ub.reshape(n_exp, int(u_ne[1]), bpe, NVFP4_MEM)
                fused = np.concatenate([gb, ub], axis=1).reshape(-1, NVFP4_MEM)
                f.seek(cur); f.write(fused.tobytes())
                n_nvfp4 += 1
            elif tt.name == "NVFP4":
                ne = src_t.shape.tolist()
                n = int(np.prod(ne)); nblocks = n // QK_NVFP4
                out = expand_nvfp4_blocks(np.asarray(src_t.data, dtype=np.uint8), nblocks)
                gval = _scale_of(g, scale_name_of(name0))
                fold_scale(out, gval)
                f.seek(cur); f.write(out.tobytes())
                n_nvfp4 += 1
            elif tt.name == "BF16":
                f.seek(cur); f.write(src_t.data.tobytes())
                n_bf16 += 1
            else:  # F32 (data is float32 native bytes -- .tobytes() keeps 4B/elem)
                f.seek(cur); f.write(src_t.data.tobytes())
                n_f32 += 1

            cur += slot_size[slot]
            if (i + 1) % 25 == 0:
                print("  ... %d/%d slots (%.1f%%), elapsed %.0fs"
                      % (i + 1, len(slots), 100.0 * (i + 1) / len(slots), time.time() - t0),
                      flush=True)

    print("[done] %d NVFP4 + %d BF16 + %d F32 slots -> %s (%.2f GB) in %.0fs"
          % (n_nvfp4, n_bf16, n_f32, args.output,
             os.path.getsize(args.output) / 1e9, time.time() - t0))


if __name__ == "__main__":
    main()
