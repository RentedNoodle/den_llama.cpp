#!/usr/bin/env python3
"""
nvfp4_scale_audit.py - Empirical differential harness for the NVFP4 scale pipeline.

Proves on REAL model data (lostgentoo_nvfp4/out.gguf + model.safetensors):

  [A] ORACLE convention (verify_nvfp4.py, cos=0.99996):
        value = kvalues_mxfp4[nibble] * (ue4m3_bias7(scale) * 0.5) * global
  [B] IQK/CONVERT convention (iqk_quantize.cpp:4386, convert.cu:1690):
        value = kvalues_mxfp4[nibble] * ue4m3_bias7(scale) * global
        -> 2x TOO BIG vs oracle (the mission's "doubled x full" case)
  [C] DMMV convention (dmmv.cu:948-1005):
        value = standard_e2m1[nibble] * ue4m3_bias7(scale)           # NO global read
        -> pairing-correct (= oracle without global) but ZERO-applies global
  [D] DMMV+FIX: [C] with global applied -> must equal oracle

  NOTE: cosine is scale-invariant, so the DISCRIMINATOR is mean-abs-err (mae):
        oracle mae ~1e-7, iqk mae ~2e-7 (2x), dmmv mae ~O(1) (missing global).

  [E] null_skip landmine: which .scale/.input_scale values have byte 144 bit 7
      set under the CURRENT fold at bytes 144-147? (bit7=1 => OMMA kernel
      null-skips EVERY tile => all-zero weight output for that tensor.)
"""
import sys, struct, json
import numpy as np

sys.path.insert(0, '/root/den_final/llama_upstream/gguf-py')
import gguf

GGUF_PATH = '/root/den_final/lostgentoo_nvfp4/out.gguf'
HF_PATH   = '/root/den_final/lostgentoo_nvfp4/model.safetensors'

# ---- decode tables ----------------------------------------------------------
KVALS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)  # kvalues_mxfp4 (doubled)
STD   = KVALS.astype(np.float32) * 0.5                                                      # standard E2M1

def ue4m3_bias7(code):
    """Mirrors ggml_cuda_ue4m3_to_fp32 (common.cuh:426): bias 7, NO x0.5."""
    code = int(code) & 0xFF
    if code >= 0x7F:
        return 0.0
    exp = (code >> 3) & 0xF
    man = code & 0x7
    if exp == 0:
        return (float(man) / 8.0) * (2.0 ** -7)
    return (1.0 + float(man) / 8.0) * (2.0 ** (exp - 7))

UE4M3_LUT = np.array([ue4m3_bias7(c) for c in range(256)], dtype=np.float32)

def read_hf_header(path):
    with open(path, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen).decode('utf-8'))
    return header, 8 + hlen

def read_hf_bytes(path, data_start, offsets):
    with open(path, 'rb') as f:
        f.seek(data_start + offsets[0])
        return f.read(offsets[1] - offsets[0])

def e4m3_std(b):
    """Standard FP8 E4M3 decode (sign, 4 exp bias=7, 3 man) - HF side."""
    b = int(b) & 0xFF
    sign = (b >> 7) & 1
    exp  = (b >> 3) & 0xF
    man  = b & 0x7
    if exp == 0:
        v = man * (2.0 ** -9)
    elif exp == 15:
        v = 448.0
    else:
        v = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
    return -v if sign else v

def dequant_hf(w_u8, scale_bytes, global_scale):
    """HF reference: global * E4M3(bias7) * standard_e2m1."""
    out, nin_half = w_u8.shape
    lo = (w_u8 & 0x0F).astype(np.int64)
    hi = ((w_u8 >> 4) & 0x0F).astype(np.int64)
    nidx = np.empty((out, nin_half * 2), dtype=np.int64)
    nidx[:, 0::2] = lo
    nidx[:, 1::2] = hi
    e2 = STD[nidx]
    s_f = np.vectorize(e4m3_std, otypes=[np.float32])(scale_bytes)
    s_exp = np.repeat(s_f, 16, axis=1)
    return (global_scale * s_exp * e2).astype(np.float32)

def dequant_gguf(rows, mode):
    """Reconstruct from RAW 36-byte sub-blocks. mode in {oracle, iqk, dmmv}."""
    n_rows, row_bytes = rows.shape
    n_super = row_bytes // 36
    blocks = rows.reshape(n_rows, n_super, 36)
    d  = blocks[:, :, :4].astype(np.int64)
    qs = blocks[:, :, 4:].reshape(n_rows, n_super, 4, 8)
    lo = (qs & 0x0F).astype(np.int64)
    hi = ((qs >> 4) & 0x0F).astype(np.int64)
    vals = np.concatenate([lo, hi], axis=-1)
    if mode in ('dmmv', 'dmmvfix'):
        e = STD[vals].astype(np.float32)      # standard E2M1
    else:
        e = KVALS[vals].astype(np.float32)    # doubled E2M1
    scale = np.empty((n_rows, n_super, 4, 1), dtype=np.float32)
    for a in range(4):
        scale[:, :, a, 0] = UE4M3_LUT[d[:, :, a]]
    if mode == 'oracle':
        scale *= 0.5                          # the verified "ue4m3 x 0.5" half
    return (scale * e).reshape(n_rows, n_super * 64)

def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def mae(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.mean(np.abs(a - b)))

def main():
    GGUF_TO_HF = [
        ('blk.0.attn_qkv.weight', 'model.language_model.layers.0.linear_attn.in_proj_qkv'),
        ('blk.3.attn_q.weight',   'model.language_model.layers.3.self_attn.q_proj'),
        ('blk.3.ffn_down.weight', 'model.language_model.layers.3.mlp.down_proj'),
    ]
    hdr, data_start = read_hf_header(HF_PATH)
    st = {k: v for k, v in hdr.items() if isinstance(v, dict) and 'dtype' in v}
    r = gguf.GGUFReader(GGUF_PATH)
    gguf_t = {t.name: t for t in r.tensors}

    print("=" * 100)
    print("NVFP4 SCALE PIPELINE DIFFERENTIAL AUDIT  (real model: lostgentoo_nvfp4)")
    print("=" * 100)
    print("ORACLE  : kvalues_mxfp4 x (bias7_ue4m3 x 0.5) x global    (verify_nvfp4.py cos=0.99996)")
    print("IQK/CONV: kvalues_mxfp4 x  bias7_ue4m3        x global    (iqk:4386 / convert.cu:1690)")
    print("DMMV    : standard_e2m1  x  bias7_ue4m3                   (dmmv.cu:948, NO global)")
    print("DMMV+FIX: standard_e2m1  x  bias7_ue4m3        x global    (dmmv + global norm read)")
    print("DISCRIMINATOR = mae (cosine is scale-invariant)")
    print()

    for gguf_name, hf_prefix in GGUF_TO_HF:
        if gguf_name not in gguf_t:
            print(f"  [skip] {gguf_name} not in GGUF")
            continue
        wname = hf_prefix + '.weight'
        sname = hf_prefix + '.weight_scale'
        gname = hf_prefix + '.weight_scale_2'
        if gname not in st:
            gname = hf_prefix + '.weight_global_scale'
        if any(m not in st for m in (wname, sname, gname)):
            print(f"  [skip] {gguf_name}: missing HF tensors")
            continue
        gt = gguf_t[gguf_name]
        rows = gt.data
        w_bytes = np.frombuffer(read_hf_bytes(HF_PATH, data_start, st[wname]['data_offsets']), dtype=np.uint8)
        s_bytes = np.frombuffer(read_hf_bytes(HF_PATH, data_start, st[sname]['data_offsets']), dtype=np.uint8)
        g_raw = read_hf_bytes(HF_PATH, data_start, st[gname]['data_offsets'])
        global_scale = struct.unpack('<f', g_raw)[0]
        out_hf, in_half = st[wname]['shape']
        n_use = min(2048, rows.shape[0], out_hf)
        rows = rows[:n_use]
        w_u8 = w_bytes.reshape(out_hf, in_half)[:n_use]
        s_u8 = s_bytes.reshape(out_hf, st[sname]['shape'][1])[:n_use]
        v_hf = dequant_hf(w_u8, s_u8, global_scale)

        print(f"--- {gguf_name}  (global_scale = {global_scale:.6g}) ---")
        for mode, apply_g in (('oracle', True), ('iqk', True), ('dmmv', False), ('dmmvfix', True)):
            v_gg = dequant_gguf(rows, mode)
            if apply_g:
                v_gg = v_gg * global_scale
            c = cosine(v_gg, v_hf)
            m = mae(v_gg, v_hf)
            if mode == 'oracle':
                verdict = 'REFERENCE'
            elif mode == 'iqk':
                verdict = '2x TOO BIG (missing ue4m3 x0.5)'
            elif mode == 'dmmv':
                verdict = 'ZERO-GLOBAL (missing bytes 144-147 read)'
            else:
                verdict = 'FIXED (matches oracle)'
            print(f"  {mode:12s} cos={c:.6f}  mae={m:.3e}   {verdict}")
        print()

    # ---- [E] null_skip landmine across ALL companion scale tensors ----
    print("=" * 100)
    print("[E] null_skip landmine: .scale float -> byte144 bit7 under CURRENT 144-147 fold")
    print("    bit7=1 => OMMA kernel null-skips EVERY tile => all-zero weight output")
    print("=" * 100)
    land = 0
    nscale = 0
    nisc = 0
    for t in r.tensors:
        nm = str(t.name)
        if nm.endswith('.scale'):
            nscale += 1
            arr = t.data.reshape(-1)
            v = float(arr[0])
            b0 = struct.pack('<f', v)[0]
            if b0 & 0x80:
                land += 1
                print(f"  !! LANDMINE  {nm}: v={v:.6g} byte144=0x{b0:02x} -> null_skip=TRUE")
        elif nm.endswith('.input_scale'):
            nisc += 1
            arr = t.data.reshape(-1)
            v = float(arr[0])
            b0 = struct.pack('<f', v)[0]
            if b0 & 0x80:
                land += 1
                print(f"  !! LANDMINE  {nm} (.input_scale): v={v:.6g} byte144=0x{b0:02x}")
    print(f"  .scale tensors={nscale}  .input_scale tensors={nisc}  LANDMINES (byte144 bit7 set)={land}")
    print("  => migration to bytes 152:155 removes this class of corruption entirely.")

if __name__ == '__main__':
    main()
