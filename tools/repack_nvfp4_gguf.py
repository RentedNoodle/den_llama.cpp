#!/usr/bin/env python3
"""
repack_nvfp4_gguf.py — Convert external "4/6 Adaptive Block Scaling" NVFP4 GGUFs
(144-byte blocks, signed E4M3 scales, separate .scale tensor) into the
den_llama.cpp fork's 160-byte NULLGLASS in-memory format:

  On-disk (144 B per 256 elements):
    [0:16]   16 signed-E4M3 scale bytes (one per 16-elem subgroup)
    [16:144] 128 E2M1 nibble bytes (low nibble = even elem, high = odd)
    + a separate fp32 `.scale` tensor (per-tensor tensor_scale)
  value[i] = decode_signed_e4m3(scale[i/16]) * e2m1_lut[nibble[i]] * tensor_scale

  Target NULLGLASS (160 B per 256 elements):
    [0:16]   d4[4] = 16 unsigned-E4M3 scale bytes (4 packed per uint32 LE)
    [16:144] E2M1 nibbles (sign-flipped for subgroups with negative scale)
    [148]    dispatch byte 0x10 (GEMV)
    [149]    flags 0 (non-WH4)
    [152:156] fp32 tensor_scale (per-tile norm, folded at load)
    rest     zero

The repack is EXACT (lossless) for typical scales: masking the E4M3 sign bit
preserves magnitude, and flipping E2M1 nibble sign bits absorbs negative scales
(E2M1 is sign-symmetric; the OMMA hardware decodes full unsigned E4M3 scales —
verified on RTX 5070 Ti sm_120a).

Also emits `<tensor>_n` float32 tensors (norm = tensor_scale) which the fork
reads into `model.nvfp4_norm_factors` for the software/Path-A fallback.

The metadata/KV section is copied byte-for-byte from the source.

Usage:
  python3 repack_nvfp4_gguf.py <input.gguf> <output.gguf>
"""
import struct
import sys

E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

TYPE_BPE = {0: 4, 1: 2, 2: 4, 3: 4, 5: 4, 6: 4, 7: 4, 8: 4,
            9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
            30: 32, 31: 32, 32: 4, 33: 4, 34: 4, 35: 4, 36: 4, 37: 4, 38: 4, 39: 4}

def decode_e4m3_signed(byte):
    s = (byte >> 7) & 1
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    v = ((m / 8.0) * (2.0 ** -7)) if e == 0 else ((1.0 + m / 8.0) * (2.0 ** (e - 7)))
    return -v if s else v

def decode_e4m3_unsigned(byte):
    if byte >= 0x7F:
        return 0.0
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    return (m / 8.0) * (2.0 ** -7) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 7))

# ── GGUF reader (handles 144B NVFP4) ──────────────────────────────────────────
def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, t):
    if t == 0: return struct.unpack('<B', f.read(1))[0]
    if t == 1: return struct.unpack('<b', f.read(1))[0]
    if t == 2: return struct.unpack('<H', f.read(2))[0]
    if t == 3: return struct.unpack('<h', f.read(2))[0]
    if t == 4: return struct.unpack('<I', f.read(4))[0]
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return struct.unpack('<?', f.read(1))[0]
    if t == 8: return read_str(f)
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    raise ValueError(f'unknown scalar type {t}')

def read_kv(f):
    t = struct.unpack('<I', f.read(4))[0]
    if t == 9:
        at = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            if at == 8: read_str(f)
            else: read_value(f, at)
        return t
    read_value(f, t)
    return t

def parse_gguf(path):
    """Returns (kv_end, n_tensors, tensors, data_start, tensor_info_off).
    kv_end = byte offset just past the last KV entry (unaligned).
    data_start = 32-byte aligned start of the tensor data section.
    tensor_info_off = byte offset where tensor infos begin (aligned).
    """
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'GGUF', f'bad magic {magic}'
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n_kv):
            key = read_str(f)
            read_kv(f)
        kv_end = f.tell()
        # Tensor infos follow the KV section immediately (no alignment).
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            tensors.append({'name': name, 'dims': dims, 'type': ttype, 'off': off})
        # Data section is 32-byte aligned; tensor offsets are relative to it.
        data_start = (f.tell() + 31) & ~31
        return {'version': ver, 'kv_end': kv_end, 'tensors': tensors,
                'data_start': data_start}

def read_tensor_data(path, data_start, off, nbytes):
    with open(path, 'rb') as f:
        f.seek(data_start + off)
        return f.read(nbytes)

# ── GGUF writer (copies KV section verbatim) ─────────────────────────────────
def write_gguf(path, src_prefix, version, tensors):
    """src_prefix: bytes of the original file through kv_end (copied verbatim).
    tensors: list of dict(name, dims, type, data(bytes))."""
    with open(path, 'wb') as f:
        f.write(src_prefix)
        pos = f.tell()  # kv_end — tensor infos follow immediately (no pad)
        # compute info lengths
        info_len = 0
        for t in tensors:
            info_len += 8 + len(t['name'].encode('utf-8')) + 4 + 8 * len(t['dims']) + 4 + 8
        # data section must be 32-byte aligned
        data_region = (pos + info_len + 31) & ~31
        # write tensor infos (offsets relative to data section start)
        off = 0
        for t in tensors:
            write_str(f, t['name'])
            f.write(struct.pack('<I', len(t['dims'])))
            for d in t['dims']:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', t['type']))
            f.write(struct.pack('<Q', off))
            off += len(t['data'])
        cur = f.tell()
        assert cur == pos + info_len, f'info length mismatch {cur} != {pos + info_len}'
        f.write(b'\x00' * (data_region - cur))
        for t in tensors:
            f.write(t['data'])

def write_str(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)

def repack_block(src144):
    out = bytearray(160)
    scales = src144[0:16]
    nibs = src144[16:144]
    for s in range(4):
        u32 = 0
        for k in range(4):
            mag = scales[4 * s + k] & 0x7F
            if mag >= 0x7F:      # E4M3 NaN/Inf — zero the subgroup scale
                mag = 0
            u32 |= mag << (8 * k)
        struct.pack_into('<I', out, 4 * s, u32)
    for s in range(16):
        mag = scales[s] & 0x7F
        if mag >= 0x7F:
            # NaN/Inf scale byte (converter edge case) — zero the subgroup.
            # The OMMA hardware decodes 0x7F as E4M3 NaN; the fork's software
            # treats >=0x7F as 0. Zeroing keeps both paths NaN-free.
            for j in range(8):
                out[16 + s * 8 + j] = 0
        elif scales[s] & 0x80:
            for j in range(8):
                out[16 + s * 8 + j] = nibs[s * 8 + j] ^ 0x88
        else:
            for j in range(8):
                out[16 + s * 8 + j] = nibs[s * 8 + j]
    out[148] = 0x10
    out[149] = 0
    return out

def main():
    if len(sys.argv) < 3:
        print('usage: repack_nvfp4_gguf.py <input.gguf> <output.gguf>')
        return 1
    inpath, outpath = sys.argv[1], sys.argv[2]

    meta = parse_gguf(inpath)
    tensors = meta['tensors']
    data_start = meta['data_start']
    kv_end = meta['kv_end']
    with open(inpath, 'rb') as f:
        prefix = f.read(kv_end)

    scale_map = {}
    for t in tensors:
        nm = t['name']
        if nm.endswith('.scale'):
            data = read_tensor_data(inpath, data_start, t['off'], 4)
            scale_map[nm[:-len('.scale')] + '.weight'] = struct.unpack('<f', data)[0]

    out_tensors = []
    n_repacked = 0
    n_dropped = 0
    for t in tensors:
        nm = t['name']
        tt = t['type']
        dims = t['dims']
        ne = 1
        for d in dims:
            ne *= d
        if tt == 40:
            nblocks = (ne + 255) // 256
            on_disk = nblocks * 144
            raw = read_tensor_data(inpath, data_start, t['off'], on_disk)
            tensor_scale = scale_map.get(nm, 1.0)
            out = bytearray()
            for b in range(nblocks):
                blk = repack_block(raw[b * 144:(b + 1) * 144])
                struct.pack_into('<f', blk, 152, float(tensor_scale))
                out += blk
            out_tensors.append({'name': nm, 'dims': dims, 'type': 40, 'data': bytes(out)})
            # The fork's loader folds the `.scale` companion (kept below) into
            # the per-tile norm channel; the OMMA cubin reads the block norm at
            # bytes 152:155. Both carry tensor_scale. No `_n` tensor emitted
            # (avoid loader "unknown tensor" warnings).
            n_repacked += 1
            if nm.endswith('ffn_up.weight'):
                vals_r = []
                max_diff = 0.0
                for b in range(8):
                    blk = out[b * 160:(b + 1) * 160]
                    src = raw[b * 144:(b + 1) * 144]
                    norm = struct.unpack('<f', blk[152:156])[0]
                    for sub in range(16):
                        u32 = struct.unpack('<I', blk[(sub // 4) * 4:(sub // 4) * 4 + 4])[0]
                        sc = decode_e4m3_unsigned((u32 >> ((sub % 4) * 8)) & 0xFF)
                        sc_orig = decode_e4m3_signed(src[sub])
                        for j in range(8):
                            q = blk[16 + sub * 8 + j]
                            vr = sc * E2M1_LUT[q & 0xF] * norm
                            vp = sc * E2M1_LUT[q >> 4] * norm
                            vals_r.append(vr); vals_r.append(vp)
                            qo = src[16 + sub * 8 + j]
                            vo0 = sc_orig * E2M1_LUT[qo & 0xF] * norm
                            vo1 = sc_orig * E2M1_LUT[qo >> 4] * norm
                            max_diff = max(max_diff, abs(vr - vo0), abs(vp - vo1))
                mean = sum(vals_r) / len(vals_r)
                var = sum((v - mean) ** 2 for v in vals_r) / len(vals_r)
                print(f'[self-check] {nm}: mean={mean:.4f} std={var ** 0.5:.4f} '
                      f'min={min(vals_r):.4f} max={max(vals_r):.4f} '
                      f'max_diff_vs_orig={max_diff:.6g}')
        else:
            if nm.endswith('.scale'):
                # Keep the `.scale` companion — the fork's loader folds it into
                # the per-tile norm channel (den_mxf4nvf4_gemv_launch path).
                nbytes = ne * 4  # f32
                raw = read_tensor_data(inpath, data_start, t['off'], nbytes)
                out_tensors.append({'name': nm, 'dims': dims, 'type': tt, 'data': raw})
                continue
            bpe = TYPE_BPE.get(tt, 4)
            nbytes = ((ne + 255) // 256) * bpe if tt >= 2 else ne * bpe
            raw = read_tensor_data(inpath, data_start, t['off'], nbytes)
            out_tensors.append({'name': nm, 'dims': dims, 'type': tt, 'data': raw})

    write_gguf(outpath, prefix, meta['version'], out_tensors)
    print(f'Repacked {n_repacked} NVFP4 tensors, kept {n_dropped} .scale companions')
    print(f'Wrote {outpath}')

if __name__ == '__main__':
    sys.exit(main())
