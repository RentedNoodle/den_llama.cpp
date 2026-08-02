#!/usr/bin/env python3
"""gguf_probe.py -- minimal GGUF parser to inventory tensors + KV metadata."""
import struct, sys, json

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

# metadata value types (GGUF spec)
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64 = range(13)

def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8")

def read_val(f, t):
    if t == T_U8:  return struct.unpack("<B", f.read(1))[0]
    if t == T_I8:  return struct.unpack("<b", f.read(1))[0]
    if t == T_U16: return struct.unpack("<H", f.read(2))[0]
    if t == T_I16: return struct.unpack("<h", f.read(2))[0]
    if t == T_U32: return struct.unpack("<I", f.read(4))[0]
    if t == T_I32: return struct.unpack("<i", f.read(4))[0]
    if t == T_F32: return struct.unpack("<f", f.read(4))[0]
    if t == T_BOOL: return struct.unpack("<?", f.read(1))[0]
    if t == T_STR: return read_str(f)
    if t == T_U64: return struct.unpack("<Q", f.read(8))[0]
    if t == T_I64: return struct.unpack("<q", f.read(8))[0]
    if t == T_F64: return struct.unpack("<d", f.read(8))[0]
    if t == T_ARR:
        vt = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [read_val(f, vt) for _ in range(n)]
    raise ValueError(f"unknown type {t}")

def parse(path):
    f = open(path, "rb")
    magic, ver, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    assert magic == GGUF_MAGIC, f"bad magic {magic:#x}"
    print(f"GGUF v{ver}: n_tensors={n_tensors} n_kv={n_kv}")
    kv = {}
    for _ in range(n_kv):
        k = read_str(f)
        t = struct.unpack("<I", f.read(4))[0]
        kv[k] = (t, read_val(f, t))
    # pad to 32-byte alignment
    pad = (32 - f.tell() % 32) % 32
    f.read(pad)
    tensors = []
    for _ in range(n_tensors):
        name = read_str(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        dims = list(struct.unpack("<" + "Q"*n_dims, f.read(8*n_dims)))
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors.append((name, n_dims, dims, ttype, off))
    pad = (32 - f.tell() % 32) % 32
    data_start = f.tell() + pad
    return kv, tensors, data_start, f

if __name__ == "__main__":
    path = sys.argv[1]
    kv, tensors, data_start, f = parse(path)
    print(f"tensor data starts at {data_start}")
    # print scalar/array hparams
    for k,(t,v) in kv.items():
        if t != T_ARR and not isinstance(v,(list,dict)):
            print(f"  KV {k} = {v}")
    print("\n=== tensor type histogram ===")
    from collections import Counter
    c = Counter(t[3] for t in tensors)
    for tt,n in sorted(c.items()): print(f"  type {tt}: {n}")
    # print first 30 tensors
    print("\n=== first 40 tensors ===")
    for name,nd,ds,tt,off in tensors[:40]:
        print(f"  {name}: ndim={nd} dims={ds} type={tt} off={off}")
    print("\n=== .scale / .input_scale companions ===")
    for name,nd,ds,tt,off in tensors:
        if name.endswith(".scale") or name.endswith(".input_scale") or name.endswith("_n"):
            print(f"  {name}: dims={ds} type={tt}")
    print("\n=== NVFP4 (type 40) sample block layout ===")
    for name,nd,ds,tt,off in tensors:
        if tt == 40:
            print(f"  sample NVFP4: {name} dims={ds} off={off}")
            # read first 160 bytes of first block
            f.seek(data_start + off)
            blk = f.read(160)
            print(f"    bytes[0:16] (scales): {blk[0:16].hex()}")
            print(f"    bytes[16:20]: {blk[16:20].hex()}")
            print(f"    bytes[144:160]: {blk[144:160].hex()}")
            break
