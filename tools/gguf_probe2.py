#!/usr/bin/env python3
import sys
from gguf import GGUFReader
from collections import Counter

path = sys.argv[1]
r = GGUFReader(path)
print("GGUF: n_tensors=%d fields=%s" % (len(r.tensors), list(r.fields.keys())))
for k, f in r.fields.items():
    try:
        if f.contents is not None:
            vals = f.contents
            s = str(vals)
            if len(s) > 130: s = s[:130] + "..."
            print("  KV %s = %s" % (k, s))
    except Exception as e:
        print("  KV %s = <err %s>" % (k, e))
print()
ttc = Counter(t.tensor_type.name for t in r.tensors)
for tt, n in sorted(ttc.items()): print("  type %s: %d" % (tt, n))
print()
print("=== first 30 tensors ===")
for t in r.tensors[:30]:
    print("  %s: %s ne=%s off=%d" % (t.name, t.tensor_type, t.shape.tolist(), t.data_offset))
print()
print("=== companions ===")
for t in r.tensors:
    n = t.name
    if n.endswith(".scale") or n.endswith(".input_scale") or n.endswith("_n"):
        print("  %s: %s ne=%s" % (n, t.tensor_type, t.shape.tolist()))
print()
for t in r.tensors:
    if t.tensor_type.name == "NVFP4":
        raw = t.data[:160].tobytes()
        print("=== NVFP4 first block layout: %s ne=%s ===" % (t.name, t.shape.tolist()))
        print("  data[0:16]   : " + raw[0:16].hex())
        print("  data[16:32]  : " + raw[16:32].hex())
        print("  data[32:48]  : " + raw[32:48].hex())
        print("  data[48:64]  : " + raw[48:64].hex())
        print("  data[144:160]: " + raw[144:160].hex())
        print("  tensor nbytes=%d" % t.n_bytes)
        break
