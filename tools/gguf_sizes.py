#!/usr/bin/env python3
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
tot = 0; tot_w = 0; tot_scale = 0; nvfp4 = 0; f32 = 0; bf16 = 0
for t in r.tensors:
    nb = t.n_bytes
    tot += nb
    if t.name.endswith(".scale") or t.name.endswith(".input_scale"):
        tot_scale += nb
    else:
        tot_w += nb
        if t.tensor_type.name == "NVFP4": nvfp4 += nb
        elif t.tensor_type.name == "F32": f32 += nb
        else: bf16 += nb
print("total bytes: %.3f GB" % (tot/1e9))
print("weights only: %.3f GB (NVFP4 %.3f, F32 %.3f, BF16 %.3f)" % (tot_w/1e9, nvfp4/1e9, f32/1e9, bf16/1e9))
print("scale companions: %.3f GB" % (tot_scale/1e9))
