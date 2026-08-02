#!/usr/bin/env python3
import numpy as np
from gguf import GGUFReader
r = GGUFReader("/mnt/i/models/Ornith-1.0-35B-NVFP4.gguf")
g = {t.name: t for t in r.tensors}
for nm in ["blk.39.ffn_gate_inp_shexp.weight", "blk.0.attn_norm.weight", "token_embd.weight", "blk.0.ffn_gate_inp.weight"]:
    t = g[nm]
    d = t.data
    print("%-32s data_type=%s data_shape=%s data_nbytes=%d tobytes_len=%d" % (
        nm, d.dtype, getattr(d, "shape", None), d.nbytes,
        len(np.asarray(d, dtype=np.uint8).tobytes())))
