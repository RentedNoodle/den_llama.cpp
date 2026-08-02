#!/usr/bin/env python3
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for name in ["output.weight","blk.0.attn_qkv.weight","blk.0.ffn_gate_exps.weight",
             "blk.0.ffn_up_exps.weight","blk.0.ffn_down_exps.weight",
             "blk.3.attn_q.weight","blk.3.attn_output.weight"]:
    for t in r.tensors:
        if t.name == name:
            ne = t.shape.tolist()
            n = 1
            for d in ne: n *= d
            print("%-32s type=%-6s shape=%s numel=%d nblocks(%d/256)=%d n_bytes=%d  ondisk/blk=%.4f" % (
                name, t.tensor_type.name, ne, n, n, n//256, t.n_bytes, t.n_bytes/(n//256)))
