#!/usr/bin/env python3
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
for L in [0,1,2,3,4,7,8,39]:
    print("=== layer %d ===" % L)
    for t in r.tensors:
        if t.name.startswith("blk.%d." % L) or t.name == ("blk.%d" % L):
            print("  %s: %s ne=%s" % (t.name, t.tensor_type, t.shape.tolist()))
