#!/usr/bin/env python3
import sys, struct, numpy as np
sys.path.insert(0, "/root/den_final/tools")
from gguf_to_den_slot import expand_nvfp4_blocks, fold_scale, NVFP4_MEM, NVFP4_DISPATCH_GEMV
from gguf import GGUFReader
r = GGUFReader("/mnt/i/models/Ornith-1.0-35B-NVFP4.gguf")
t = next(x for x in r.tensors if x.name == "blk.0.attn_qkv.weight")
data = np.asarray(t.data, dtype=np.uint8)
print("t.data shape:", data.shape, "size:", data.size)
raw = data.reshape(-1)          # flatten to 1D
nblocks = 65536
assert raw.size == nblocks * 144

vec = expand_nvfp4_blocks(raw, nblocks)
print("vec shape:", vec.shape)

# C-style reference for first 2 blocks
ref = np.zeros((2, NVFP4_MEM), dtype=np.uint8)
for b in range(2):
    srcp = raw[b*144:(b+1)*144]
    dst = ref[b]
    for sb in range(4):
        s = srcp[sb*36:(sb+1)*36]
        dst[sb*4:sb*4+4] = s[0:4]
        dst[16+sb*32:16+sb*32+32] = s[4:36]
    dst[144:160] = 0
print("first-2-blocks match C loop:", np.array_equal(vec[:2], ref))
print("bytes144:160 zero:", np.all(vec[:, 144:160] == 0))
print("scales[0:16] nonzero:", np.any(vec[0, 0:16] != 0))
print("nibbles[16:144] nonzero:", np.any(vec[0, 16:144] != 0))

sc = next(x for x in r.tensors if x.name == "blk.0.attn_qkv.scale")
gval = np.asarray(sc.data, dtype=np.float32).copy()
print("attn_qkv.scale =", gval)
fold_scale(vec, gval)
exp = struct.unpack("<f", vec[0, 152:156].tobytes())[0]
print("folded blk[152:156] = %f (expect %f)" % (exp, gval[0]))
print("dispatch blk[148] = %#x (expect %#x)" % (vec[0,148], NVFP4_DISPATCH_GEMV))

t2 = next(x for x in r.tensors if x.name == "blk.0.ffn_gate_exps.weight")
raw2 = np.asarray(t2.data, dtype=np.uint8).reshape(-1)
nb2 = 1048576
vec2 = expand_nvfp4_blocks(raw2, nb2)
sc2 = next(x for x in r.tensors if x.name == "blk.0.ffn_gate_exps.scale")
gval2 = np.asarray(sc2.data, dtype=np.float32).copy()
fold_scale(vec2, gval2)
bpe = nb2 // 256
ok = True
for e in [0, 1, 100, 255]:
    got = struct.unpack("<f", vec2[e*bpe, 152:156].tobytes())[0]
    m = abs(got - gval2[e]) < 1e-6
    ok &= m
    print("expert %d scale blk[152:156]=%f expect=%f %s" % (e, got, gval2[e], "OK" if m else "MISMATCH"))
print("per-expert fold all OK:", ok)
