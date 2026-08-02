#!/usr/bin/env python3
import sys, struct, numpy as np
sys.path.insert(0, "/root/den_final/tools")
from gguf_to_den_slot import expand_nvfp4_blocks, fold_scale, _scale_of, NVFP4_MEM, QK_NVFP4
from gguf import GGUFReader
r = GGUFReader("/mnt/i/models/Ornith-1.0-35B-NVFP4.gguf")
g = {x.name: x for x in r.tensors}
gt = g["blk.0.ffn_gate_exps.weight"]; ut = g["blk.0.ffn_up_exps.weight"]
g_ne = gt.shape.tolist(); u_ne = ut.shape.tolist()
n_exp = g_ne[2]; bpe = g_ne[0] // QK_NVFP4
ng = int(np.prod(g_ne)) // QK_NVFP4; nu = int(np.prod(u_ne)) // QK_NVFP4
gb = expand_nvfp4_blocks(np.asarray(gt.data, dtype=np.uint8).reshape(-1), ng)
ub = expand_nvfp4_blocks(np.asarray(ut.data, dtype=np.uint8).reshape(-1), nu)
fold_scale(gb, _scale_of(g, "blk.0.ffn_gate_exps.scale"))
fold_scale(ub, _scale_of(g, "blk.0.ffn_up_exps.scale"))
gb3 = gb.reshape(n_exp, g_ne[1], bpe, NVFP4_MEM)
ub3 = ub.reshape(n_exp, u_ne[1], bpe, NVFP4_MEM)
fused = np.concatenate([gb3, ub3], axis=1).reshape(-1, NVFP4_MEM)
print("fused blocks:", fused.shape[0], "expect", 2*ng)
# check fused block stream order: for expert e, j2 in [0,1024): 
#   block stream index = e*1024*bpe + j2*bpe + k
# first 8 blocks = expert 0, j2=0 (gate) -> equals gate blocks [0:8]
print("expert0 j2=0 gate == gate[0:8]:", np.array_equal(fused[0:8], gb[0:8]))
# j2=512 (up) blocks for expert 0 = up blocks [0:8]
off = 0*1024*bpe + 512*bpe
print("expert0 j2=512 up == up[0:8]:", np.array_equal(fused[off:off+8], ub[0:8]))
# expert1 j2=0 gate == gate[8:16]
off2 = 1*1024*bpe
print("expert1 j2=0 gate == gate[8:16]:", np.array_equal(fused[off2:off2+8], gb[1*512*8:1*512*8+8]))
# expert1 j2=512 up == up[8:16]
off3 = 1*1024*bpe + 512*bpe
print("expert1 j2=512 up == up[8:16]:", np.array_equal(fused[off3:off3+8], ub[1*512*8:1*512*8+8]))
