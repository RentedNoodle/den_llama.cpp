#!/usr/bin/env python3
# Recompute the .den plan from the GGUF to find the size discrepancy.
import sys, re, numpy as np
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
g = {t.name: t for t in r.tensors}
QK=256
SUB = {"attn_q.weight":1,"attn_k.weight":2,"attn_v.weight":3,"attn_output.weight":4,
 "attn_q_norm.weight":5,"attn_k_norm.weight":6,"ffn_gate_inp.weight":7,
 "ffn_gate.weight":8,"ffn_up.weight":9,"ffn_down.weight":10,
 "ssm_alpha.weight":12,"attn_gate.weight":13,"ssm_out.weight":14,"ssm_a":15,
 "ssm_dt.bias":16,"ssm_conv1d.weight":17,"ssm_norm.weight":19,"attn_norm.weight":20,
 "post_attention_norm.weight":21,"attn_qkv.weight":24,"ssm_beta.weight":25,
 "ffn_gate_shexp.weight":27,"ffn_up_shexp.weight":28,"ffn_down_shexp.weight":29,
 "ffn_gate_inp_shexp.weight":30}
def slot_of(name):
    if name=="token_embd.weight": return 0
    if name=="output_norm.weight": return 1
    if name=="output.weight": return 2
    if name.endswith(".scale") or name.endswith(".input_scale"): return None
    m=re.match(r"blk\.(\d+)\.(.*)$",name)
    if not m: return None
    layer=int(m.group(1)); tail=m.group(2)
    base=3+layer*32
    if tail in ("ffn_gate_exps.weight","ffn_up_exps.weight"): return base+11
    if tail=="ffn_down_exps.weight": return base+26
    if tail in SUB: return base+SUB[tail]
    return None
slots={}
for t in r.tensors:
    s=slot_of(t.name)
    if s is None: continue
    slots.setdefault(s,[]).append(t.name)
tot_nv=tot_f32=tot_bf16=0
n_nv=n_f=n_b=0
for s,lst in sorted(slots.items()):
    sub=s%32 if s>=3 else s
    if sub==17:
        ne=g[lst[0]].shape.tolist(); n=int(np.prod(ne)); sz=n*2; tot_bf16+=sz; n_b+=1
    elif sub==11:
        gn=g[lst[0]]; un=g[lst[1]]
        ne=[gn.shape.tolist()[0], gn.shape.tolist()[1]+un.shape.tolist()[1], gn.shape.tolist()[2]]
        n=int(np.prod(ne)); sz=(n//QK)*160; tot_nv+=sz; n_nv+=1
    else:
        t=g[lst[0]]; ne=t.shape.tolist(); n=int(np.prod(ne))
        if t.tensor_type.name=="NVFP4":
            sz=(n//QK)*160; tot_nv+=sz; n_nv+=1
        elif t.tensor_type.name=="BF16":
            sz=n*2; tot_bf16+=sz; n_b+=1
        else:
            sz=n*4; tot_f32+=sz; n_f+=1
print("slots=%d NVFP4=%.3fGB(%d) F32=%.3fGB(%d) BF16=%.3fGB(%d) total=%.3fGB"
      %(len(slots),tot_nv/1e9,n_nv,tot_f32/1e9,n_f,tot_bf16/1e9,n_b,(tot_nv+tot_f32+tot_bf16)/1e9))

# print largest 12 slots
rows = []
for s,lst in sorted(slots.items()):
    sub=s%32 if s>=3 else s
    if sub==17:
        ne=g[lst[0]].shape.tolist(); n=int(np.prod(ne)); sz=n*2
    elif sub==11:
        gn=g[lst[0]]; un=g[lst[1]]
        n=int(np.prod(gn.shape.tolist()))+int(np.prod(un.shape.tolist()))
        sz=(n//QK)*160
    else:
        t=g[lst[0]]; ne=t.shape.tolist(); n=int(np.prod(ne))
        if t.tensor_type.name=="NVFP4": sz=(n//QK)*160
        elif t.tensor_type.name=="BF16": sz=n*2
        else: sz=n*4
    rows.append((sz,s,lst[0]))
for sz,s,nm in sorted(rows, reverse=True)[:12]:
    print("slot=%d sub=%d size=%.1fMB %s" % (s, s%32 if s>=3 else s, sz/1e6, nm))
