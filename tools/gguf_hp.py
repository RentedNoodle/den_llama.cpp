#!/usr/bin/env python3
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
want = ["block_count","context_length","embedding_length","attention.head_count",
        "attention.head_count_kv","attention.layer_norm_rms_epsilon","expert_used_count",
        "attention.key_length","attention.value_length","expert_count",
        "expert_feed_forward_length","expert_shared_feed_forward_length",
        "ssm.state_size","ssm.inner_size","ssm.conv_kernel","ssm.time_step_rank",
        "ssm.group_count","rope.dimension_sections","rope.freq_base","rope.dimension_count",
        "full_attention_interval","general.file_type","general.quantization_version"]
for k, f in r.fields.items():
    for w in want:
        if k.endswith(w):
            val = f.contents[0] if hasattr(f.contents, "__len__") else f.contents
            print("%s = %s" % (k, val))
