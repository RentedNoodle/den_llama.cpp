#!/usr/bin/env python3
"""
Convert Qwen3.5 models (HF safetensors) to Den native format.

Supports: 0.8B, 2B, 4B, 9B (any Qwen3.5 family model).
Handles sharded safetensors via index.json.

Reads den_format.h definitions directly:
  - Header: 4096 bytes (DEN_HEADER_SIZE)
  - Tensor index: starts at 4096, 128-byte entries (DEN_TENSOR_ENTRY_SIZE)
  - Data: page-aligned after all entries

Hardware target (den_hw_target_t):
  1 = NVFP4, 2 = BF16, 3 = F32, 4 = F16

PORTED from C:\Den\den-nvfp4-optimizations\tools\den_convert_heretic.py
→ I:\den_llama.cpp\tools\den_convert_heretic.py
"""

import argparse
import json
import math
import os
import struct
import sys

# ────────────────────────────────────────────────────────────────────────────
# Format constants (from den_format.h)
# ────────────────────────────────────────────────────────────────────────────
DEN_MAGIC = 0x4E454400          # "DEN\0"
DEN_VERSION = 0x00050000        # Den format version
DEN_HEADER_SIZE = 4096
DEN_TENSOR_ENTRY_SIZE = 128
DEN_MAX_SLOTS = 256
DEN_LAYER_STRIDE = 32

# Architecture enum
DEN_ARCH_QWEN35 = 1

# Hardware target enum
DEN_TARGET_NVFP4 = 1
DEN_TARGET_BF16 = 2
DEN_TARGET_F32 = 3

# Global slot numbers
DEN_SLOT_TOKEN_EMBD = 0
DEN_SLOT_OUTPUT_NORM = 1
DEN_SLOT_OUTPUT = 2

# Per-layer base
def DEN_SLOT_LAYER_BASE(layer):
    return 3 + layer * DEN_LAYER_STRIDE

# Sub-slot offsets (minimal set needed for Qwen3.5)
DEN_SLOT_ATTN_Q = 1
DEN_SLOT_ATTN_K = 2
DEN_SLOT_ATTN_V = 3
DEN_SLOT_ATTN_O = 4
DEN_SLOT_ATTN_Q_NORM = 5
DEN_SLOT_ATTN_K_NORM = 6
DEN_SLOT_MOE_ROUTER = 7           # mlp.gate.weight (MoE router)
DEN_SLOT_MLP_GATE = 8
DEN_SLOT_MLP_UP = 9
DEN_SLOT_MLP_DOWN = 10
DEN_SLOT_EXPERT_GATE_UP = 11      # mlp.experts.gate_up_proj
DEN_SLOT_GDN_IN_PROJ_X = 12      # in_proj_a (alpha gate)
DEN_SLOT_GDN_IN_PROJ_Z = 13      # in_proj_z (output gate)
DEN_SLOT_GDN_OUT_PROJ = 14       # output projection
DEN_SLOT_GDN_A_LOG = 15          # SSM A parameter
DEN_SLOT_GDN_DT_BIAS = 16        # SSM dt bias
DEN_SLOT_GDN_CONV1D = 17         # conv1d weight
DEN_SLOT_GDN_NORM = 19           # GDN/output norm (linear_attn.norm)
DEN_SLOT_EXPERT_DOWN = 26        # mlp.experts.down_proj
DEN_SLOT_SHARED_GATE = 27        # mlp.shared_expert.gate_proj
DEN_SLOT_SHARED_UP = 28          # mlp.shared_expert.up_proj
DEN_SLOT_SHARED_DOWN = 29        # mlp.shared_expert.down_proj
DEN_SLOT_SHARED_GATE_W = 30      # mlp.shared_expert_gate
DEN_SLOT_INPUT_LAYERNORM = 20    # input layernorm
DEN_SLOT_POST_ATTN_NORM = 21     # post attention norm
DEN_SLOT_GDN_IN_PROJ_B = 25      # in_proj_b (beta gate)
DEN_SLOT_GDN_IN_PROJ_QKV = 24    # fused in_proj_qkv

# Tier flags
DEN_TFLAG_HOT_TIER = 1 << 0
DEN_TFLAG_WARM_TIER = 1 << 1
DEN_TFLAG_COLD_TIER = 1 << 2


# ═══════════════════════════════════════════════════════════════════════════
# Tensor name -> slot mapping
# ═══════════════════════════════════════════════════════════════════════════

def name_to_slot(hf_name):
    """Map a stripped HF tensor name to a (slot, hw_target, convert_f32) tuple."""
    # Global tensors
    if hf_name == 'embed_tokens.weight':
        return (DEN_SLOT_TOKEN_EMBD, DEN_TARGET_BF16, False)
    if hf_name == 'norm.weight':
        return (DEN_SLOT_OUTPUT_NORM, DEN_TARGET_F32, True)
    # Skip output.weight (tied with embed_tokens) — dengine uses embed_tokens for LM head
    if hf_name == 'output.weight':
        return (DEN_SLOT_OUTPUT, DEN_TARGET_BF16, False)
    if hf_name == 'lm_head.weight':
        return (DEN_SLOT_OUTPUT, DEN_TARGET_BF16, False)

    # Per-layer tensors
    parts = hf_name.split('.')
    # Expected formats:
    #   layers.N.linear_attn.X
    #   layers.N.self_attn.X
    #   layers.N.mlp.X
    #   layers.N.input_layernorm.weight
    #   layers.N.post_attention_layernorm.weight
    if not parts[0].startswith('layers'):
        raise ValueError(f"Unknown tensor: {hf_name}")

    layer = int(parts[1])
    base = DEN_SLOT_LAYER_BASE(layer)

    suffix = '.'.join(parts[2:])

    # Linear attention (GDN) tensors
    if suffix == 'linear_attn.A_log':
        return (base + DEN_SLOT_GDN_A_LOG, DEN_TARGET_F32, True)
    if suffix == 'linear_attn.dt_bias':
        return (base + DEN_SLOT_GDN_DT_BIAS, DEN_TARGET_F32, True)
    if suffix == 'linear_attn.conv1d.weight':
        return (base + DEN_SLOT_GDN_CONV1D, DEN_TARGET_BF16, False)
    if suffix == 'linear_attn.in_proj_qkv.weight':
        return (base + DEN_SLOT_GDN_IN_PROJ_QKV, DEN_TARGET_BF16, False)
    if suffix == 'linear_attn.in_proj_a.weight':
        return (base + DEN_SLOT_GDN_IN_PROJ_X, DEN_TARGET_BF16, False)
    if suffix == 'linear_attn.in_proj_b.weight':
        return (base + DEN_SLOT_GDN_IN_PROJ_B, DEN_TARGET_BF16, False)
    if suffix == 'linear_attn.in_proj_z.weight':
        return (base + DEN_SLOT_GDN_IN_PROJ_Z, DEN_TARGET_BF16, False)
    if suffix == 'linear_attn.out_proj.weight':
        return (base + DEN_SLOT_GDN_OUT_PROJ, DEN_TARGET_BF16, False)
    if suffix == 'linear_attn.norm.weight':
        return (base + DEN_SLOT_GDN_NORM, DEN_TARGET_F32, True)  # F32 for norm weights

    # Self-attention tensors
    if suffix == 'self_attn.q_proj.weight':
        return (base + DEN_SLOT_ATTN_Q, DEN_TARGET_BF16, False)
    if suffix == 'self_attn.k_proj.weight':
        return (base + DEN_SLOT_ATTN_K, DEN_TARGET_BF16, False)
    if suffix == 'self_attn.v_proj.weight':
        return (base + DEN_SLOT_ATTN_V, DEN_TARGET_BF16, False)
    if suffix == 'self_attn.o_proj.weight':
        return (base + DEN_SLOT_ATTN_O, DEN_TARGET_BF16, False)
    # Attention Q/K norms
    if suffix == 'self_attn.q_norm.weight':
        return (base + DEN_SLOT_ATTN_Q_NORM, DEN_TARGET_F32, True)
    if suffix == 'self_attn.k_norm.weight':
        return (base + DEN_SLOT_ATTN_K_NORM, DEN_TARGET_F32, True)

    # MoE expert tensors
    if suffix == 'mlp.gate.weight':
        return (base + DEN_SLOT_MOE_ROUTER, DEN_TARGET_BF16, False)
    if suffix == 'mlp.experts.gate_up_proj':
        return (base + DEN_SLOT_EXPERT_GATE_UP, DEN_TARGET_BF16, False)
    if suffix == 'mlp.experts.down_proj':
        return (base + DEN_SLOT_EXPERT_DOWN, DEN_TARGET_BF16, False)
    if suffix == 'mlp.shared_expert.gate_proj.weight':
        return (base + DEN_SLOT_SHARED_GATE, DEN_TARGET_BF16, False)
    if suffix == 'mlp.shared_expert.up_proj.weight':
        return (base + DEN_SLOT_SHARED_UP, DEN_TARGET_BF16, False)
    if suffix == 'mlp.shared_expert.down_proj.weight':
        return (base + DEN_SLOT_SHARED_DOWN, DEN_TARGET_BF16, False)
    if suffix == 'mlp.shared_expert_gate.weight':
        return (base + DEN_SLOT_SHARED_GATE_W, DEN_TARGET_BF16, False)

    # MLP tensors (always BF16)
    if suffix == 'mlp.gate_proj.weight':
        return (base + DEN_SLOT_MLP_GATE, DEN_TARGET_BF16, False)
    if suffix == 'mlp.up_proj.weight':
        return (base + DEN_SLOT_MLP_UP, DEN_TARGET_BF16, False)
    if suffix == 'mlp.down_proj.weight':
        return (base + DEN_SLOT_MLP_DOWN, DEN_TARGET_BF16, False)

    # Layer norms
    if suffix == 'input_layernorm.weight':
        return (base + DEN_SLOT_INPUT_LAYERNORM, DEN_TARGET_F32, True)
    if suffix == 'post_attention_layernorm.weight':
        return (base + DEN_SLOT_POST_ATTN_NORM, DEN_TARGET_F32, True)

    raise ValueError(f"Unmapped tensor: {hf_name}")


def read_safetensors_header(path):
    """Read safetensors JSON header and return (tensors_dict, data_start_offset)."""
    with open(path, 'rb') as f:
        hdr_len_bytes = f.read(8)
        hdr_len = struct.unpack('<Q', hdr_len_bytes)[0]
        hdr_json = f.read(hdr_len).decode('utf-8')
        hdr = json.loads(hdr_json)
        data_start = 8 + hdr_len
    return hdr, data_start


def read_tensor_data(f, safetensors_hdr, name, data_start):
    """Read raw bytes of a tensor from the safetensors file."""
    info = safetensors_hdr[name]
    dtype = info['dtype']
    data_offsets = info['data_offsets']
    start = data_start + data_offsets[0]
    end = data_start + data_offsets[1]
    f.seek(start)
    return f.read(end - start), dtype


def align_up(x, alignment):
    """Round x up to the nearest multiple of alignment."""
    return ((x + alignment - 1) // alignment) * alignment


def bf16_bytes_to_f32_bytes(bf16_data):
    """Convert BF16 little-endian bytes to F32 little-endian bytes.

    BF16 is the upper 16 bits of F32 with zeroed lower 16 bits.
    In little-endian F32 [byte0, byte1, byte2, byte3], the value is:
      byte0 | byte1<<8 | byte2<<16 | byte3<<24
    BF16 occupies byte2 and byte3 (the upper 16 bits).

    BF16 LE [b0, b1] maps to F32 LE [0x00, 0x00, b0, b1].
    """
    result = bytearray()
    for i in range(0, len(bf16_data), 2):
        result.append(0)                    # F32 byte 0 (low byte)
        result.append(0)                    # F32 byte 1
        result.append(bf16_data[i])         # F32 byte 2 = BF16 byte 0
        result.append(bf16_data[i + 1])     # F32 byte 3 = BF16 byte 1
    return bytes(result)


def load_qwen35_config(src_dir):
    """Load model hyperparams from config.json in src_dir."""
    config_path = os.path.join(src_dir, 'config.json')
    with open(config_path) as f:
        raw = json.load(f)

    # Qwen3.5 nests under text_config
    tc = raw.get('text_config', raw)

    rope_params = tc.get('rope_parameters', {})
    if isinstance(rope_params, dict):
        partial_rotary_factor = rope_params.get('partial_rotary_factor', 0.25)
        rope_theta = rope_params.get('rope_theta', 10000000.0)
    else:
        partial_rotary_factor = tc.get('partial_rotary_factor', 0.25)
        rope_theta = tc.get('rope_theta', 10000000.0)

    hidden_size = tc['hidden_size']
    n_heads = tc['num_attention_heads']
    head_dim = hidden_size // n_heads
    n_rot = int(n_heads * head_dim * partial_rotary_factor)

    n_vh = tc.get('linear_num_key_heads', 16)
    kd   = tc.get('linear_key_head_dim', 128)
    vd   = tc.get('linear_value_head_dim', kd)  # 128 for 0.8B/2B, 256 for 4B

    # MoE models use moe_intermediate_size; dense models use intermediate_size
    ffn_size = tc.get('intermediate_size') or tc.get('moe_intermediate_size', 0)
    n_experts = tc.get('num_experts', 0)
    n_experts_used = tc.get('num_experts_per_tok', 0)

    return {
        'n_layers': tc['num_hidden_layers'],
        'n_heads': n_heads,
        'n_kv_heads': tc['num_key_value_heads'],
        'hidden_size': hidden_size,
        'ffn_size': ffn_size,
        'vocab_size': tc['vocab_size'],
        'max_seq_len': tc.get('max_position_embeddings', 262144),
        'n_rot': n_rot,
        'n_experts': n_experts,
        'n_experts_used': n_experts_used,
        'rope_theta': rope_theta,
        'rms_norm_eps': tc.get('rms_norm_eps', 1e-6),
        # SSM parameters
        'ssm_state_size': kd,           # key dimension
        'ssm_value_size': vd,           # value dimension (128 for 0.8B/2B, 256 for 4B)
        'ssm_conv_kernel': tc.get('linear_conv_kernel_dim', 4),
        'ssm_group_count': n_vh,
        'ssm_inner_size': n_vh * kd,    # ssm_inner_size = n_vh * kd
        'ssm_time_step_rank': tc.get('dt_bias_dim', 16),
        'full_attention_interval': tc.get('full_attention_interval', 4),
    }


class ShardedSafetensors:
    """Handle single or sharded safetensors files.

    If index.json exists, load it. Otherwise treat as single-file.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.shard_files = {}       # shard_name -> (header, data_start)
        self.shard_handles = {}     # shard_name -> open file handle
        self.weight_map = {}        # tensor_name -> shard_name
        self.all_tensors = {}       # tensor_name -> info (merged from all shards)

        index_path = os.path.join(src_dir, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            self._load_sharded(index_path)
        else:
            self._load_single()

    def _load_single(self):
        """Single safetensors file (no index.json)."""
        sf_path = os.path.join(self.src_dir, 'model.safetensors')
        if not os.path.exists(sf_path):
            raise FileNotFoundError(f"No safetensors found in {self.src_dir}")
        hdr, data_start = read_safetensors_header(sf_path)
        self.shard_files[sf_path] = (hdr, data_start)
        self.shard_handles[sf_path] = open(sf_path, 'rb')
        for name in hdr:
            if name != '__metadata__':
                self.weight_map[name] = sf_path
                self.all_tensors[name] = hdr[name]

    def _load_sharded(self, index_path):
        """Sharded safetensors with index.json."""
        with open(index_path) as f:
            index = json.load(f)
        self.weight_map = index.get('weight_map', {})

        # Get unique shard files
        shard_names = sorted(set(self.weight_map.values()))
        for shard_name in shard_names:
            sf_path = os.path.join(self.src_dir, shard_name)
            if not os.path.exists(sf_path):
                raise FileNotFoundError(f"Shard not found: {sf_path}")
            hdr, data_start = read_safetensors_header(sf_path)
            self.shard_files[sf_path] = (hdr, data_start)
            self.shard_handles[sf_path] = open(sf_path, 'rb')
            for name in hdr:
                if name != '__metadata__':
                    self.all_tensors[name] = hdr[name]

        print(f"  Shards: {len(shard_names)} files, {len(self.all_tensors)} tensors total")

    def get_tensor_data(self, orig_name):
        """Read raw bytes for a tensor, using the correct shard."""
        shard_name = self.weight_map.get(orig_name)
        if shard_name is None:
            raise KeyError(f"Tensor {orig_name} not found in weight map")
        shard_path = os.path.join(self.src_dir, shard_name)
        hdr, data_start = self.shard_files[shard_path]
        f = self.shard_handles[shard_path]
        return read_tensor_data(f, hdr, orig_name, data_start)

    def close(self):
        for f in self.shard_handles.values():
            f.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main converter
# ═══════════════════════════════════════════════════════════════════════════

def convert(src_dir, dst_path):
    """Convert a Qwen3.5 HF model directory to .den format."""

    if not os.path.isdir(src_dir):
        print(f"ERROR: Source directory not found: {src_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load config ──────────────────────────────────────────────────────
    print("Loading model config...")
    cfg = load_qwen35_config(src_dir)
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Compute SSM inner size
    ssm_inner_size = cfg['ssm_group_count'] * cfg['ssm_state_size']

    # ── Open safetensors ─────────────────────────────────────────────────
    print("\nOpening safetensors...")
    st = ShardedSafetensors(src_dir)

    # Filter: only language model tensors (skip visual/vision encoder)
    language_tensors = {}
    for name, info in st.all_tensors.items():
        # Skip vision encoder tensors entirely
        if 'visual' in name.lower() or name.startswith('model.visual'):
            continue
        if name.startswith('model.language_model.'):
            stripped = name[len('model.language_model.'):]
        else:
            stripped = name
        language_tensors[stripped] = (name, info)

    print(f"  Language model tensors: {len(language_tensors)}")

    # ── Map each tensor to a slot ────────────────────────────────────────
    entries = []  # list of (slot, hw_target, stripped_name, shape, orig_name, convert_f32)
    skipped = []
    for stripped_name, (orig_name, info) in sorted(language_tensors.items()):
        try:
            slot, hw_target, convert_f32 = name_to_slot(stripped_name)
        except ValueError as e:
            skipped.append(str(e))
            continue

        shape = info['shape']
        entries.append((slot, hw_target, stripped_name, list(shape), orig_name, convert_f32))

    # Sort by slot number
    entries.sort(key=lambda e: e[0])

    if skipped:
        print(f"  Skipped {len(skipped)} tensors:")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    print(f"  Mapped tensor entries: {len(entries)}")
    for slot, hw, name, shape, _, _ in entries[:5]:
        hw_name = {2: 'BF16', 3: 'F32'}.get(hw, str(hw))
        print(f"    slot={slot:3d} {hw_name:4s} {str(shape):20s} {name}")
    if len(entries) > 5:
        print(f"    ... and {len(entries) - 5} more")

    # ── Derive vd from tensor shapes (config's linear_value_head_dim may be wrong) ──
    vd_derived = cfg.get('ssm_value_size', 0)
    for _, _, name, shape, _, _ in entries:
        if 'in_proj_z' in name:
            n_vh = cfg['ssm_group_count']
            vd_derived = shape[0] // n_vh
            if vd_derived != cfg.get('ssm_value_size', 0):
                print(f"  NOTE: Overriding config vd={cfg.get('ssm_value_size')} -> derived vd={vd_derived} from {name} shape={shape}")
            break
    cfg['ssm_value_size'] = vd_derived
    print(f"  GDN value dim (vd): {vd_derived} (kd={cfg['ssm_state_size']})")

    # ── Compute layout ───────────────────────────────────────────────────
    tensor_count = len(entries)
    index_offset = DEN_HEADER_SIZE  # always 4096
    index_size = tensor_count * DEN_TENSOR_ENTRY_SIZE
    data_offset = align_up(DEN_HEADER_SIZE + index_size, 4096)

    print(f"\n  Header: {DEN_HEADER_SIZE} bytes")
    print(f"  Index: {index_offset} + {index_size} = {index_offset + index_size}")
    print(f"  Data offset: {data_offset} (page-aligned: {data_offset % 4096 == 0})")

    # ── Read & convert all tensor data (stream to temp file for large models) ──
    import tempfile, gc
    tensor_data_offsets = []
    # Use a temp file as backing store so we don't need 2x RAM for large models
    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.den_tmp')
    _tmp_path = _tmp.name

    print("\nReading & converting tensor data...")
    for idx, (slot, hw_target, stripped_name, shape, orig_name, convert_f32) in enumerate(entries):
        raw_bytes, dtype = st.get_tensor_data(orig_name)

        # Handle 3D conv1d -> 2D squeeze
        if len(shape) == 3 and stripped_name.endswith('conv1d.weight'):
            shape[:] = [shape[0], shape[2]]

        is_bf16 = 'bf16' in dtype.lower() or 'bfloat16' in dtype.lower()
        if convert_f32 and is_bf16:
            converted = bf16_bytes_to_f32_bytes(raw_bytes)
        else:
            converted = raw_bytes

        tensor_data_offsets.append(_tmp.tell())
        _tmp.write(converted)
        del raw_bytes, converted
        if (idx + 1) % 50 == 0:
            print(f"  ... {idx + 1}/{tensor_count} tensors ({_tmp.tell() / 1024**3:.2f} GB)")
            gc.collect()

    total_data_size = _tmp.tell()
    _tmp.close()
    print(f"  Total: {tensor_count} tensors, {total_data_size / 1024**3:.2f} GB data")

    # Done with source files
    st.close()

    # ── Build header ────────────────────────────────────────────────────
    print("Building header...")
    header = bytearray(DEN_HEADER_SIZE)

    # Identification
    struct.pack_into('<I', header, 0, DEN_MAGIC)
    struct.pack_into('<I', header, 4, DEN_VERSION)
    struct.pack_into('<I', header, 8, DEN_ARCH_QWEN35)
    struct.pack_into('<I', header, 12, 0)  # flags

    # Model hyperparams (offset 16)
    struct.pack_into('<10I2f', header, 16,
                     cfg['n_layers'], cfg['n_heads'], cfg['n_kv_heads'],
                     cfg['hidden_size'], cfg['ffn_size'],
                     cfg['vocab_size'], cfg['max_seq_len'], cfg['n_rot'],
                     cfg['n_experts'], cfg['n_experts_used'],
                     cfg['rope_theta'], cfg['rms_norm_eps'])

    # SSM params (offset 64)
    struct.pack_into('<6I', header, 64,
                     cfg['ssm_state_size'], cfg['ssm_conv_kernel'],
                     ssm_inner_size, cfg['ssm_group_count'],
                     cfg['ssm_time_step_rank'], cfg['full_attention_interval'])

    # MTP layer count (offset 88) + SSM value dimension (offset 92)
    struct.pack_into('<I', header, 88, 0)  # mtp_layer_count = 0
    struct.pack_into('<I', header, 92, cfg.get('ssm_value_size', 0))  # vd

    # Data layout info (offset 104)
    struct.pack_into('<II', header, 104, tensor_count, index_offset)
    struct.pack_into('<Q', header, 112, data_offset)
    struct.pack_into('<Q', header, 120, total_data_size)

    # Tier counts (offset 128)
    struct.pack_into('<III', header, 128,
                     0, tensor_count, 0)  # all warm tier

    # Tier sizes (offset 144)
    struct.pack_into('<Q', header, 144, 0)                     # hot_tier_size
    struct.pack_into('<Q', header, 152, total_data_size)       # warm_tier_size
    struct.pack_into('<Q', header, 160, 0)                     # cold_tier_size

    # ── Build tensor index ──────────────────────────────────────────────
    print(f"Building tensor index ({tensor_count} entries)...")
    index_data = bytearray(tensor_count * DEN_TENSOR_ENTRY_SIZE)

    for idx, (slot, hw_target, stripped_name, shape, orig_name, convert_f32) in enumerate(entries):
        ent_off = idx * DEN_TENSOR_ENTRY_SIZE

        # Shape -> 4D padded
        dims = [0, 0, 0, 0]
        for i, d in enumerate(shape):
            dims[i] = d
        ndim = len(shape)

        # Numel
        numel = 1
        for d in shape:
            numel *= d

        data_off = tensor_data_offsets[idx]
        # Data size — derive from actual payload offsets to avoid converter mismatch
        if idx + 1 < len(tensor_data_offsets):
            data_size = tensor_data_offsets[idx + 1] - data_off
        else:
            data_size = total_data_size - data_off
        # Warn if size doesn't match expectations
        expected = numel * (4 if convert_f32 else 2)
        if data_size > 0 and data_size != expected:
            print(f"  WARN: slot {slot} size mismatch: payload={data_size} expected={expected}")

        # Tier flag
        if idx < 3:
            tier_flag = DEN_TFLAG_HOT_TIER
        elif idx >= tensor_count - 3:
            tier_flag = DEN_TFLAG_COLD_TIER
        else:
            tier_flag = DEN_TFLAG_WARM_TIER

        # Write entry
        struct.pack_into('<IIII', index_data, ent_off,
                         slot, hw_target, ndim, tier_flag)
        struct.pack_into('<4q', index_data, ent_off + 16,
                         dims[0], dims[1], dims[2], dims[3])
        struct.pack_into('<Q', index_data, ent_off + 48, numel)
        struct.pack_into('<Q', index_data, ent_off + 56, data_off)
        struct.pack_into('<Q', index_data, ent_off + 64, data_size)
        struct.pack_into('<Q', index_data, ent_off + 72, 0)  # scale_offset
        struct.pack_into('<Q', index_data, ent_off + 80, 0)  # scale_size
        struct.pack_into('<IIII', index_data, ent_off + 88, 0, 0, 0, 0)
        struct.pack_into('<Q', index_data, ent_off + 104, 0)  # norm_offset
        struct.pack_into('<I', index_data, ent_off + 112, 0)  # norm_size
        struct.pack_into('<III', index_data, ent_off + 116, 0, 0, 0)  # kernel params

    # ── Write .den file ──────────────────────────────────────────────────
    print(f"\nWriting {dst_path}...")
    with open(dst_path, 'wb') as f:
        f.write(header)
        f.write(index_data)
        pad_size = data_offset - (DEN_HEADER_SIZE + index_size)
        if pad_size > 0:
            f.write(b'\x00' * pad_size)

        # Chunked write from temp file: single f.write() of 8+ GB exceeds Win32.
        CHUNK = 64 * 1024 * 1024  # 64 MB
        with open(_tmp_path, 'rb') as dt:
            while True:
                chunk = dt.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
        f.flush()
        os.fsync(f.fileno())

    # Clean up temp file
    os.unlink(_tmp_path)

    expected_size = data_offset + total_data_size
    actual_size = os.path.getsize(dst_path)
    print(f"  File size: {actual_size} bytes ({actual_size / 1024**3:.2f} GB)")
    print(f"  Data region: {total_data_size} bytes ({total_data_size / 1024**3:.2f} GB)")
    if actual_size != expected_size:
        print(f"  ERROR: File truncated! Expected {expected_size}, got {actual_size} "
              f"(missing {expected_size - actual_size} bytes)")
        sys.exit(1)

    # ── Verify ───────────────────────────────────────────────────────────
    print("\nVerifying file...")
    with open(dst_path, 'rb') as f:
        magic = struct.unpack_from('<I', f.read(4))[0]
        assert magic == DEN_MAGIC, f"Bad magic: 0x{magic:08X}"
        f.seek(0)

        hdr = f.read(DEN_HEADER_SIZE)
        ver = struct.unpack_from('<I', hdr, 4)[0]
        arch = struct.unpack_from('<I', hdr, 8)[0]
        n_layers_r = struct.unpack_from('<I', hdr, 16)[0]
        hidden_r = struct.unpack_from('<I', hdr, 28)[0]
        tcnt_r = struct.unpack_from('<I', hdr, 104)[0]
        data_off_r = struct.unpack_from('<Q', hdr, 112)[0]
        total_sz_r = struct.unpack_from('<Q', hdr, 120)[0]
        fai_r = struct.unpack_from('<I', hdr, 84)[0]

        print(f"  magic=0x{magic:08X} version=0x{ver:08X} arch={arch}")
        print(f"  n_layers={n_layers_r} hidden={hidden_r}")
        print(f"  tensor_count={tcnt_r} data_off={data_off_r} total_data={total_sz_r}")
        print(f"  full_attention_interval={fai_r}")

        # Verify slot ordering
        idx_data = f.read(tcnt_r * DEN_TENSOR_ENTRY_SIZE)
        last_slot = -1
        slot_errors = 0
        for i in range(tcnt_r):
            off = i * DEN_TENSOR_ENTRY_SIZE
            slot = struct.unpack_from('<I', idx_data, off)[0]
            if slot <= last_slot:
                print(f"  ERROR: Slot not monotonic at entry {i}: slot={slot} <= last={last_slot}")
                slot_errors += 1
            last_slot = slot

        payload_check = f.read(16)
        print(f"  Payload first bytes: {payload_check[:16].hex()}")

    if slot_errors == 0:
        print("  Slot ordering: OK (monotonic)")
    else:
        print(f"  Slot ordering: {slot_errors} errors")

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Qwen3.5 HF model to .den format')
    parser.add_argument('--src', required=True, help='Source HF model directory (with config.json + safetensors)')
    parser.add_argument('--dst', required=True, help='Output .den file path')
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        print(f"ERROR: Source directory not found: {args.src}", file=sys.stderr)
        sys.exit(1)

    convert(args.src, args.dst)
