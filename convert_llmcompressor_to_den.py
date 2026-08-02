#!/usr/bin/env python3
"""
Convert LLM-Compressor / AutoRound NVFP4 exports to dengine .den format.

Bridges three NVFP4 storage formats into the Den NULLGLASS tile format:
  - 'modelopt':           NVIDIA ModelOpt (AxionML, surogate, cosmicproc)
  - 'compressed_tensors': llmcompressor / vLLM compressed-tensors
  - 'mlx_nvfp4':          Apple MLX native NVFP4

AutoRound with --format llm_compressor exports as 'compressed_tensors' format.
The converter auto-detects the source format from safetensors metadata/tensor names.

Usage:
  python tools/convert_llmcompressor_to_den.py \\
      --safetensors model.safetensors \\
      --config config.json \\
      --output model.den

  # Sharded safetensors (provide the directory containing model.safetensors.index.json)
  python tools/convert_llmcompressor_to_den.py \\
      --safetensors /path/to/model_dir/model.safetensors-00001-of-00002.safetensors \\
      --config /path/to/model_dir/config.json \\
      --output model.den

  # With explicit quant config
  python tools/convert_llmcompressor_to_den.py \\
      --safetensors model.safetensors \\
      --config config.json \\
      --output model.den \\
      --quant-config quant_config.json
"""

import argparse
import json
import os
import re
import struct
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# .den format constants (from denrt_format.h)
# ═══════════════════════════════════════════════════════════════════════════

DEN_MAGIC = 0x4E454400
DEN_VERSION = 0x00050000
DEN_HEADER_SIZE = 4096
DEN_TENSOR_ENTRY_SIZE = 128

# Architecture IDs
DEN_ARCH_QWEN35 = 1       # Qwen3.5 dense (GDN + attention)
DEN_ARCH_QWEN36_MOE = 2   # Qwen3.6 MoE

# Hardware targets
DEN_TARGET_NVFP4 = 1
DEN_TARGET_BF16 = 2
DEN_TARGET_F32 = 3

# Slot constants
DEN_SLOT_TOKEN_EMBD = 0
DEN_SLOT_OUTPUT_NORM = 1
DEN_SLOT_OUTPUT = 2
DEN_LAYER_STRIDE = 32

def DEN_SLOT_LAYER_BASE(layer: int) -> int:
    return 3 + layer * DEN_LAYER_STRIDE

# NVFP4 tile geometry
TILE_ELEMS = 256          # elements per tile (K dimension)
GROUP_SIZE = 16           # elements per NVFP4 group
BLOCKS_PER_TILE = TILE_ELEMS // GROUP_SIZE  # 16
NIBBLES_PER_TILE = TILE_ELEMS // 2          # 128
NVFP4_TILE_SIZE = 160     # bytes per tile

# Tile header offsets
TILE_OFF_SCALES = 0       # 16 bytes: UE4M3 scale codes
TILE_OFF_NIBBLES = 16     # 128 bytes: packed E2M1 nibbles
TILE_OFF_NORM = 144       # 4 bytes: float32 tile_norm
TILE_OFF_META = 148       # dispatch + format byte

# Dispatch codes (tile[148])
DEN_DISPATCH_OMMA_4X = 0x10
DEN_TILE_SCALE_E4M3 = 0x80  # bit 7: E4M3 scale format

# UE4M3 values (16-level lookup table)
UE4M3_LUT = np.array([
    0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
    1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875
], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Tensor name → .den slot mapping
# ═══════════════════════════════════════════════════════════════════════════

def name_to_slot(hf_name: str, arch: int = DEN_ARCH_QWEN35) -> int:
    """Map HF tensor name to .den slot ID.

    Supports both Qwen3.5 (dense GDN) and Qwen3.6 (MoE) architectures.
    The stripped name should NOT have a 'model.' prefix.
    """
    # Global tensors
    if hf_name == 'embed_tokens.weight' or hf_name == 'model.embed_tokens.weight':
        return DEN_SLOT_TOKEN_EMBD
    if hf_name == 'norm.weight' or hf_name == 'model.norm.weight':
        return DEN_SLOT_OUTPUT_NORM
    if hf_name == 'lm_head.weight' or hf_name == 'output.weight':
        return DEN_SLOT_OUTPUT

    # Strip 'model.' or 'model.language_model.' prefix
    stripped = hf_name
    if stripped.startswith('model.language_model.'):
        stripped = stripped[len('model.language_model.'):]
    elif stripped.startswith('model.'):
        stripped = stripped[len('model.'):]

    # Per-layer tensors
    parts = stripped.split('.')
    if len(parts) < 3 or not parts[0].startswith('layers'):
        raise ValueError(f"Unknown tensor: {hf_name}")

    layer = int(parts[1])
    base = DEN_SLOT_LAYER_BASE(layer)
    suffix = '.'.join(parts[2:])

    # ── GDN (linear attention) tensors ──
    if 'linear_attn' in suffix or 'ssm' in suffix:
        return _map_gdn(suffix, base)

    # ── Self-attention tensors ──
    if 'self_attn' in suffix:
        return _map_attention(suffix, base)

    # ── MoE tensors ──
    if arch == DEN_ARCH_QWEN36_MOE:
        moe_slot = _map_moe(suffix, base)
        if moe_slot is not None:
            return moe_slot

    # ── MLP tensors (dense) ──
    if 'mlp' in suffix:
        return _map_mlp(suffix, base)

    # ── Layer norms ──
    if suffix == 'input_layernorm.weight':
        return base + 20
    if suffix == 'post_attention_layernorm.weight':
        return base + 21
    if suffix == 'pre_mlp_layernorm.weight':
        return base + 22
    if suffix == 'post_mlp_layernorm.weight':
        return base + 23

    raise ValueError(f"Unmapped tensor: {hf_name}")


def _map_gdn(suffix: str, base: int) -> int:
    if suffix.endswith('A_log') or suffix.endswith('.A_log'):
        return base + 15
    if suffix.endswith('dt_bias') or suffix.endswith('.dt_bias'):
        return base + 16
    if suffix.endswith('conv1d.weight') or suffix.endswith('.conv1d.weight'):
        return base + 17
    if suffix.endswith('in_proj_qkv.weight') or suffix.endswith('.in_proj_qkv.weight'):
        return base + 24
    if suffix.endswith('in_proj_a.weight') or suffix.endswith('.in_proj_a.weight'):
        return base + 12
    if suffix.endswith('in_proj_b.weight') or suffix.endswith('.in_proj_b.weight'):
        return base + 25
    if suffix.endswith('in_proj_z.weight') or suffix.endswith('.in_proj_z.weight'):
        return base + 13
    if suffix.endswith('out_proj.weight') or suffix.endswith('.out_proj.weight'):
        return base + 14
    if suffix.endswith('norm.weight') or suffix.endswith('.norm.weight'):
        return base + 19
    if suffix.endswith('d_proj.weight') or suffix.endswith('.d_proj.weight'):
        return base + 18
    raise ValueError(f"Unmapped GDN tensor: {suffix}")


def _map_attention(suffix: str, base: int) -> int:
    if suffix.endswith('q_proj.weight') or suffix.endswith('.q_proj.weight'):
        return base + 1
    if suffix.endswith('k_proj.weight') or suffix.endswith('.k_proj.weight'):
        return base + 2
    if suffix.endswith('v_proj.weight') or suffix.endswith('.v_proj.weight'):
        return base + 3
    if suffix.endswith('o_proj.weight') or suffix.endswith('.o_proj.weight'):
        return base + 4
    if suffix.endswith('q_norm.weight') or suffix.endswith('.q_norm.weight'):
        return base + 5
    if suffix.endswith('k_norm.weight') or suffix.endswith('.k_norm.weight'):
        return base + 6
    if suffix.endswith('qkv.weight') or suffix.endswith('.qkv.weight'):
        return base + 0
    raise ValueError(f"Unmapped attention tensor: {suffix}")


def _map_moe(suffix: str, base: int) -> Optional[int]:
    # Router
    if suffix.endswith('gate.weight') or 'mlp.gate' in suffix:
        return base + 7
    # Expert gate+up (fused) — [n_experts, intermediate*2, hidden]
    if suffix.endswith('gate_up_proj.weight') or suffix.endswith('gate_up_proj.weight'):
        return base + 11
    # Expert down
    if suffix.endswith('down_proj.weight') and 'shared' not in suffix:
        return base + 26
    # Shared expert tensors
    if 'shared_expert.gate_proj' in suffix or 'shared_expert.gate' in suffix:
        return base + 27
    if 'shared_expert.up_proj' in suffix:
        return base + 28
    if 'shared_expert.down_proj' in suffix:
        return base + 29
    if 'shared_expert_gate' in suffix:
        return base + 30
    return None


def _map_mlp(suffix: str, base: int) -> int:
    if suffix.endswith('gate_proj.weight') or suffix.endswith('.gate_proj.weight'):
        return base + 8
    if suffix.endswith('up_proj.weight') or suffix.endswith('.up_proj.weight'):
        return base + 9
    if suffix.endswith('down_proj.weight') or suffix.endswith('.down_proj.weight'):
        return base + 10
    raise ValueError(f"Unmapped MLP tensor: {suffix}")


# ═══════════════════════════════════════════════════════════════════════════
# NVFP4 format detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_format(safetensors_path: str) -> str:
    """Detect NVFP4 storage format from safetensors file.

    Returns one of: 'modelopt', 'compressed_tensors', 'mlx_nvfp4', 'unknown'
    """
    import safetensors
    with safetensors.safe_open(safetensors_path, framework="np") as f:
        keys = list(f.keys())
        if '__metadata__' in keys:
            keys.remove('__metadata__')

    # Format A: compressed-tensors uses *_packed + *_scale
    packed_keys = [k for k in keys if k.endswith('_packed')]
    scale_suffix_keys = [k for k in keys if k.endswith('_scale')
                         and not k.endswith('_global_scale')
                         and not k.endswith('_input_scale')]
    if packed_keys and scale_suffix_keys:
        return 'compressed_tensors'

    # Format B: ModelOpt uses *.weight + *.weight_scale + (optional *.input_scale)
    weight_keys = [k for k in keys if k.endswith('.weight')]
    if not weight_keys:
        return 'unknown'

    sample = weight_keys[0]
    base = sample.replace('.weight', '')

    with safetensors.safe_open(safetensors_path, framework="np") as f:
        has_input_scale = f'{base}.input_scale' in f.keys()
        has_weight_scale_2 = f'{base}.weight_scale_2' in f.keys()
        has_weight_scale = f'{base}.weight_scale' in f.keys()

    if has_input_scale or has_weight_scale_2:
        return 'modelopt'

    if has_weight_scale:
        with safetensors.safe_open(safetensors_path, framework="np") as f:
            try:
                scale_dtype = f.get_dtype(f'{base}.weight_scale')
                scale_dtype_str = str(scale_dtype)
            except Exception:
                scale_dtype_str = ''
        if 'F8_E4M3' in scale_dtype_str or 'F8' in scale_dtype_str:
            return 'compressed_tensors'
        if 'BF16' in scale_dtype_str or 'F32' in scale_dtype_str:
            return 'mlx_nvfp4'

    # Fallback: broader patterns
    if any('_scale' in k for k in keys):
        return 'compressed_tensors'

    return 'unknown'


# ═══════════════════════════════════════════════════════════════════════════
# Scale decode helpers
# ═══════════════════════════════════════════════════════════════════════════

def e4m3_to_f32(data: np.ndarray) -> np.ndarray:
    """Decode F8_E4M3 to float32. Matches NVIDIA NVFP4 block scale format."""
    u8 = np.frombuffer(data.tobytes(), dtype=np.uint8).reshape(data.shape)
    sign = (u8 >> 7).astype(np.float32)
    exp = ((u8 >> 3) & 0xF).astype(np.int32)
    man = (u8 & 0x7).astype(np.int32)
    normal = exp > 0
    val = np.zeros_like(sign)
    val[normal] = np.ldexp(1.0 + man[normal] / 8.0, exp[normal] - 7)
    val[~normal] = np.ldexp(man[~normal] / 8.0, -6)
    return val * (1.0 - 2.0 * sign)


def ue8m0_to_f32(ue8m0: np.ndarray) -> np.ndarray:
    """Decode UE8M0 to float32. value = 2^(byte - 127)."""
    result = np.zeros_like(ue8m0, dtype=np.float32)
    nonzero = ue8m0 > 0
    result[nonzero] = np.ldexp(1.0, ue8m0[nonzero].astype(np.int32) - 127)
    result[ue8m0 == 0] = np.float32(2.0 ** -127)
    return result


def ue4m3_encode(value: float) -> int:
    """Encode float32 scale to UE4M3 byte (nearest-neighbor LUT)."""
    idx = int(np.argmin(np.abs(UE4M3_LUT - value)))
    return idx


def bf16_to_f32(bf16_bytes: bytes) -> np.ndarray:
    """Convert BF16 bytes to float32 numpy array."""
    as_u16 = np.frombuffer(bf16_bytes, dtype=np.uint16)
    as_u32 = as_u16.astype(np.uint32) << 16
    return as_u32.view(np.float32)


def bf16_bytes_from_f32(f32_data: np.ndarray) -> bytes:
    """Convert float32 numpy array to packed BF16 bytes (truncation)."""
    as_u32 = f32_data.view(np.uint32)
    as_u16 = (as_u32 >> 16).astype(np.uint16)
    return as_u16.tobytes()


# ═══════════════════════════════════════════════════════════════════════════
# NVFP4 tensor extraction (normalized nibbles + scales)
# ═══════════════════════════════════════════════════════════════════════════

def extract_nvfp4_tensors(
    safetensors_path: str, format_type: str
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, int, int]]:
    """Extract NVFP4 tensors normalized to (nibbles_uint8, scales_f32).

    Yields: (tensor_name, nibbles [N, K/2] uint8, scales [N, K/16] f32, N, K)
    """
    if format_type == 'modelopt':
        yield from _extract_modelopt(safetensors_path)
    elif format_type == 'compressed_tensors':
        yield from _extract_compressed_tensors(safetensors_path)
    elif format_type == 'mlx_nvfp4':
        yield from _extract_mlx(safetensors_path)
    else:
        raise ValueError(f"Unknown NVFP4 format: {format_type}")


def _extract_modelopt(path: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray, int, int]]:
    import safetensors
    with safetensors.safe_open(path, framework="np") as f:
        keys = list(f.keys())
    weight_keys = [k for k in keys if k.endswith('.weight')
                   and not k.startswith('__')]

    for weight_key in weight_keys:
        base = weight_key.replace('.weight', '')
        scale_key = f'{base}.weight_scale'
        s2_key = f'{base}.weight_scale_2'
        is_key = f'{base}.input_scale'

        with safetensors.safe_open(path, framework="np") as f:
            if scale_key not in f.keys():
                continue

            nibbles = f.get_tensor(weight_key)  # uint8 [N, K/2]
            scale_raw = f.get_tensor(scale_key)  # F8_E4M3

            # E4M3 → F32
            scale_f32 = e4m3_to_f32(scale_raw)

            # Bake in weight_scale_2 (global multiplier)
            if s2_key in f.keys():
                s2 = f.get_tensor(s2_key)
                if s2.ndim == 0 or s2.size == 1:
                    scale_f32 *= float(s2)
                else:
                    scale_f32 *= s2.reshape(-1, 1)

            # Bake in input_scale
            if is_key in f.keys():
                is_val = f.get_tensor(is_key)
                if is_val.ndim == 0 or is_val.size == 1:
                    scale_f32 *= float(is_val)
                else:
                    scale_f32 *= is_val.reshape(-1, 1)

            N = nibbles.shape[0]
            K = nibbles.shape[1] * 2
            yield weight_key, nibbles, scale_f32.astype(np.float32), N, K


def _extract_compressed_tensors(path: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray, int, int]]:
    import safetensors
    import torch

    with safetensors.safe_open(path, framework="pt") as f:
        keys = list(f.keys())

    # Convention B: *_packed + *_scale (+ *_global_scale)
    packed_keys = [k for k in keys if k.endswith('_packed')]
    scale_keys_map = {k[:-len('_scale')]: k for k in keys
                      if k.endswith('_scale') and not k.endswith('_global_scale')}

    for packed_key in packed_keys:
        base = packed_key[:-len('_packed')]
        scale_name = f'{base}_scale'
        global_scale_name = f'{base}_global_scale'

        with safetensors.safe_open(path, framework="pt") as f_pt:
            if scale_name not in keys:
                continue

            nibbles = f_pt.get_tensor(packed_key).numpy()  # uint8

            # F8_E4M3 bytes are UE8M0 codes in compressed-tensors format
            scale_raw = f_pt.get_tensor(scale_name)
            scale_ue8m0 = scale_raw.view(torch.uint8).numpy()
            scale_f32 = ue8m0_to_f32(scale_ue8m0)

            if global_scale_name in keys:
                gs = f_pt.get_tensor(global_scale_name)
                gs_float = float(gs.ravel()[0]) if gs.size > 0 else 1.0
                scale_f32 *= gs_float

            N = nibbles.shape[0]
            K = nibbles.shape[1] * 2
            yield packed_key, nibbles, scale_f32.astype(np.float32), N, K

    # Convention A fallback: *.weight + *.weight_scale (when no *_packed)
    if not packed_keys:
        weight_keys = [k for k in keys if k.endswith('.weight')
                       and not k.endswith('_global_scale')]
        for weight_key in weight_keys:
            base = weight_key.replace('.weight', '')
            scale_key = f'{base}.weight_scale'

            with safetensors.safe_open(path, framework="pt") as f_pt:
                if scale_key not in keys:
                    continue

                nibbles = f_pt.get_tensor(weight_key).numpy()
                scale_raw = f_pt.get_tensor(scale_key)
                scale_ue8m0 = scale_raw.view(torch.uint8).numpy()
                scale_f32 = ue8m0_to_f32(scale_ue8m0)

                # Optional input_scale / weight_scale_2 (rare in compressed-tensors)
                is_key = f'{base}.input_scale'
                s2_key = f'{base}.weight_scale_2'
                if is_key in keys:
                    is_val = f_pt.get_tensor(is_key)
                    scale_f32 *= (float(is_val) if is_val.ndim == 0 or is_val.size == 1
                                  else is_val.reshape(-1, 1))
                if s2_key in keys:
                    s2 = f_pt.get_tensor(s2_key)
                    scale_f32 *= (float(s2) if s2.ndim == 0 or s2.size == 1
                                  else s2.reshape(-1, 1))

                N = nibbles.shape[0]
                K = nibbles.shape[1] * 2
                yield weight_key, nibbles, scale_f32.astype(np.float32), N, K


def _extract_mlx(path: str) -> Iterator[Tuple[str, np.ndarray, np.ndarray, int, int]]:
    import safetensors
    with safetensors.safe_open(path, framework="np") as f:
        keys = list(f.keys())

    weight_keys = [k for k in keys if k.endswith('.weight')]

    for weight_key in weight_keys:
        base = weight_key.replace('.weight', '')
        scale_key = f'{base}.weight_scale'

        with safetensors.safe_open(path, framework="np") as f:
            if scale_key not in f.keys():
                continue
            nibbles = f.get_tensor(weight_key)
            scale_raw = f.get_tensor(scale_key)

            if scale_raw.dtype == np.uint16:
                scale_f32 = bf16_to_f32(scale_raw.tobytes()).reshape(scale_raw.shape)
            else:
                scale_f32 = scale_raw.astype(np.float32)

            # MLX uses 32-element blocks. For 16-element block OMMA compat,
            # duplicate each scale to cover 2 blocks.
            if scale_f32.shape[1] * 32 == nibbles.shape[1] * 2:
                scale_f32 = np.repeat(scale_f32, 2, axis=1)

            N = nibbles.shape[0]
            K = nibbles.shape[1] * 2
            yield weight_key, nibbles, scale_f32.astype(np.float32), N, K


# ═══════════════════════════════════════════════════════════════════════════
# NULLGLASS tile packing
# ═══════════════════════════════════════════════════════════════════════════

def pack_nullglass_tile(nibbles_128b: np.ndarray, scales_f32_16: np.ndarray,
                        k_stride: int = 1) -> np.ndarray:
    """Pack 256 E2M1 elements + 16 scales into a 160-byte NULLGLASS tile.

    Args:
        nibbles_128b: [128] uint8 — 256 E2M1 values packed 2 per byte
        scales_f32_16: [16] float32 — one scale per 16-element block
        k_stride: tile[149] value indicating effective K stride

    Returns:
        [160] uint8 tile
    """
    tile = np.zeros(NVFP4_TILE_SIZE, dtype=np.uint8)

    # Normalize scales for OMMA range (max scale < 1.875 = max UE4M3)
    tile_norm_val = float(max(np.max(scales_f32_16) / 1.4375, 1e-6))
    scales_norm = scales_f32_16 / tile_norm_val

    # Clamp to valid UE4M3 LUT range
    scales_norm = np.clip(scales_norm, 0.0, 1.875)

    # Encode 16 UE4M3 scale bytes
    for g in range(BLOCKS_PER_TILE):
        tile[TILE_OFF_SCALES + g] = ue4m3_encode(float(scales_norm[g]))

    # Copy E2M1 nibbles (bytes 16..143)
    tile[TILE_OFF_NIBBLES:TILE_OFF_NIBBLES + 128] = nibbles_128b[:128]

    # Store tile_norm at bytes 144-147 (float32 LE)
    tile[TILE_OFF_NORM:TILE_OFF_NORM + 4] = np.frombuffer(
        struct.pack('<f', tile_norm_val), dtype=np.uint8)

    # Dispatch byte at 148: OMMA_4X | E4M3 scale format
    tile[TILE_OFF_META] = DEN_DISPATCH_OMMA_4X | DEN_TILE_SCALE_E4M3

    # K_stride at 149
    tile[149] = k_stride & 0xFF

    return tile


def convert_nvfp4_to_tiles(nibbles: np.ndarray, scales: np.ndarray,
                           logical_K: int) -> np.ndarray:
    """Convert NVFP4 weight tensor to NULLGLASS tile array.

    Args:
        nibbles: [N, K/2] uint8 — packed E2M1 values
        scales: [N, K/16] float32 — per-block scales
        logical_K: logical input dimension (output dimension of transposed weight)

    Returns:
        [N, n_tiles_per_row, 160] uint8 — tile array
    """
    N, packed_K = nibbles.shape
    logical_K_infer = packed_K * 2  # K from nibbles

    # Pad K to tile boundary
    K_padded = ((logical_K + TILE_ELEMS - 1) // TILE_ELEMS) * TILE_ELEMS
    n_tiles_per_row = K_padded // TILE_ELEMS

    # Compute k_stride
    k_stride = K_padded // TILE_ELEMS  # number of tiles per row
    if k_stride > 4:
        k_stride = 4  # cap at 4 (max K_stride value)

    n_total_tiles = N * n_tiles_per_row
    all_tiles = np.zeros((n_total_tiles, NVFP4_TILE_SIZE), dtype=np.uint8)

    n_scale_groups = scales.shape[1]  # K/16

    tile_idx = 0
    for row in range(N):
        for tile_start in range(0, K_padded, TILE_ELEMS):
            # Extract 128 bytes of nibbles for this tile
            nibble_start = tile_start // 2
            nibble_end = min(nibble_start + NIBBLES_PER_TILE, packed_K)
            nibbles_slice = np.zeros(NIBBLES_PER_TILE, dtype=np.uint8)
            if nibble_start < packed_K:
                actual = nibble_end - nibble_start
                nibbles_slice[:actual] = nibbles[row, nibble_start:nibble_start + actual]

            # Extract 16 scales for this tile
            scale_start = tile_start // GROUP_SIZE
            scale_end = min(scale_start + BLOCKS_PER_TILE, n_scale_groups)
            scale_slice = np.ones(BLOCKS_PER_TILE, dtype=np.float32)
            if scale_start < n_scale_groups:
                actual = scale_end - scale_start
                scale_slice[:actual] = scales[row, scale_start:scale_start + actual]

            all_tiles[tile_idx] = pack_nullglass_tile(nibbles_slice, scale_slice, k_stride)
            tile_idx += 1

    return all_tiles


# ═══════════════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: str) -> dict:
    """Load model config from config.json."""
    with open(config_path) as f:
        raw = json.load(f)

    # Qwen3.5/3.6 nests under text_config
    tc = raw.get('text_config', raw)
    is_moe = tc.get('num_experts', 0) > 0
    arch = DEN_ARCH_QWEN36_MOE if is_moe else DEN_ARCH_QWEN35

    # RoPE params
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
    kd = tc.get('linear_key_head_dim', 128)
    vd = tc.get('linear_value_head_dim', kd)

    ssm_inner_size = n_vh * kd

    return {
        'arch': arch,
        'n_layers': tc['num_hidden_layers'],
        'n_heads': n_heads,
        'n_kv_heads': tc.get('num_key_value_heads', n_heads),
        'hidden_size': hidden_size,
        'ffn_size': tc['intermediate_size'],
        'vocab_size': tc['vocab_size'],
        'max_seq_len': tc.get('max_position_embeddings', 262144),
        'n_rot': n_rot,
        'n_experts': tc.get('num_experts', 0),
        'n_experts_used': tc.get('num_experts_per_tok', 0) if is_moe else 0,
        'rope_theta': rope_theta,
        'rms_norm_eps': tc.get('rms_norm_eps', 1e-6),
        'ssm_state_size': kd,
        'ssm_value_size': vd,
        'ssm_conv_kernel': tc.get('linear_conv_kernel_dim', 4),
        'ssm_group_count': n_vh,
        'ssm_inner_size': ssm_inner_size,
        'ssm_time_step_rank': tc.get('dt_bias_dim', 16),
        'full_attention_interval': tc.get('full_attention_interval', 4),
    }


def load_quant_config(quant_config_path: Optional[str],
                      metadata: Optional[dict] = None) -> dict:
    """Load quantization configuration from file or safetensors metadata.

    Returns a dict with keys: 'format', 'group_size', 'scale_dtype'.
    """
    result = {
        'format': 'unknown',
        'group_size': 16,
        'scale_dtype': 'float32',
    }

    if quant_config_path and os.path.exists(quant_config_path):
        with open(quant_config_path) as f:
            qc = json.load(f)
        # LLM-Compressor quant_config.json format
        if 'quantization' in qc:
            qz = qc['quantization']
            result['format'] = qz.get('format', 'unknown')
            result['group_size'] = qz.get('group_size', 16)
            result['scale_dtype'] = qz.get('scale_dtype', 'float32')
        elif 'config_groups' in qc:
            # AutoGPTQ/AutoRound format
            for group in qc.get('config_groups', {}).values():
                if 'quant_method' in group:
                    result['format'] = group['quant_method']
                if 'group_size' in group:
                    result['group_size'] = group['group_size']
                break

    if metadata:
        # LLM-Compressor embeds quant info in safetensors metadata
        if 'quantization_config' in metadata:
            qc = metadata['quantization_config']
            if isinstance(qc, dict):
                result['format'] = qc.get('quant_method', result['format'])
                result['group_size'] = qc.get('group_size', result['group_size'])
        if 'compressed_tensors_config' in metadata:
            ct = metadata['compressed_tensors_config']
            if isinstance(ct, dict) and 'weight_compression' in ct:
                wc = ct['weight_compression']
                result['format'] = 'compressed_tensors'
                result['group_size'] = wc.get('group_size', result['group_size'])
                result['scale_dtype'] = wc.get('scale_dtype', result['scale_dtype'])

    return result


def derive_ssm_value_size(entries: list, cfg: dict) -> int:
    """Derive actual SSM value dimension from in_proj_z tensor shapes.

    The config's linear_value_head_dim can be wrong; override from actual tensor shapes.
    """
    n_vh = cfg.get('ssm_group_count', 16)
    vd_derived = cfg.get('ssm_value_size', 0)
    for _, _, _, shape, name in entries:
        if 'in_proj_z' in name:
            vd_derived = shape[0] // n_vh
            if vd_derived != cfg.get('ssm_value_size', 0):
                print(f"  NOTE: Overriding ssm_value_size {cfg.get('ssm_value_size')} "
                      f"-> {vd_derived} from {name} shape={shape}")
            break
    return vd_derived


# ═══════════════════════════════════════════════════════════════════════════
# ShardedSafetensors — handle single or sharded safetensors
# ═══════════════════════════════════════════════════════════════════════════

class ShardedSafetensors:
    """Handle single or sharded safetensors files with key abstraction."""

    def __init__(self, src: str):
        self.src = src
        self.weight_map: Dict[str, str] = {}
        self.all_tensors: Dict[str, dict] = {}
        self.metadata: Optional[dict] = None
        self._shard_headers: Dict[str, tuple] = {}
        self._shard_handles: Dict[str, object] = {}

        if os.path.isdir(src):
            self._load_directory(src)
        else:
            self._load_file(src)

    def _load_directory(self, directory: str):
        """Load from a directory containing model.safetensors.index.json."""
        index_path = os.path.join(directory, 'model.safetensors.index.json')
        if os.path.exists(index_path):
            self._load_from_index(index_path)
        else:
            # Try loading all .safetensors files in the directory
            import glob
            sf_files = sorted(glob.glob(os.path.join(directory, '*.safetensors')))
            if not sf_files:
                raise FileNotFoundError(f"No safetensors files found in {directory}")
            for sf_path in sf_files:
                self._load_file(sf_path, check_dup=False)

    def _load_from_index(self, index_path: str):
        """Load sharded safetensors via index.json."""
        directory = os.path.dirname(index_path)
        with open(index_path) as f:
            index = json.load(f)
        self.weight_map = index.get('weight_map', {})

        if '__metadata__' in index:
            self.metadata = index['__metadata__']

        shard_names = sorted(set(self.weight_map.values()))
        for shard_name in shard_names:
            sf_path = os.path.join(directory, shard_name)
            if not os.path.exists(sf_path):
                raise FileNotFoundError(f"Shard not found: {sf_path}")
            self._load_file(sf_path, check_dup=False)

        print(f"  Shards: {len(shard_names)} files, {len(self.all_tensors)} tensors")

    def _load_file(self, sf_path: str, check_dup: bool = True):
        """Load a single safetensors file."""
        import safetensors
        with safetensors.safe_open(sf_path, framework="np") as f:
            keys = list(f.keys())

            for name in keys:
                if name == '__metadata__':
                    md = f.metadata()
                    if md:
                        self.metadata = md
                    continue
                if check_dup and name in self.all_tensors:
                    continue
                self.weight_map[name] = sf_path
                self.all_tensors[name] = f.get_slice(name)

    def has_tensor(self, name: str) -> bool:
        return name in self.all_tensors

    def get_tensor_data(self, name: str) -> np.ndarray:
        """Read tensor data and return numpy array."""
        if name not in self.all_tensors:
            raise KeyError(f"Tensor {name} not found")
        slc = self.all_tensors[name]
        data = slc[:]  # Load full tensor
        return data

    def close(self):
        pass  # safetensors safe_open handles file lifecycle


# ═══════════════════════════════════════════════════════════════════════════
# Firewall patterns — tensors that should stay BF16/F32
# ═══════════════════════════════════════════════════════════════════════════

FIREWALL_PATTERNS = [
    r"embed_tokens\.weight",
    r"lm_head\.weight",
    r"output\.weight",
    r".*linear_attn\.in_proj_qkv\.weight",
    r".*linear_attn\.in_proj_z\.weight",
    r".*linear_attn\.in_proj_a\.weight",
    r".*linear_attn\.in_proj_b\.weight",
    r".*linear_attn\.out_proj\.weight",
    r".*linear_attn\.conv1d\.weight",
    r".*linear_attn\.norm\.weight",
    r".*self_attn\.q_proj\.weight",
    r".*self_attn\.o_proj\.weight",
    r".*self_attn\.k_norm\.weight",
    r".*self_attn\.q_norm\.weight",
    r".*mlp\.gate$",
    r".*shared_expert_gate$",
]

F32_PINNED = [
    "norm", "ssm_", "A_log", "dt_bias", "bias", "gate",
]

# MoE tensors that should always be BF16 (not quantized to NVFP4)
MOE_BF16_PATTERNS = [
    r".*mlp\.gate\.weight",      # router
    r".*shared_expert_gate\.weight",
    r".*shared_expert\.gate_proj\.weight",
    r".*shared_expert\.up_proj\.weight",
    r".*shared_expert\.down_proj\.weight",
]


def is_firewalled(name: str) -> bool:
    return any(re.match(p, name) for p in FIREWALL_PATTERNS)


def is_f32_pinned(name: str) -> bool:
    return any(p in name for p in F32_PINNED)


def is_moe_bf16(name: str) -> bool:
    return any(re.match(p, name) for p in MOE_BF16_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════
# Main conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert(safetensors_path: str, config_path: str, output_path: str,
            quant_config_path: Optional[str] = None,
            bf16_ref_path: Optional[str] = None,
            force_format: Optional[str] = None) -> int:
    """Main conversion entry point."""

    # ── Load config ──
    print("Loading model config...")
    cfg = load_config(config_path)
    arch = cfg['arch']
    is_moe = arch == DEN_ARCH_QWEN36_MOE
    arch_name = "Qwen3.6 MoE" if is_moe else "Qwen3.5 Dense"
    print(f"  Architecture: {arch_name}")
    print(f"  Layers: {cfg['n_layers']}, Hidden: {cfg['hidden_size']}, "
          f"FFN: {cfg['ffn_size']}, Vocab: {cfg['vocab_size']}")
    if is_moe:
        print(f"  Experts: {cfg['n_experts']} (top-{cfg['n_experts_used']})")

    # ── Open safetensors ──
    print("\nOpening safetensors...")
    st = ShardedSafetensors(safetensors_path)
    print(f"  Found {len(st.all_tensors)} tensors")

    # Read metadata
    metadata = st.metadata
    if metadata:
        print(f"  Metadata present: {len(metadata)} keys")

    # ── Detect NVFP4 format ──
    if force_format:
        nvfp4_format = force_format
        print(f"\nNVFP4 format: {nvfp4_format} (forced)")
    else:
        # Probe from first safetensors file
        first_sf = list(st.weight_map.values())[0] if st.weight_map else safetensors_path
        if os.path.isdir(safetensors_path):
            first_sf = os.path.join(safetensors_path,
                                    os.path.basename(list(st.weight_map.values())[0]))
        nvfp4_format = detect_format(first_sf)
        print(f"\nNVFP4 format: {nvfp4_format}")

    # ── Load quant config ──
    quant_cfg = load_quant_config(quant_config_path, metadata)
    if quant_cfg['format'] != 'unknown':
        print(f"  Quant config: format={quant_cfg['format']}, "
              f"group_size={quant_cfg['group_size']}, "
              f"scale_dtype={quant_cfg['scale_dtype']}")

    # ── Load reference BF16 weights for firewalled tensors ──
    bf16_ref: Dict[str, np.ndarray] = {}
    if bf16_ref_path:
        print(f"\nLoading reference BF16 from: {bf16_ref_path}")
        bf16_st = ShardedSafetensors(bf16_ref_path)
        for name in st.all_tensors:
            ref_name = name
            if ref_name not in bf16_st.all_tensors:
                # Try with model.language_model prefix
                if ref_name.startswith('model.'):
                    ref_name = 'model.language_model.' + ref_name[6:]
            if ref_name in bf16_st.all_tensors:
                bf16_ref[name] = bf16_st.get_tensor_data(ref_name)
        print(f"  Loaded {len(bf16_ref)} reference BF16 tensors")

    # ── Build tensor entries ──
    entries: List[Tuple[int, int, bytes, list, str]] = []
    nvfp4_count = 0
    bf16_count = 0
    f32_count = 0
    skipped = 0

    quantized_weight_names = set()

    print("\nProcessing tensors...")

    # First pass: find quantized tensors
    for name in st.all_tensors:
        if name.endswith('.weight'):
            base = name.replace('.weight', '')
            if f'{base}.weight_scale' in st.all_tensors:
                quantized_weight_names.add(name)
        elif name.endswith('_packed'):
            base = name[:-len('_packed')]
            if f'{base}_scale' in st.all_tensors:
                quantized_weight_names.add(name)

    # Process each tensor
    for name in st.all_tensors:
        # Skip metadata and scale-only tensors
        if name == '__metadata__':
            continue
        if any(s in name for s in ['weight_scale', 'input_scale', '_global_scale']):
            continue
        if name.endswith('_scale') and not name.endswith('.weight_scale'):
            continue

        data = st.get_tensor_data(name)
        dtype = str(data.dtype)

        # Detect if this weight is NVFP4 quantized
        is_quantized = name in quantized_weight_names

        # Determine target hardware
        if is_quantized:
            if is_firewalled(name):
                # Firewalled: keep as BF16/F32
                target = 'F32' if is_f32_pinned(name) else 'BF16'
            elif is_f32_pinned(name):
                target = 'F32'
            elif is_moe and is_moe_bf16(name):
                target = 'BF16'
            else:
                target = 'NVFP4'
        elif 'float32' in dtype or 'f32' in dtype or dtype == '<f4':
            target = 'F32'
        elif 'int8' in dtype:
            target = 'BF16'  # unusual; treat as passthrough
        else:
            # Default: BF16
            target = 'BF16'

        # Get shape and determine if this is a valid weight tensor
        shape = list(data.shape)
        if not shape:
            skipped += 1
            continue
        if len(shape) < 2 and target == 'NVFP4':
            print(f"  WARN: {name} shape={shape} is 1D, cannot be NVFP4, forcing BF16")
            target = 'BF16'

        # Skip biases, norms with small dimensions, etc.
        if len(shape) == 1 and 'weight' not in name and 'bias' in name:
            skipped += 1
            continue

        # ── Convert ──
        entry_shape = list(shape)
        hw_target = DEN_TARGET_BF16

        if target == 'NVFP4':
            # Extract NVFP4 nibbles + scales via direct per-tensor lookup
            # (handles all three NVFP4 formats by finding companion scale tensors)
            if len(shape) < 2:
                print(f"  WARN: {name} shape={shape} is 1D, cannot be NVFP4, forcing BF16")
                target = 'BF16'
            else:
                result = _extract_nvfp4_direct(
                    data, shape, name, st, nvfp4_format, is_moe=is_moe,
                    n_experts=cfg.get('n_experts', 0))
                if result:
                    nibbles, scales_f32, N, K = result
                else:
                    print(f"  WARN: Cannot extract NVFP4 data for {name}, keeping as BF16")
                    target = 'BF16'

        if target == 'NVFP4':
            # Logical K from the logical shape (last dimension = K = input dim)
            logical_K = entry_shape[-1] if len(entry_shape) >= 2 else K

            # Flatten nibbles and scales to 2D for tiling
            if nibbles.ndim > 2:
                nibbles_2d = nibbles.reshape(-1, nibbles.shape[-1])
            else:
                nibbles_2d = nibbles
            if scales_f32.ndim > 2:
                scales_2d = scales_f32.reshape(-1, scales_f32.shape[-1])
            else:
                scales_2d = scales_f32

            # Tile all rows (each row = K logical elements)
            all_tiles = convert_nvfp4_to_tiles(nibbles_2d, scales_2d, logical_K)

            # Update entry_shape to logical (unpacked) dimensions.
            # The packed safetensors shape [N, K/2] becomes [N, K]
            # or [n_experts, intermediate, K/2] becomes [n_experts, intermediate, K].
            # The last dimension is always the packed K/2 — multiply by 2.
            entry_shape = list(shape)
            if entry_shape:
                entry_shape[-1] *= 2  # packed K/2 → logical K
            raw_bytes = all_tiles.tobytes()
            hw_target = DEN_TARGET_NVFP4
            nvfp4_count += 1

            raw_bytes = all_tiles.tobytes()
            hw_target = DEN_TARGET_NVFP4
            nvfp4_count += 1

            # Use BF16 reference if available (for firewalled tensors that got quantized)
            if name in bf16_ref:
                ref_data = bf16_ref[name]
                raw_bytes = bf16_bytes_from_f32(ref_data.astype(np.float32))
                hw_target = DEN_TARGET_BF16
                entry_shape = list(ref_data.shape)
                bf16_count += 1
                nvfp4_count -= 1

        elif target == 'F32':
            if 'bfloat16' in dtype or 'bf16' in dtype:
                raw_bytes = data.astype(np.float32).tobytes()
            elif 'float32' in dtype:
                raw_bytes = data.astype(np.float32).tobytes()
            else:
                raw_bytes = data.astype(np.float32).tobytes()
            hw_target = DEN_TARGET_F32
            f32_count += 1

        else:  # BF16
            if 'bfloat16' in dtype or 'bf16' in dtype:
                # Native BF16 — grab bytes via uint16 view
                raw_bytes = data.view(np.uint16).tobytes()
            elif 'float32' in dtype:
                # F32 → BF16 truncation
                as_u32 = data.view(np.uint32)
                as_u16 = (as_u32 >> 16).astype(np.uint16)
                raw_bytes = as_u16.tobytes()
            else:
                raw_bytes = data.astype(np.float32).tobytes()
            hw_target = DEN_TARGET_BF16
            bf16_count += 1

        # Map to slot
        try:
            slot = name_to_slot(name, arch)
        except ValueError as e:
            print(f"  SKIP: {e}")
            skipped += 1
            continue

        entries.append((slot, hw_target, raw_bytes, entry_shape, name))

    # Final sort by slot
    entries.sort(key=lambda e: e[0])

    # Derive vd
    ssm_value_size = derive_ssm_value_size(entries, cfg)

    print(f"\n  Packed: {nvfp4_count} NVFP4 + {bf16_count} BF16 + {f32_count} F32 "
          f"({skipped} skipped)")
    print(f"  Total entries: {len(entries)}")

    # ── Write .den file ──
    print(f"\nWriting: {output_path}")
    num_tensors = len(entries)
    index_end = DEN_HEADER_SIZE + num_tensors * DEN_TENSOR_ENTRY_SIZE
    data_offset = ((index_end + 4095) // 4096) * 4096

    with open(output_path, "wb") as f:
        # ── Header (4096 bytes) ──
        hdr_parts = []

        # Base header (offsets 0-15)
        hdr_parts.append(struct.pack("<IIII", DEN_MAGIC, DEN_VERSION, arch, 0))

        # Model hyperparams (offsets 16-63)
        hdr_parts.append(struct.pack("<IIIIIIII",
                                      cfg['n_layers'], cfg['n_heads'],
                                      cfg['n_kv_heads'], cfg['hidden_size'],
                                      cfg['ffn_size'], cfg['vocab_size'],
                                      cfg['max_seq_len'], cfg['n_rot']))
        hdr_parts.append(struct.pack("<II", cfg['n_experts'], cfg['n_experts_used']))
        hdr_parts.append(struct.pack("<ff", cfg['rope_theta'], cfg['rms_norm_eps']))

        # SSM params (offsets 64-87)
        hdr_parts.append(struct.pack("<IIIIII",
                                      cfg['ssm_state_size'], cfg['ssm_conv_kernel'],
                                      cfg['ssm_inner_size'], cfg['ssm_group_count'],
                                      cfg['ssm_time_step_rank'],
                                      cfg['full_attention_interval']))

        # SSM value size, MTP (offsets 88-103)
        hdr_parts.append(struct.pack("<II", 0, ssm_value_size))  # mtp_layer_count, ssm_value_size
        hdr_parts.append(struct.pack("<II", 0, 0))  # _padding

        # Data layout (offsets 104-127)
        hdr_parts.append(struct.pack("<IIQ", num_tensors, DEN_HEADER_SIZE, data_offset))
        hdr_parts.append(struct.pack("<Q", 0))  # total_data_size placeholder

        # Tier metadata (offsets 128-167)
        hdr_parts.append(struct.pack("<III", 0, 0, 0))  # tier counts
        hdr_parts.append(struct.pack("<QQQ", 0, 0, 0))  # tier sizes

        header = b''.join(hdr_parts)
        header = header.ljust(DEN_HEADER_SIZE, b'\x00')
        f.write(header)

        # ── Tensor entries ──
        data_pos = data_offset
        for slot, hw_target, raw_bytes, shape, name in entries:
            ndim = min(len(shape), 4)
            dims = [0, 0, 0, 0]
            for i, d in enumerate(shape[:4]):
                dims[i] = d
            numel = int(np.prod(shape))

            if hw_target == DEN_TARGET_NVFP4:
                data_size = len(raw_bytes)
                n_tiles = data_size // NVFP4_TILE_SIZE
                tile_k = TILE_ELEMS
                tile_n = 0
                scale_count = n_tiles * BLOCKS_PER_TILE
            else:
                data_size = len(raw_bytes)
                n_tiles = 0
                tile_k = 0
                tile_n = 0
                scale_count = 0

            entry = struct.pack("<IIII", slot, hw_target, ndim, 0)
            entry += struct.pack("<qqqq", dims[0], dims[1], dims[2], dims[3])
            entry += struct.pack("<Q", numel)
            entry += struct.pack("<QQ", data_pos - data_offset, data_size)
            entry += struct.pack("<QQ", 0, 0)  # scale_offset, scale_size
            entry += struct.pack("<IIII", tile_k, tile_n, n_tiles, scale_count)
            entry += struct.pack("<QI", 0, 0)  # norm_offset, norm_size
            entry += struct.pack("<II", TILE_ELEMS, 0)  # block_size=256, grid_size=0
            entry += struct.pack("<I", 0)  # padding
            assert len(entry) == DEN_TENSOR_ENTRY_SIZE, \
                f"Entry {len(entry)} != {DEN_TENSOR_ENTRY_SIZE}"

            f.write(entry)
            data_pos += data_size

        # Pad to data offset
        pad = data_offset - f.tell()
        if pad > 0:
            f.write(b'\x00' * pad)

        # ── Tensor data ──
        total_data_size = 0
        for slot, hw_target, raw_bytes, shape, name in entries:
            f.write(raw_bytes)
            total_data_size += len(raw_bytes)

    # ── Update total_data_size in header ──
    with open(output_path, "r+b") as f:
        hdr_bytes = bytearray(f.read(DEN_HEADER_SIZE))
        struct.pack_into("<Q", hdr_bytes, 120, total_data_size)
        f.seek(0)
        f.write(hdr_bytes)

    file_size = os.path.getsize(output_path)
    print(f"\n  Written: {file_size:,} bytes ({num_tensors} tensors)")
    print(f"  Data: {total_data_size:,} bytes at offset {data_offset}")
    print(f"  File size: {file_size / 1024**3:.2f} GB")
    print("\nDone.")

    return 0


def _resolve_sf_path(name: str, st: ShardedSafetensors) -> str:
    """Resolve a tensor name to its safetensors file path."""
    sf_path = st.weight_map.get(name, '')
    if sf_path and os.path.exists(sf_path):
        return sf_path
    if hasattr(st, 'src') and os.path.isdir(st.src):
        candidate = os.path.join(st.src, sf_path) if sf_path else ''
        if candidate and os.path.exists(candidate):
            return candidate
    # Use source itself if it's a file
    if hasattr(st, 'src') and os.path.isfile(st.src):
        return st.src
    return name


def _extract_nvfp4_direct(data: np.ndarray, shape: list, name: str,
                          st: ShardedSafetensors,
                          fmt: str, is_moe: bool = False,
                          n_experts: int = 0
                          ) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """Extract NVFP4 nibbles + scales from a single tensor by finding companion scales.

    Handles all three NVFP4 formats (modelopt, compressed_tensors, mlx_nvfp4)
    by looking up companion scale tensors (`*.weight_scale`, `*_scale`, etc.)
    in the same safetensors shard as the weight.

    For MoE expert tensors (shape=[n_experts, ...]), scales are looked up
    from per-expert scale tensors, handling the added first dimension.

    Returns:
        (nibbles_uint8, scales_f32, N, K) or None on failure
    """
    import safetensors
    dtype = str(data.dtype)

    # Need uint8 data for packed E2M1 nibbles
    if 'uint8' not in dtype and 'int8' not in dtype:
        return None

    # Resolve safetensors file path for this tensor
    sf_path = _resolve_sf_path(name, st)
    if not os.path.exists(sf_path):
        return None

    # Determine companion scale key names based on format conventions
    base_stem = name
    scale_key = None
    s2_key = None
    is_key = None

    if name.endswith('.weight'):
        base_stem = name[:-len('.weight')]
        scale_key = f'{base_stem}.weight_scale'
        s2_key = f'{base_stem}.weight_scale_2'
        is_key = f'{base_stem}.input_scale'
    elif name.endswith('_packed'):
        base_stem = name[:-len('_packed')]
        scale_key = f'{base_stem}_scale'
        s2_key = f'{base_stem}_global_scale'
    else:
        # Try common patterns
        # For MoE tensors like "model.layers.0.mlp.experts.gate_up_proj.weight"
        # the scales might be stored under "model.layers.0.mlp.experts.gate_up_proj.weight_scale"
        if '.weight' in name:
            base_stem = name[:name.rindex('.weight')]
            scale_key = f'{base_stem}.weight_scale'
            s2_key = f'{base_stem}.weight_scale_2'
            is_key = f'{base_stem}.input_scale'
        elif '_packed' in name:
            base_stem = name[:name.rindex('_packed')]
            scale_key = f'{base_stem}_scale'
            s2_key = f'{base_stem}_global_scale'

    if not scale_key:
        return None

    # For MoE expert tensors, also try expert-indexed scale naming
    # Some formats store scales flattened: [n_experts * K/16]
    is_moe_expert = is_moe and len(shape) >= 3 and shape[0] > 1

    try:
        with safetensors.safe_open(sf_path, framework="np") as f:
            all_keys = list(f.keys())

            if scale_key not in all_keys and is_moe_expert:
                # MoE variant: scales may be stored under a 2D key
                # Try with expert.suffix pattern
                alt_scale_key = None
                for k in all_keys:
                    if k.endswith('.weight_scale') or k.endswith('_scale'):
                        if base_stem in k or name.replace('.weight', '') in k:
                            alt_scale_key = k
                            if alt_scale_key:
                                break
                scale_key = alt_scale_key if alt_scale_key in all_keys else None

            if not scale_key or scale_key not in all_keys:
                return None

            scale_raw = f.get_tensor(scale_key)

            # Format-specific scale decode
            if fmt == 'compressed_tensors':
                scale_f32 = ue8m0_to_f32(scale_raw.view(np.uint8))
            elif fmt == 'mlx_nvfp4':
                if scale_raw.dtype == np.uint16:
                    scale_f32 = bf16_to_f32(scale_raw.tobytes()).reshape(scale_raw.shape)
                else:
                    scale_f32 = scale_raw.astype(np.float32)
                # MLX uses 32-element blocks; duplicate for 16-element OMMA compat
                if scale_f32.ndim >= 2 and scale_f32.shape[-1] * 32 == shape[-1] * 2:
                    scale_f32 = np.repeat(scale_f32, 2, axis=-1)
            else:  # modelopt or fallback
                scale_f32 = e4m3_to_f32(scale_raw)

            # Apply secondary scales
            if s2_key and s2_key in all_keys:
                s2 = f.get_tensor(s2_key)
                if s2.ndim == 0 or s2.size == 1:
                    scale_f32 *= float(s2)
                else:
                    scale_f32 *= s2.reshape(-1, 1)

            # NOTE: input_scale is the ACTIVATION scale (W4A8 calibration). For W4A16
            # weight-only inference it is NEVER read by the kernel, and folding it here
            # attenuates weights by ~0.0077 (~130x) -> near-zero logits -> garble.
            # Same rationale as the input_scale/input_global_scale folds removed from
            # build_nvfp4_den_v4.py (audit fixes FIX-4/FIX-5, 2026-08-01).
            # DELIBERATELY SKIPPED.
            # if is_key and is_key in all_keys:
            #     is_val = f.get_tensor(is_key)
            #     scale_f32 *= (float(is_val) if is_val.ndim == 0 or is_val.size == 1
            #                   else is_val.reshape(-1, 1))

    except Exception as e:
        print(f"    Error extracting scales for {name}: {e}")
        return None

    N = shape[0]
    K = shape[-1] if len(shape) >= 2 else shape[0]

    # Validate scale shape matches expectation
    # Expected: scales [N, K/16] for 2D weights
    # For MoE expert tensors [n_experts, ...], scales [n_experts, ..., K/16]
    expected_scale_count = K // GROUP_SIZE
    if scale_f32.ndim == 1:
        if scale_f32.shape[0] >= expected_scale_count:
            # 1D scales — reshape to [1, -1] (shared across rows)
            scale_f32 = scale_f32.reshape(1, -1)
    elif scale_f32.ndim == 2:
        if scale_f32.shape[0] == N and scale_f32.shape[1] >= expected_scale_count:
            pass  # Already [N, K/16]
        elif scale_f32.shape[0] == 1:
            pass  # [1, K/16] — broadcast
        elif scale_f32.shape[0] == N * expected_scale_count:
            # Flattened — reshape
            scale_f32 = scale_f32.reshape(N, expected_scale_count)

    # For MoE expert tensors, ensure scales match nibble shape
    if is_moe_expert and scale_f32.ndim >= 2:
        # Data is [n_experts, ...], nibbles is [N, K/2] (flat first 2 dims)
        # Scales should be [N, K/16] or [n_experts, ...]
        pass  # Already handled by N computation above

    return (data, scale_f32.astype(np.float32), N, K)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Convert LLM-Compressor / AutoRound NVFP4 exports to .den format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic single-file conversion
  python tools/convert_llmcompressor_to_den.py \\
      --safetensors model.safetensors \\
      --config config.json \\
      --output model.den

  # Sharded safetensors (point to any shard or the directory)
  python tools/convert_llmcompressor_to_den.py \\
      --safetensors /path/to/model/ \\
      --config /path/to/model/config.json \\
      --output model.den

  # With explicit quant config and BF16 reference for firewalled tensors
  python tools/convert_llmcompressor_to_den.py \\
      --safetensors model.safetensors \\
      --config config.json \\
      --output model.den \\
      --quant-config quant_config.json \\
      --bf16-ref /path/to/original/model

  # Force a specific NVFP4 format
  python tools/convert_llmcompressor_to_den.py \\
      --safetensors model.safetensors \\
      --config config.json \\
      --output model.den \\
      --format modelopt
        """)
    parser.add_argument('--safetensors', required=True,
                        help='Path to safetensors file, shard, or directory')
    parser.add_argument('--config', required=True,
                        help='Path to config.json')
    parser.add_argument('--output', '-o', required=True,
                        help='Output .den file path')
    parser.add_argument('--quant-config',
                        help='Path to quant_config.json (optional)')
    parser.add_argument('--bf16-ref',
                        help='Path to original HF model directory for firewalled BF16 weights')
    parser.add_argument('--format',
                        choices=['modelopt', 'compressed_tensors', 'mlx_nvfp4', 'auto'],
                        default='auto',
                        help='Force NVFP4 format (default: auto-detect)')
    args = parser.parse_args()

    force_format = None if args.format == 'auto' else args.format

    if not os.path.exists(args.safetensors):
        print(f"ERROR: safetensors path not found: {args.safetensors}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.config):
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Convert
    sys.exit(convert(
        args.safetensors, args.config, args.output,
        quant_config_path=args.quant_config,
        bf16_ref_path=args.bf16_ref,
        force_format=force_format,
    ))


if __name__ == '__main__':
    main()
