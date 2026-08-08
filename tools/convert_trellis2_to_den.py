#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_trellis2_to_den.py -- Microsoft TRELLIS.2-4B -> .den converter

Converts TRELLIS.2-4B safetensors checkpoints into the .den format for the
Den inference engine. Produces three .den files:
  - name-shape.den    (shape generation flow-matching transformer)
  - name-texture.den  (texture generation flow-matching transformer)
  - name-vision.den   (DINOv3 ViT-L vision encoder)

TRELLIS.2-4B architecture:
  - 4B flow-matching diffusion transformer (DiT)
  - DINOv3 ViT-L vision encoder
  - O-Voxel sparse 3D VAE (16x spatial compression)
  - Two-stage: shape generation + texture generation
  - Resolution: 512^3, 1024^3, or 1536^3 voxel grids

Usage:
  python tools/convert_trellis2_to_den.py \
      --shape trellis_2_shape_bf16.safetensors \
      --texture trellis_2_texture_bf16.safetensors \
      --vision dino_v3_vit_l.safetensors \
      --output trellis2 --quantize wh4

PORTED from C:\Den\den-nvfp4-optimizations\tools\convert_trellis2_to_den.py
-> I:\den_llama.cpp\tools\convert_trellis2_to_den.py
"""

import argparse
import json
import math
import os
import re
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import safetensors
except ImportError:
    safetensors = None
    print("WARNING: safetensors not installed. Install with: pip install safetensors")


# ═══════════════════════════════════════════════════════════════════════════
# .den format constants (from denrt_format.h / den_format.h)
# ═══════════════════════════════════════════════════════════════════════════

DEN_MAGIC = 0x4E454400
DEN_VERSION = 0x00050000
DEN_HEADER_SIZE = 4096
DEN_TENSOR_ENTRY_SIZE = 128

# Architecture IDs
DEN_ARCH_DIT = 8           # single-stream DiT
DEN_ARCH_CUSTOM = 255

# Hardware targets
DEN_TARGET_NVFP4 = 1
DEN_TARGET_BF16 = 2
DEN_TARGET_F32 = 3

# Flags
DEN_FLAG_DIT_FLOW_MATCHING = 1 << 13

# Layer stride for DiT
DEN_LAYER_STRIDE_DIT = 64

# Global slots
DEN_SLOT_TOKEN_EMBD = 0
DEN_SLOT_OUTPUT_NORM = 1
DEN_SLOT_OUTPUT = 2

# DiT per-layer sub-slots (within a layer's 64-slot block)
DEN_SLOT_DIT_TIME_EMBED = 32
DEN_SLOT_DIT_MODULATION = 33
DEN_SLOT_DIT_SELF_Q = 34
DEN_SLOT_DIT_SELF_K = 35
DEN_SLOT_DIT_SELF_V = 36
DEN_SLOT_DIT_SELF_O = 37
DEN_SLOT_DIT_XATTN_Q = 38
DEN_SLOT_DIT_XATTN_K = 39
DEN_SLOT_DIT_XATTN_V = 40
DEN_SLOT_DIT_XATTN_O = 41
DEN_SLOT_DIT_FFN_GATE = 42
DEN_SLOT_DIT_FFN_UP = 43
DEN_SLOT_DIT_FFN_DOWN = 44
DEN_SLOT_DIT_ADALN_SILU = 45
DEN_SLOT_DIT_LN1 = 46
DEN_SLOT_DIT_LN2 = 47
DEN_SLOT_DIT_LN3 = 48
DEN_SLOT_DIT_FINAL_NORM = 49
DEN_SLOT_DIT_PROJ_IN = 50
DEN_SLOT_DIT_PROJ_OUT = 51

# Tile metadata offsets (within 160B tile)
OMMA_TILE_BYTES = 160
OMMA_TILE_ELEMS = 256
OMMA_BLOCK_SIZE = 16
OMMA_TILE_SCALES = 16
OMMA_TILE_NORMOFF = 144
OMMA_TILE_META_OFF = 148

# Dispatch codes for tile[148]
DEN_PATH_OMMA_GEMV = 0x10    # OMMA 4X path
DEN_PATH_WHT_OMMA = 0x08     # WHT-domain quantization

# UE4M3 scale table (must match CUDA kernel tables)
UE4M3 = np.array([0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125,
                  0.375, 0.4375, 1.0, 1.125, 1.25, 1.375,
                  1.5, 1.625, 1.75, 1.875], dtype=np.float32)

# E2M1 representable magnitudes
E2M1_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

# -- Which sub-slots to WH4 quantize (FFN + self-attn + cross-attn projections) --
QUANT_SUB_SLOTS = {
    DEN_SLOT_DIT_SELF_Q, DEN_SLOT_DIT_SELF_K, DEN_SLOT_DIT_SELF_V, DEN_SLOT_DIT_SELF_O,
    DEN_SLOT_DIT_XATTN_Q, DEN_SLOT_DIT_XATTN_K, DEN_SLOT_DIT_XATTN_V, DEN_SLOT_DIT_XATTN_O,
    DEN_SLOT_DIT_FFN_GATE, DEN_SLOT_DIT_FFN_UP, DEN_SLOT_DIT_FFN_DOWN,
}

# Firewalled tensors (keep BF16 -- norms, biases, small projections)
FIREWALL_PATTERNS = [
    r".*norm.*", r".*bias.*", r".*pos_embed.*", r".*cls_token.*",
    r".*time_embed.*", r".*modulation.*",
]

FIREWALL_SUB_SLOTS = {
    DEN_SLOT_DIT_TIME_EMBED, DEN_SLOT_DIT_MODULATION,
    DEN_SLOT_DIT_LN1, DEN_SLOT_DIT_LN2, DEN_SLOT_DIT_LN3,
    DEN_SLOT_DIT_FINAL_NORM, DEN_SLOT_DIT_PROJ_IN, DEN_SLOT_DIT_PROJ_OUT,
    DEN_SLOT_DIT_ADALN_SILU,
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def dit_layer_base(layer: int) -> int:
    """Compute the base slot for a DiT layer."""
    return 3 + layer * DEN_LAYER_STRIDE_DIT


def is_firewalled(name: str) -> bool:
    return any(re.match(p, name) for p in FIREWALL_PATTERNS)


# ═══════════════════════════════════════════════════════════════════════════
# Tensor name -> .den slot mapper for TRELLIS.2 DiT
# ═══════════════════════════════════════════════════════════════════════════

def map_dit_tensor(name: str, n_layers: int) -> int:
    """Map a TRELLIS.2 transformer tensor name to a .den slot ID.

    Expected naming patterns:
      transformer.layers.{N}.self_attn.q_proj.weight
      transformer.layers.{N}.cross_attn.q_proj.weight
      transformer.layers.{N}.ffn.gate_proj.weight
      transformer.layers.{N}.norm1.weight
      transformer.layers.{N}.norm2.weight
      transformer.layers.{N}.norm3.weight
      time_embed.mlp.0.weight       (global)
      time_embed.mlp.2.weight       (global)
      adaln_modulation.1.weight     (global)
      proj_in.weight                (global input patch embed)
      proj_out.weight               (global output projection)
      final_norm.weight              (global final norm)
      pos_embed                     (global positional embedding)
      cls_token                     (global class token)
    """
    # Strip common prefix variants
    n = name
    if n.startswith("model."):
        n = n[len("model."):]
    if n.startswith("transformer."):
        n = n[len("transformer."):]

    # -- Global (non-layer) tensors --
    if n == "proj_in.weight":
        return DEN_SLOT_DIT_PROJ_IN
    if n == "proj_out.weight":
        return DEN_SLOT_DIT_PROJ_OUT
    if n == "final_norm.weight" or n == "final_norm.bias":
        return DEN_SLOT_DIT_FINAL_NORM

    # Time embedding
    if n.startswith("time_embed."):
        return DEN_SLOT_DIT_TIME_EMBED

    # AdaLN modulation
    if n.startswith("adaln_modulation."):
        return DEN_SLOT_DIT_MODULATION

    # Positional embedding / class token
    if n == "pos_embed" or n.startswith("pos_embed."):
        return DEN_SLOT_TOKEN_EMBD
    if n == "cls_token" or n.startswith("cls_token."):
        return DEN_SLOT_TOKEN_EMBD + 1

    # -- Per-layer tensors --
    # Pattern: layers.{N}.{module}.{proj}.weight
    m = re.match(r"layers\.(\d+)\.(.+)", n)
    if not m:
        m = re.match(r"transformer_block\.(\d+)\.(.+)", n)
    if not m:
        m = re.match(r"blocks\.(\d+)\.(.+)", n)
    if not m:
        raise ValueError(f"Unrecognized tensor name (no layer pattern): {name}")

    layer = int(m.group(1))
    suffix = m.group(2)
    base = dit_layer_base(layer)

    if layer >= n_layers:
        raise ValueError(f"Layer {layer} >= n_layers {n_layers}: {name}")

    # -- Self-attention --
    if suffix.startswith("self_attn.") or suffix.startswith("self_attn_"):
        return _map_attention(suffix, base, DEN_SLOT_DIT_SELF_Q)

    # -- Cross-attention --
    if suffix.startswith("cross_attn.") or suffix.startswith("cross_attn_"):
        return _map_attention(suffix, base, DEN_SLOT_DIT_XATTN_Q)

    # -- FFN --
    if suffix.startswith("ffn.") or suffix.startswith("ffn_"):
        return _map_ffn(suffix, base)

    # -- Layer norms --
    if suffix == "norm1.weight" or suffix == "norm1.bias":
        return base + DEN_SLOT_DIT_LN1
    if suffix == "norm2.weight" or suffix == "norm2.bias":
        return base + DEN_SLOT_DIT_LN2
    if suffix == "norm3.weight" or suffix == "norm3.bias":
        return base + DEN_SLOT_DIT_LN3

    raise ValueError(f"Unmapped per-layer tensor: {name} (suffix={suffix})")


def _map_attention(suffix: str, base: int, q_slot: int) -> int:
    """Map attention projection to sub-slot."""
    if suffix.endswith("q_proj.weight") or suffix.endswith(".q_proj.weight"):
        return base + q_slot
    if suffix.endswith("k_proj.weight") or suffix.endswith(".k_proj.weight"):
        return base + q_slot + 1  # K = Q+1
    if suffix.endswith("v_proj.weight") or suffix.endswith(".v_proj.weight"):
        return base + q_slot + 2  # V = Q+2
    if suffix.endswith("o_proj.weight") or suffix.endswith(".o_proj.weight"):
        return base + q_slot + 3  # O = Q+3
    if suffix.endswith("qkv.weight") or suffix.endswith(".qkv.weight"):
        return base + q_slot  # fused QKV -> Q slot (K,V inferred)
    raise ValueError(f"Unmapped attention tensor: {suffix}")


def _map_ffn(suffix: str, base: int) -> int:
    """Map FFN projection to sub-slot."""
    if (suffix.endswith("gate_proj.weight") or suffix.endswith(".gate_proj.weight")
            or suffix.endswith("gate.weight")):
        return base + DEN_SLOT_DIT_FFN_GATE
    if suffix.endswith("up_proj.weight") or suffix.endswith(".up_proj.weight"):
        return base + DEN_SLOT_DIT_FFN_UP
    if suffix.endswith("down_proj.weight") or suffix.endswith(".down_proj.weight"):
        return base + DEN_SLOT_DIT_FFN_DOWN
    raise ValueError(f"Unmapped FFN tensor: {suffix}")


# ═══════════════════════════════════════════════════════════════════════════
# DINOv3 ViT-L tensor mapper
# ═══════════════════════════════════════════════════════════════════════════

def map_vision_tensor(name: str, n_layers: int) -> int:
    """Map a DINOv3 ViT-L tensor name to a .den slot for a vision encoder."""
    n = name

    # Global tensors
    if n == "cls_token" or n == "class_token":
        return 0
    if n == "pos_embed" or n == "positional_embedding":
        return 1
    if n.startswith("patch_embed.") or n.startswith("patch_embed_"):
        if "weight" in n:
            return 2
        return 3
    if n == "norm.weight" or n == "norm.bias" or n == "final_norm.weight" or n == "final_norm.bias":
        return 4

    # Per-block tensors
    m = re.match(r"blocks?\.(\d+)\.(.+)", n)
    if not m:
        m = re.match(r"vit\.blocks?\.(\d+)\.(.+)", n)
    if not m:
        m = re.match(r"transformer_block\.(\d+)\.(.+)", n)
    if not m:
        raise ValueError(f"Unrecognized vision tensor: {name}")

    layer = int(m.group(1))
    suffix = m.group(2)
    base = 5 + layer * 32  # vision blocks start at slot 5, stride 32

    if layer >= n_layers:
        raise ValueError(f"Vision layer {layer} >= {n_layers}: {name}")

    # Attention block
    if "attn" in suffix or "attention" in suffix:
        if "qkv" in suffix:
            return base + 0
        if "q_proj" in suffix or "q_proj" in suffix:
            return base + 0
        if "k_proj" in suffix or "k_proj" in suffix:
            return base + 1
        if "v_proj" in suffix or "v_proj" in suffix:
            return base + 2
        if "proj" in suffix and ("out" in suffix or "o_proj" in suffix or "output" in suffix):
            return base + 3
        return base + 0  # default: fused attention

    # Norms
    if suffix == "norm1.weight" or suffix == "norm1.bias":
        return base + 4
    if suffix == "norm2.weight" or suffix == "norm2.bias":
        return base + 5
    if suffix == "norm.weight" or suffix == "norm.bias":
        return base + 4  # pre-attention norm

    # MLP / FFN
    if "mlp" in suffix or "ffn" in suffix:
        if "fc1" in suffix or "gate" in suffix:
            return base + 8
        if "fc2" in suffix or "down" in suffix:
            return base + 10
        if "up" in suffix:
            return base + 9
        return base + 8  # default first MLP layer

    raise ValueError(f"Unmapped vision tensor: {name} (suffix={suffix})")


# ═══════════════════════════════════════════════════════════════════════════
# WH4 WHT-domain quantization
# ═══════════════════════════════════════════════════════════════════════════

def batched_wht(mat: np.ndarray) -> None:
    """In-place fast Walsh-Hadamard Transform on every row of mat [N, K2]."""
    N, K2 = mat.shape
    step = 1
    while step < K2:
        for i in range(0, K2, step << 1):
            a = mat[:, i:i + step]
            b = mat[:, i + step:i + (step << 1)]
            mat[:, i:i + step] = a + b
            mat[:, i + step:i + (step << 1)] = a - b
        step <<= 1
    mat *= (1.0 / math.sqrt(K2))


def quantize_wh4(weights: np.ndarray, K: int, row_batch: int = 0) -> bytes:
    """WH4-quantize a BF16 weight matrix to 160B tiles with WHT-domain encoding.

    Args:
        weights: [total_rows, K] float32 array.
        K: Original column dimension (may differ from weights.shape[1] if padded).
        row_batch: Batch size for processing (0 = auto).

    Returns:
        Packed tile bytes: [total_rows * tpr * 160] bytes.
    """
    total_rows, K_orig = weights.shape
    if K_orig < 256:
        return None  # too small to quantize

    # Pad K to power of 2
    K2 = 1
    while K2 < K_orig:
        K2 <<= 1

    # Tile layout
    tpr = K2 // OMMA_TILE_ELEMS
    if K2 % OMMA_TILE_ELEMS:
        tpr += 1
    n_blocks = (K2 + OMMA_BLOCK_SIZE - 1) // OMMA_BLOCK_SIZE
    tiles_per_row = (n_blocks + OMMA_TILE_SCALES - 1) // OMMA_TILE_SCALES

    # Auto batch size
    if row_batch <= 0:
        row_batch = min(total_rows, 16384)

    total_tiles = total_rows * tiles_per_row
    tile_data = bytearray(total_tiles * OMMA_TILE_BYTES)

    for batch_start in range(0, total_rows, row_batch):
        batch_end = min(batch_start + row_batch, total_rows)
        batch_rows = batch_end - batch_start
        batch = np.ascontiguousarray(weights[batch_start:batch_end].astype(np.float32))

        # Pad K -> K2
        if K2 > K_orig:
            pad = np.zeros((batch_rows, K2 - K_orig), dtype=np.float32)
            batch = np.hstack([batch, pad])

        # Batched WHT
        batched_wht(batch)

        # Reshape to blocks
        block_padded = np.zeros((batch_rows, n_blocks * OMMA_BLOCK_SIZE), dtype=np.float32)
        block_padded[:, :K2] = batch
        blocks = block_padded.reshape(batch_rows, n_blocks, OMMA_BLOCK_SIZE)
        del batch

        # O(1) scale selection from max_abs per block
        max_abs = np.max(np.abs(blocks), axis=2)
        scale_codes = np.zeros((batch_rows, n_blocks), dtype=np.uint8)
        for code in range(1, 16):
            covered = (max_abs / 6.0 <= UE4M3[code]) & (scale_codes == 0)
            scale_codes[covered] = code

        scales = np.zeros((batch_rows, n_blocks), dtype=np.float32)
        for code in range(1, 16):
            scales[scale_codes == code] = UE4M3[code]
        del max_abs

        # Quantize to E2M1 nibbles
        effective_scales = scales.reshape(batch_rows, n_blocks, 1)
        qv = blocks / np.where(effective_scales > 1e-10, effective_scales, 1.0)
        qv = np.clip(qv, -6.0, 6.0)

        abs_qv = np.abs(qv)
        nib_mags = np.zeros((batch_rows, n_blocks, OMMA_BLOCK_SIZE), dtype=np.uint8)
        for mag_idx in range(8):
            dist = np.abs(abs_qv - E2M1_MAGS[mag_idx])
            if mag_idx == 0:
                best_dist = dist.copy()
            else:
                mask = dist < best_dist
                best_dist[mask] = dist[mask]
                nib_mags[mask] = mag_idx
        del abs_qv, best_dist

        sign_nib = np.where(qv < 0, 8, 0).astype(np.uint8)
        nibbles = nib_mags | sign_nib
        del sign_nib, qv

        # Pack nibbles (two per byte)
        even_nibs = nibbles[:, :, 0::2]
        odd_nibs = nibbles[:, :, 1::2]
        packed = even_nibs | (odd_nibs << 4)
        del even_nibs, odd_nibs

        # Tile norm correction
        deq_vals = np.zeros_like(blocks)
        for mag_idx in range(8):
            mask = nib_mags == mag_idx
            sign_mask = nibbles[:, :, :] // 8 == 1
            sign_arr = np.where(sign_mask, -1.0, 1.0)
            deq_vals += np.where(mask, sign_arr * E2M1_MAGS[mag_idx] * effective_scales, 0.0)
        del mask, sign_mask, sign_arr

        orig_sq = np.sum(blocks ** 2, axis=2)
        deq_sq_flat = deq_vals.reshape(batch_rows, n_blocks, OMMA_BLOCK_SIZE)
        deq_sq = np.sum(deq_sq_flat ** 2, axis=2)
        tn = np.ones((batch_rows, n_blocks), dtype=np.float32)
        nonzero = deq_sq > 1e-20
        tn[nonzero] = np.sqrt(orig_sq[nonzero] / deq_sq[nonzero])
        tn = np.clip(tn, 0.25, 4.0)
        tn *= 15.0 / 16.0
        del deq_vals, orig_sq, deq_sq

        # Tile assembly
        for r in range(batch_rows):
            for t in range(tiles_per_row):
                blk_start = t * OMMA_TILE_SCALES
                blk_end = min(blk_start + OMMA_TILE_SCALES, n_blocks)
                n_active = blk_end - blk_start
                abs_row = batch_start + r
                tile_off = (abs_row * tiles_per_row + t) * OMMA_TILE_BYTES

                # Scale bytes (UE4M3 codes)
                for b in range(n_active):
                    tile_data[tile_off + b] = int(scale_codes[r, blk_start + b])

                # E2M1 nibbles
                for b in range(n_active):
                    byte_start = b * (OMMA_BLOCK_SIZE // 2)
                    for b8 in range(OMMA_BLOCK_SIZE // 2):
                        tile_data[tile_off + 16 + byte_start + b8] = int(
                            packed[r, blk_start + b, b8])

                # Tile norm
                blk_tn = np.mean(tn[r, blk_start:blk_end]) if n_active > 0 else 1.0
                struct.pack_into('<f', tile_data, tile_off + OMMA_TILE_NORMOFF,
                                 float(blk_tn))

                # Tile metadata
                tile_data[tile_off + OMMA_TILE_META_OFF] = DEN_PATH_OMMA_GEMV | DEN_PATH_WHT_OMMA
                tile_data[tile_off + OMMA_TILE_META_OFF + 1] = 8  # WHT-domain
                tile_data[tile_off + OMMA_TILE_META_OFF + 2] = 0
                tile_data[tile_off + OMMA_TILE_META_OFF + 3] = 0

        del blocks, scale_codes, scales, effective_scales
        del nib_mags, nibbles, packed, tn

    return bytes(tile_data)


# ═══════════════════════════════════════════════════════════════════════════
# .den writer
# ═══════════════════════════════════════════════════════════════════════════

class DenWriter:
    """Writes a .den file with streaming tensor data."""

    def __init__(self, output_path: str, arch: int = DEN_ARCH_DIT,
                 flags: int = DEN_FLAG_DIT_FLOW_MATCHING,
                 n_layers: int = 28, hidden_size: int = 4096,
                 ffn_size: int = 16384, n_heads: int = 32,
                 n_kv_heads: int = 32, vocab_size: int = 0,
                 max_seq_len: int = 16384, rope_theta: float = 10000.0,
                 rms_norm_eps: float = 1e-6):
        self.path = output_path
        self.arch = arch
        self.flags = flags
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.tensor_entries: List[dict] = []
        self.tensor_data: Dict[int, bytes] = {}

    def add_tensor(self, slot: int, data: np.ndarray,
                   hw_target: int = DEN_TARGET_BF16, flags: int = 0,
                   tile_k: int = 0, tile_n: int = 0, n_tiles: int = 0,
                   scale_count: int = 0, scale_data: Optional[bytes] = None) -> None:
        """Register a tensor for writing."""
        ndim = data.ndim
        dims = list(data.shape[:4]) + [1] * (4 - min(ndim, 4))
        numel = int(np.prod(data.shape))
        data_bytes = data.tobytes()
        data_size = len(data_bytes)
        scale_sz = len(scale_data) if scale_data else 0

        self.tensor_entries.append({
            'slot': slot,
            'hw_target': hw_target,
            'ndim': ndim,
            'flags': flags,
            'dims': dims,
            'numel': numel,
            'data_size': data_size,
            'data': data_bytes,
            'scale_data': scale_data or b'',
            'scale_size': scale_sz,
            'tile_k': tile_k,
            'tile_n': tile_n,
            'n_tiles': n_tiles,
            'scale_count': scale_count,
        })

    def write(self) -> str:
        """Write the .den file. Returns the output path."""
        entries = self.tensor_entries
        n_tot = len(entries)

        # Compute data offset (header + index, aligned to 4096)
        index_sz = n_tot * DEN_TENSOR_ENTRY_SIZE
        data_off = align_up(DEN_HEADER_SIZE + index_sz, 4096)

        # Calculate data offsets for each tensor (relative to data section)
        cur = 0
        for e in entries:
            e['data_offset'] = cur
            cur += e['data_size']
            if e['scale_size'] > 0:
                e['scale_offset'] = cur
                cur += e['scale_size']
            else:
                e['scale_offset'] = 0

        total_data_sz = cur
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)

        with open(self.path, 'wb') as f:
            # -- Write header (4096 bytes) --
            hdr = bytearray(DEN_HEADER_SIZE)
            struct.pack_into('<4I', hdr, 0,
                             DEN_MAGIC, DEN_VERSION, self.arch, self.flags)
            struct.pack_into('<10I2f', hdr, 16,
                             self.n_layers, self.n_heads, self.n_kv_heads,
                             self.hidden_size, self.ffn_size,
                             self.vocab_size, self.max_seq_len,
                             0,  # n_rot
                             0,  # n_experts
                             0,  # n_experts_used
                             self.rope_theta, self.rms_norm_eps)
            # SSM fields (not used for DiT -- zero)
            struct.pack_into('<6I', hdr, 64, 0, 0, 0, 0, 0, 0)
            struct.pack_into('<II', hdr, 88, 0, 0)   # mtp_layer_count, ssm_value_size
            struct.pack_into('<II', hdr, 96, 0, 0)   # _padding[2]
            # Data layout
            struct.pack_into('<II', hdr, 104, n_tot, DEN_HEADER_SIZE)
            struct.pack_into('<Q', hdr, 112, data_off)
            struct.pack_into('<Q', hdr, 120, total_data_sz)
            # Tier counts/sizes (unused)
            struct.pack_into('<III', hdr, 128, 0, 0, 0)
            struct.pack_into('<QQQ', hdr, 144, 0, 0, 0)
            f.write(hdr)

            # -- Write tensor index --
            idx = bytearray(index_sz)
            for i, e in enumerate(entries):
                off = i * DEN_TENSOR_ENTRY_SIZE
                struct.pack_into('<IIII', idx, off,
                                 e['slot'], e['hw_target'], e['ndim'], e['flags'])
                struct.pack_into('<4q', idx, off + 16,
                                 e['dims'][0], e['dims'][1], e['dims'][2], e['dims'][3])
                struct.pack_into('<Q', idx, off + 48, e['numel'])
                struct.pack_into('<Q', idx, off + 56, e['data_offset'])
                struct.pack_into('<Q', idx, off + 64, e['data_size'])
                struct.pack_into('<Q', idx, off + 72, e['scale_offset'])
                struct.pack_into('<Q', idx, off + 80, e['scale_size'])
                struct.pack_into('<I', idx, off + 88, e['tile_k'])
                struct.pack_into('<I', idx, off + 92, e['tile_n'])
                struct.pack_into('<I', idx, off + 96, e['n_tiles'])
                struct.pack_into('<I', idx, off + 100, e['scale_count'])
                struct.pack_into('<Q', idx, off + 104, 0)  # norm_offset
                struct.pack_into('<I', idx, off + 112, 0)  # norm_size
                struct.pack_into('<I', idx, off + 116, 0)  # block_size
                struct.pack_into('<I', idx, off + 120, 0)  # grid_size
                struct.pack_into('<I', idx, off + 124, 0)  # smem_bytes
            f.write(idx)

            # Pad to data offset
            pad = data_off - f.tell()
            if pad > 0:
                f.write(b'\x00' * pad)

            # -- Write tensor data --
            for e in entries:
                f.write(e['data'])
                if e['scale_size'] > 0:
                    f.write(e['scale_data'])

        file_size = os.path.getsize(self.path)
        print(f"  Wrote {self.path} ({file_size / 1024 / 1024:.1f} MB, {n_tot} tensors)")
        return self.path


# ═══════════════════════════════════════════════════════════════════════════
# Safetensors loader
# ═══════════════════════════════════════════════════════════════════════════

def load_safetensors_metadata(path: str) -> dict:
    """Read safetensors metadata (tensor names, shapes, dtypes) without loading data."""
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    return hdr


def get_tensor_from_safetensors(path: str, name: str) -> np.ndarray:
    """Load a single tensor from a safetensors file as numpy array."""
    if safetensors is None:
        raise ImportError("safetensors package required")
    with safetensors.safe_open(path, framework='np') as f:
        return f.get_tensor(name)


def detect_n_layers(tensor_names: list, prefix_patterns: list = None) -> int:
    """Detect the number of transformer layers from tensor names."""
    if prefix_patterns is None:
        prefix_patterns = [
            r"transformer\.layers\.(\d+)\.",
            r"transformer_block\.(\d+)\.",
            r"blocks\.(\d+)\.",
        ]
    max_layer = -1
    for name in tensor_names:
        for pat in prefix_patterns:
            m = re.match(pat, name)
            if m:
                layer = int(m.group(1))
                if layer > max_layer:
                    max_layer = layer
    return max_layer + 1 if max_layer >= 0 else 0


def detect_hidden_size(tensor_names: list, path: str) -> int:
    """Detect hidden_size from the first self-attention Q projection."""
    for name in tensor_names:
        if "self_attn.q_proj" in name and "weight" in name:
            t = get_tensor_from_safetensors(path, name)
            return t.shape[1]  # [out_features, in_features]
    return 4096  # default fallback


def detect_ffn_size(tensor_names: list, path: str) -> int:
    """Detect FFN intermediate size."""
    for name in tensor_names:
        if "ffn.gate_proj" in name and "weight" in name:
            t = get_tensor_from_safetensors(path, name)
            return t.shape[0]  # [out_features, in_features]
        if "ffn.up_proj" in name and "weight" in name:
            t = get_tensor_from_safetensors(path, name)
            return t.shape[0]
    return 16384  # default fallback


def detect_n_heads(tensor_names: list, path: str) -> int:
    """Detect number of attention heads from Q projection shapes."""
    for name in tensor_names:
        if "self_attn.q_proj" in name and "weight" in name:
            t = get_tensor_from_safetensors(path, name)
            out_dim = t.shape[0]
            # Common head dimensions: 64, 128
            for hd in [128, 64, 96, 112, 256]:
                if out_dim % hd == 0:
                    return out_dim // hd
            return out_dim // 128  # guess
    return 32


# ═══════════════════════════════════════════════════════════════════════════
# Main conversion logic
# ═══════════════════════════════════════════════════════════════════════════

def convert_dit_model(safetensors_path: str, output_path: str,
                      quantize: str = "none", model_type: str = "shape") -> str:
    """Convert a TRELLIS.2 safetensors file to .den format.

    Args:
        safetensors_path: Path to the safetensors file.
        output_path: Output .den file path.
        quantize: Quantization mode ("none", "wh4").
        model_type: Model type ("shape", "texture", "vision").

    Returns:
        Path to the written .den file.
    """
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Input not found: {safetensors_path}")

    print(f"\n{'=' * 60}")
    print(f"Converting {model_type} model: {safetensors_path}")
    print(f"{'=' * 60}")

    # -- Pass 1: Read metadata --
    hdr = load_safetensors_metadata(safetensors_path)
    tensor_names = [k for k in hdr if k != '__metadata__']
    print(f"  Found {len(tensor_names)} tensors")

    # Detect architecture
    n_layers = detect_n_layers(tensor_names)
    hidden_size = detect_hidden_size(tensor_names, safetensors_path)
    ffn_size = detect_ffn_size(tensor_names, safetensors_path)
    n_heads = detect_n_heads(tensor_names, safetensors_path)

    print(f"  Architecture: {n_layers} layers, H={hidden_size}, FFN={ffn_size}, heads={n_heads}")

    # -- Determine slot mapping --
    mapper = map_dit_tensor
    slot_map = {}  # name -> slot

    for name in tensor_names:
        try:
            slot = mapper(name, n_layers)
            slot_map[name] = slot
        except (ValueError, IndexError) as e:
            print(f"  WARNING: Skipping unmapped tensor: {name} ({e})")
            continue

    if not slot_map:
        raise RuntimeError("No tensors could be mapped. Check tensor name patterns.")

    print(f"  Mapped {len(slot_map)}/{len(tensor_names)} tensors to slots")

    # -- Group tensors by slot (handle collisions) --
    slot_tensors: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    for name, slot in slot_map.items():
        if slot not in slot_tensors:
            slot_tensors[slot] = []
        t = get_tensor_from_safetensors(safetensors_path, name)
        if 'bfloat16' in str(t.dtype):
            t = t.astype(np.float32)
        slot_tensors[slot].append((name, t))

    # -- Group and concatenate per-slot --
    slot_data: Dict[int, Tuple[np.ndarray, List[str]]] = {}
    for slot, tensors in slot_tensors.items():
        if len(tensors) == 1:
            slot_data[slot] = (tensors[0][1], [tensors[0][0]])
        else:
            arrays = [t[1] for t in tensors]
            names = [t[0] for t in tensors]
            if all(a.ndim == 1 for a in arrays):
                combined = np.concatenate(arrays)
            elif all(a.ndim == 2 for a in arrays):
                combined = np.vstack(arrays)
            else:
                combined = max(arrays, key=lambda a: a.size)
            slot_data[slot] = (combined, names)

    # -- Determine which slots need firewall (keep BF16) --
    def slot_needs_firewall(slot: int) -> bool:
        for layer in range(n_layers):
            base = dit_layer_base(layer)
            sub = slot - base
            if 0 <= sub < DEN_LAYER_STRIDE_DIT:
                if sub in FIREWALL_SUB_SLOTS:
                    return True
                if slot in slot_data:
                    names = slot_data[slot][1]
                    if any(is_firewalled(n) for n in names):
                        return True
                return False
        return False

    def slot_is_quantizable(slot: int) -> bool:
        """Check if a slot should be WH4 quantized."""
        if slot <= 2:
            return False  # global slots stay BF16
        for layer in range(n_layers):
            base = dit_layer_base(layer)
            sub = slot - base
            if 0 <= sub < DEN_LAYER_STRIDE_DIT:
                return sub in QUANT_SUB_SLOTS
        return False

    # -- Write .den file --
    writer = DenWriter(
        output_path=output_path,
        arch=DEN_ARCH_DIT,
        flags=DEN_FLAG_DIT_FLOW_MATCHING,
        n_layers=n_layers,
        hidden_size=hidden_size,
        ffn_size=ffn_size,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=0,
        max_seq_len=16384,
    )

    qtype = quantize.lower()
    do_quant = (qtype == "wh4")
    quant_count = 0
    bf16_count = 0

    # Sort slots for deterministic output
    for slot in sorted(slot_data.keys()):
        data, names = slot_data[slot]

        # Skip non-2D tensors for quantization
        if data.ndim < 2 or data.shape[1] < 256:
            do_this_quant = False
        else:
            do_this_quant = do_quant and slot_is_quantizable(slot) and not slot_needs_firewall(slot)

        if do_this_quant:
            # WH4 quantize
            tile_bytes = quantize_wh4(data, data.shape[1])
            if tile_bytes is not None:
                total_rows, K = data.shape
                K2 = 1
                while K2 < K:
                    K2 <<= 1
                tpr = K2 // OMMA_TILE_ELEMS
                if K2 % OMMA_TILE_ELEMS:
                    tpr += 1
                n_tiles = total_rows * tpr

                writer.add_tensor(
                    slot=slot,
                    data=np.frombuffer(tile_bytes, dtype=np.uint8),
                    hw_target=DEN_TARGET_NVFP4,
                    flags=1 << 7,  # DEN_TFLAG_HADAMARD
                    tile_k=OMMA_TILE_ELEMS,
                    tile_n=tpr,
                    n_tiles=n_tiles,
                    scale_count=total_rows * OMMA_TILE_SCALES,
                )
                quant_count += 1
                print(f"  Slot {slot:3d} ({names[0][:50]}): WH4 quantized "
                      f"({total_rows}x{K} -> {len(tile_bytes):,} bytes)")
            else:
                writer.add_tensor(slot=slot, data=data, hw_target=DEN_TARGET_BF16)
                bf16_count += 1
                print(f"  Slot {slot:3d} ({names[0][:50]}): BF16 (too small for WH4)")
        else:
            # BF16 passthrough
            if data.dtype == np.float32:
                u32 = data.view(np.uint32)
                u16 = (u32 >> 16).astype(np.uint16)
                bf16_data = u16.reshape(data.shape)
                writer.add_tensor(slot=slot, data=bf16_data, hw_target=DEN_TARGET_BF16)
            else:
                writer.add_tensor(slot=slot, data=data, hw_target=DEN_TARGET_BF16)
            bf16_count += 1
            print(f"  Slot {slot:3d} ({names[0][:50]}): BF16 "
                  f"({data.shape}, {data.nbytes:,} bytes)")

    result_path = writer.write()
    print(f"  Summary: {quant_count} WH4 quantized + {bf16_count} BF16 tensors")
    return result_path


def convert_vision_model(safetensors_path: str, output_path: str) -> str:
    """Convert DINOv3 ViT-L to .den format."""
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Input not found: {safetensors_path}")

    print(f"\n{'=' * 60}")
    print(f"Converting vision encoder: {safetensors_path}")
    print(f"{'=' * 60}")

    hdr = load_safetensors_metadata(safetensors_path)
    tensor_names = [k for k in hdr if k != '__metadata__']
    print(f"  Found {len(tensor_names)} tensors")

    # Detect number of ViT blocks
    n_layers = detect_n_layers(tensor_names, [
        r"blocks\.(\d+)\.",
        r"vit\.blocks\.(\d+)\.",
        r"transformer_block\.(\d+)\.",
    ])
    print(f"  Vision layers: {n_layers}")

    mapper = map_vision_tensor
    slot_map = {}
    for name in tensor_names:
        try:
            slot = mapper(name, n_layers if n_layers > 0 else 24)
            slot_map[name] = slot
        except (ValueError, IndexError) as e:
            print(f"  WARNING: Skipping unmapped vision tensor: {name} ({e})")
            continue

    if not slot_map:
        raise RuntimeError("No vision tensors could be mapped.")

    print(f"  Mapped {len(slot_map)}/{len(tensor_names)} tensors")

    # Load and group by slot
    slot_tensors: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    for name, slot in slot_map.items():
        if slot not in slot_tensors:
            slot_tensors[slot] = []
        t = get_tensor_from_safetensors(safetensors_path, name)
        if 'bfloat16' in str(t.dtype):
            t = t.astype(np.float32)
        slot_tensors[slot].append((name, t))

    # Write with DEN_ARCH_CUSTOM
    writer = DenWriter(
        output_path=output_path,
        arch=DEN_ARCH_CUSTOM,
        flags=0,
        n_layers=n_layers if n_layers > 0 else 24,
        hidden_size=1024,  # ViT-L hidden
        ffn_size=4096,
        n_heads=16,  # ViT-L heads
    )

    bf16_count = 0
    for slot in sorted(slot_tensors.keys()):
        tensors = slot_tensors[slot]
        if len(tensors) == 1:
            data = tensors[0][1]
            name = tensors[0][0]
        else:
            arrays = [t[1] for t in tensors]
            if all(a.ndim == 1 for a in arrays):
                data = np.concatenate(arrays)
            elif all(a.ndim == 2 for a in arrays):
                data = np.vstack(arrays)
            else:
                data = max(arrays, key=lambda a: a.size)
            name = tensors[0][0]

        if data.dtype == np.float32:
            u32 = data.view(np.uint32)
            u16 = (u32 >> 16).astype(np.uint16)
            bf16_data = u16.reshape(data.shape)
            writer.add_tensor(slot=slot, data=bf16_data, hw_target=DEN_TARGET_BF16)
        else:
            writer.add_tensor(slot=slot, data=data, hw_target=DEN_TARGET_BF16)
        bf16_count += 1
        print(f"  Slot {slot:3d} ({name[:50]}): BF16 ({data.shape}, {data.nbytes:,} bytes)")

    result_path = writer.write()
    print(f"  Summary: {bf16_count} BF16 tensors")
    return result_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Convert Microsoft TRELLIS.2-4B to .den format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--shape", help="Path to trellis_2_shape_bf16.safetensors")
    ap.add_argument("--texture", help="Path to trellis_2_texture_bf16.safetensors")
    ap.add_argument("--vision", help="Path to dino_v3_vit_l.safetensors")
    ap.add_argument("--output", default="trellis2", help="Output basename (default: trellis2)")
    ap.add_argument("--quantize", choices=["none", "wh4"], default="none",
                    help="Quantization mode for weight matrices (default: none)")
    ap.add_argument("--skip-shape", action="store_true", help="Skip shape model")
    ap.add_argument("--skip-texture", action="store_true", help="Skip texture model")
    ap.add_argument("--skip-vision", action="store_true", help="Skip vision encoder")
    args = ap.parse_args()

    if not args.shape and not args.texture and not args.vision:
        ap.error("At least one of --shape, --texture, --vision must be provided")

    if safetensors is None:
        print("FATAL: safetensors package not installed. Install with: pip install safetensors")
        sys.exit(1)

    start = time.time()
    results = []

    # -- Convert shape model --
    if args.shape and not args.skip_shape:
        out = os.path.join(os.path.dirname(args.output) or '.',
                           os.path.basename(args.output) + "-shape.den")
        results.append(convert_dit_model(args.shape, out, args.quantize, "shape"))

    # -- Convert texture model --
    if args.texture and not args.skip_texture:
        out = os.path.join(os.path.dirname(args.output) or '.',
                           os.path.basename(args.output) + "-texture.den")
        results.append(convert_dit_model(args.texture, out, args.quantize, "texture"))

    # -- Convert vision encoder --
    if args.vision and not args.skip_vision:
        out = os.path.join(os.path.dirname(args.output) or '.',
                           os.path.basename(args.output) + "-vision.den")
        results.append(convert_vision_model(args.vision, out))

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Conversion complete in {elapsed:.1f}s")
    for r in results:
        sz = os.path.getsize(r)
        print(f"  {r} ({sz / 1024 / 1024:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
