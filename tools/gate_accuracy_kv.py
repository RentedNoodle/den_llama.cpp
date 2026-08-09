#!/usr/bin/env python3
"""
gate_accuracy_kv.py — In-process NVFP4 KV accuracy gate vs F32 oracle.

Loads ONE model, creates TWO contexts (F32 KV oracle + NVFP4 KV candidate),
runs SHARED decode loop, compares logit distributions at every position.

Single-process = deterministic thread pool + CUDA state.
Logit-level = measures distribution divergence, not token-match.
Immune to thinking-path stochasticity (reasoning traces differ but
logit distributions at each position are comparable).

Metrics and thresholds:
  median KLD        < 0.001   (BeeLlama q8_0 tier)
  99.9%ile KLD      < 0.1
  mean logit cosine  >= 0.9995
  min logit cosine   >= 0.99
  top-1 match rate   >= 0.95   (informational only)

Usage:
  python tools/gate_accuracy_kv.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/gate_accuracy_kv.py --model I:\\models\\ornith-35b-NVFP4.gguf --tokens 200 --ngl 0 --no-expert-stage

Exit: 0 = PASS (all hard gates green), 1 = FAIL.
"""

import sys
import os
import argparse
import ctypes
from ctypes import (
    c_int32, c_uint32, c_int8, c_bool, c_float, c_double,
    c_char, c_char_p, c_void_p, c_size_t, POINTER, Structure,
    CFUNCTYPE, byref, cast, pointer, sizeof,
    create_string_buffer, CDLL,
)
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BUILD_DIRS = [
    PROJECT_DIR / "build_ninja" / "bin",
    PROJECT_DIR / "build_bench" / "bin",
    PROJECT_DIR / "build" / "bin",
]


def find_llama_dll_dir() -> Path:
    """Find directory containing llama.dll and dependencies."""
    for d in BUILD_DIRS:
        candidate = d / "llama.dll"
        if candidate.is_file():
            return d
    raise FileNotFoundError(
        "llama.dll not found. Build llama.cpp with GGML_CUDA=ON first."
    )


LLAMA_DLL_DIR = find_llama_dll_dir()

# ─────────────────────────────────────────────────────────────────────────────
# GGML / LLAMA CONSTANTS (from ggml.h + llama.h)
# ─────────────────────────────────────────────────────────────────────────────

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30

LLAMA_SPLIT_MODE_NONE = 0
LLAMA_SPLIT_MODE_LAYER = 1
LLAMA_LOAD_MODE_MMAP = 0
LLAMA_LOAD_MODE_NO_MMAP = 1
LLAMA_CONTEXT_TYPE_DEFAULT = 0
LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
LLAMA_POOLING_TYPE_NONE = 0
LLAMA_ATTENTION_TYPE_CAUSAL = 0
LLAMA_FLASH_ATTN_TYPE_AUTO = -1

# llama_token = int32_t
llama_token = c_int32

# Opaque pointer types
class llama_model_p(ctypes.c_void_p):
    pass


class llama_context_p(ctypes.c_void_p):
    pass


class llama_vocab_p(ctypes.c_void_p):
    pass


# Function pointer types
llama_progress_callback_t = CFUNCTYPE(c_bool, c_float, c_void_p)

# ── Struct: llama_batch ────────────────────────────────────────────────────


class llama_batch(Structure):
    _fields_ = [
        ("n_tokens", c_int32),
        ("token", POINTER(llama_token)),
        ("embd", POINTER(c_float)),
        ("pos", POINTER(llama_token)),
        ("n_seq_id", POINTER(c_int32)),
        ("seq_id", POINTER(POINTER(llama_token))),
        ("logits", POINTER(c_int8)),
    ]


# ── Struct: llama_model_params ─────────────────────────────────────────────
# sizeof = 72 bytes (MSVC x64, natural alignment, 8-byte struct align)


class llama_model_params(Structure):
    _fields_ = [
        ("devices", c_void_p),  # offset 0
        ("tensor_buft_overrides", c_void_p),  # offset 8
        ("n_gpu_layers", c_int32),  # offset 16
        ("split_mode", c_int32),  # offset 20 (enum)
        ("load_mode", c_int32),  # offset 24 (enum)
        ("main_gpu", c_int32),  # offset 28
        # 4 bytes implicit padding to align pointer
        ("tensor_split", c_void_p),  # offset 32
        ("progress_callback", llama_progress_callback_t),  # offset 40
        ("progress_callback_user_data", c_void_p),  # offset 48
        ("kv_overrides", c_void_p),  # offset 56
        ("vocab_only", c_bool),  # offset 64
        ("check_tensors", c_bool),  # offset 65
        ("use_extra_bufts", c_bool),  # offset 66
        ("no_host", c_bool),  # offset 67
        ("no_alloc", c_bool),  # offset 68
        ("load_mtp", c_bool),  # offset 69
        # 2 bytes implicit tail padding -> total 72
    ]


# ── Struct: llama_kvarn_params ─────────────────────────────────────────────


class llama_kvarn_params(Structure):
    _fields_ = [
        ("type", c_int32),
        ("key_bits", c_int32),
        ("value_bits", c_int32),
        ("swa_key_bits", c_int32),
        ("swa_value_bits", c_int32),
        ("group", c_int32),
        ("sinkhorn_iters", c_int32),
        ("sink_tokens", c_int32),
        ("fail_if_unsupported", c_bool),
        # 3 bytes implicit tail padding — ctypes auto-aligns to 4 (sizeof = 36)
    ]


# ── Struct: llama_context_params ───────────────────────────────────────────
# Verified against llama.h (MSVC x64, natural alignment). sizeof = 208.
# kvarn sizeof = 36, ends at offset 148.
# After kvarn: 4B pad → abort_callback at 152, abort_callback_data at 160.
# bools at 168-177, 6B pad → samplers at 184, n_samplers at 192, ctx_other at 200.


class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_uint32),  # 0
        ("n_batch", c_uint32),  # 4
        ("n_ubatch", c_uint32),  # 8
        ("n_seq_max", c_uint32),  # 12
        ("n_rs_seq", c_uint32),  # 16
        ("n_outputs_max", c_uint32),  # 20
        ("n_threads", c_int32),  # 24
        ("n_threads_batch", c_int32),  # 28
        ("ctx_type", c_int32),  # 32
        ("rope_scaling_type", c_int32),  # 36
        ("pooling_type", c_int32),  # 40
        ("attention_type", c_int32),  # 44
        ("flash_attn_type", c_int32),  # 48
        ("rope_freq_base", c_float),  # 52
        ("rope_freq_scale", c_float),  # 56
        ("yarn_ext_factor", c_float),  # 60
        ("yarn_attn_factor", c_float),  # 64
        ("yarn_beta_fast", c_float),  # 68
        ("yarn_beta_slow", c_float),  # 72
        ("yarn_orig_ctx", c_uint32),  # 76
        ("defrag_thold", c_float),  # 80
        ("cb_eval", c_void_p),  # 88 (4B pad at 84)
        ("cb_eval_user_data", c_void_p),  # 96
        ("type_k", c_int32),  # 104
        ("type_v", c_int32),  # 108
        ("kvarn", llama_kvarn_params),  # 112 (sizeof=36, ends 148)
        ("abort_callback", c_void_p),  # 152 (4B pad after kvarn)
        ("abort_callback_data", c_void_p),  # 160
        ("embeddings", c_bool),  # 168
        ("offload_kqv", c_bool),  # 169
        ("no_perf", c_bool),  # 170
        ("op_offload", c_bool),  # 171
        ("swa_full", c_bool),  # 172
        ("kv_unified", c_bool),  # 173
        ("expert_stage", c_bool),  # 174
        ("expert_stage_probe", c_bool),  # 175
        ("nvfp4_kv_enabled", c_bool),  # 176
        ("sparse_kv_enabled", c_bool),  # 177
        # 6B pad to align pointer
        ("samplers", c_void_p),  # 184
        ("n_samplers", c_size_t),  # 192
        ("ctx_other", c_void_p),  # 200
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DLL LOADING
# ─────────────────────────────────────────────────────────────────────────────


def _load_llama_dll() -> CDLL:
    """Load llama.dll with dependency resolution from build directory."""
    dll_dir = str(LLAMA_DLL_DIR)

    # Add DLL directory to search path (Windows 8+)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)

    # Prepend to PATH so runtime dependencies are found
    if dll_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

    # Load dependent DLLs first
    for dep in ["ggml.dll", "ggml-cpu.dll", "ggml-base.dll"]:
        dep_path = os.path.join(dll_dir, dep)
        if os.path.isfile(dep_path):
            ctypes.CDLL(dep_path)

    # Load CUDA runtime
    cuda_path = os.path.join(dll_dir, "cudart64_13.dll")
    if os.path.isfile(cuda_path):
        ctypes.CDLL(cuda_path)

    # Load llama.dll
    lib_path = os.path.join(dll_dir, "llama.dll")
    lib = ctypes.CDLL(lib_path)

    # ── Set function signatures ─────────────────────────────────────────

    # Backend
    lib.llama_backend_init.argtypes = []
    lib.llama_backend_init.restype = None
    lib.llama_backend_free.argtypes = []
    lib.llama_backend_free.restype = None

    # Model
    lib.llama_model_default_params.argtypes = []
    lib.llama_model_default_params.restype = llama_model_params
    lib.llama_model_load_from_file.argtypes = [c_char_p, llama_model_params]
    lib.llama_model_load_from_file.restype = c_void_p

    # Context
    lib.llama_context_default_params.argtypes = []
    lib.llama_context_default_params.restype = llama_context_params
    lib.llama_init_from_model.argtypes = [c_void_p, llama_context_params]
    lib.llama_init_from_model.restype = c_void_p

    # Vocab
    lib.llama_model_get_vocab.argtypes = [c_void_p]
    lib.llama_model_get_vocab.restype = c_void_p
    lib.llama_vocab_n_tokens.argtypes = [c_void_p]
    lib.llama_vocab_n_tokens.restype = c_int32

    # Tokenize
    lib.llama_tokenize.argtypes = [
        c_void_p, c_char_p, c_int32, POINTER(llama_token), c_int32, c_bool, c_bool,
    ]
    lib.llama_tokenize.restype = c_int32

    # Batch
    lib.llama_batch_get_one.argtypes = [POINTER(llama_token), c_int32]
    lib.llama_batch_get_one.restype = llama_batch

    # Decode
    lib.llama_decode.argtypes = [c_void_p, llama_batch]
    lib.llama_decode.restype = c_int32

    # Logits
    lib.llama_get_logits_ith.argtypes = [c_void_p, c_int32]
    lib.llama_get_logits_ith.restype = POINTER(c_float)

    # Free
    lib.llama_model_free.argtypes = [c_void_p]
    lib.llama_model_free.restype = None
    lib.llama_free.argtypes = [c_void_p]
    lib.llama_free.restype = None

    return lib


# Module-level lib handle (initialized lazily)
_LIB = None


def _get_lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_llama_dll()
        _LIB.llama_backend_init()
    return _LIB


# ─────────────────────────────────────────────────────────────────────────────
# MATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# DUAL-CONTEXT KV MODEL
# ─────────────────────────────────────────────────────────────────────────────


class DualContextKVModel:
    """
    Load one model, create TWO contexts sharing the same model weights.

    ctx_f32  — F32 KV cache, nvfp4_kv_enabled=False (oracle)
    ctx_nvfp4 — F32 KV type, nvfp4_kv_enabled=True (candidate under test)

    Both contexts see identical tokens. Only the KV cache backing store
    differs. Any logit divergence is purely from NVFP4 KV quantization error.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, ngl: int = 99,
                 n_threads: int = 6, expert_stage: bool = True):
        lib = _get_lib()
        self.lib = lib
        self.model_path = str(model_path)

        # ── Load model ONCE ──────────────────────────────────────────────
        mparams = lib.llama_model_default_params()
        mparams.n_gpu_layers = ngl
        self.model = lib.llama_model_load_from_file(
            self.model_path.encode("utf-8"), mparams
        )
        if not self.model:
            raise RuntimeError(f"Failed to load model: {model_path}")

        self.vocab = lib.llama_model_get_vocab(self.model)
        self.n_vocab = lib.llama_vocab_n_tokens(self.vocab)

        # ── F32 oracle context ───────────────────────────────────────────
        # CRITICAL: Qwen35/Ornith models AUTO-ENABLE NVFP4 KV even with
        # nvfp4_kv_enabled=False. Must set DEN_NVFP4_KV_CACHE=0 to defeat
        # auto-enable for the oracle context. Also unset thrift attention.
        saved_kv = os.environ.get("DEN_NVFP4_KV_CACHE")
        saved_thrift = os.environ.get("DEN_THRIFT_ATTENTION")

        os.environ["DEN_NVFP4_KV_CACHE"] = "0"
        os.environ.pop("DEN_THRIFT_ATTENTION", None)

        cp = lib.llama_context_default_params()
        cp.n_ctx = n_ctx
        cp.n_threads = n_threads
        cp.n_threads_batch = n_threads
        cp.type_k = GGML_TYPE_F32
        cp.type_v = GGML_TYPE_F32
        cp.nvfp4_kv_enabled = False
        cp.sparse_kv_enabled = False
        cp.expert_stage = expert_stage
        self.ctx_f32 = lib.llama_init_from_model(self.model, cp)
        if not self.ctx_f32:
            raise RuntimeError("Failed to create F32 oracle context")

        # ── NVFP4 candidate context ──────────────────────────────────────
        # Enable NVFP4 KV path + K8V8 thrift attention.
        os.environ["DEN_NVFP4_KV_CACHE"] = "1"
        os.environ["DEN_THRIFT_ATTENTION"] = "1"

        cp2 = lib.llama_context_default_params()
        cp2.n_ctx = n_ctx
        cp2.n_threads = n_threads
        cp2.n_threads_batch = n_threads
        cp2.type_k = GGML_TYPE_F32
        cp2.type_v = GGML_TYPE_F32
        cp2.nvfp4_kv_enabled = True
        cp2.sparse_kv_enabled = False
        cp2.expert_stage = expert_stage
        self.ctx_nvfp4 = lib.llama_init_from_model(self.model, cp2)
        if not self.ctx_nvfp4:
            raise RuntimeError("Failed to create NVFP4 candidate context")

        self._tokens_decoded = 0

    # ── Tokenization ─────────────────────────────────────────────────────

    def tokenize(self, text: str, add_special: bool = True) -> list:
        """Tokenize text, return list of token IDs."""
        text_bytes = text.encode("utf-8")
        n_max = len(text_bytes) + 32
        tokens = (llama_token * n_max)()
        n = self.lib.llama_tokenize(
            self.vocab, text_bytes, len(text_bytes),
            tokens, n_max, add_special, True,
        )
        if n < 0:
            raise RuntimeError(f"Tokenization failed: n={n}")
        return list(tokens[:n])

    # ── Decode ───────────────────────────────────────────────────────────

    def decode_both(self, tokens: list) -> None:
        """Feed identical token batch to BOTH contexts."""
        n = len(tokens)
        token_array = (llama_token * n)(*tokens)
        batch = self.lib.llama_batch_get_one(token_array, n)
        r1 = self.lib.llama_decode(self.ctx_f32, batch)
        r2 = self.lib.llama_decode(self.ctx_nvfp4, batch)
        if r1 < 0:
            raise RuntimeError(f"F32 decode failed at token {self._tokens_decoded}: ret={r1}")
        if r2 < 0:
            raise RuntimeError(f"NVFP4 decode failed at token {self._tokens_decoded}: ret={r2}")
        self._tokens_decoded += n

    # ── Logits ───────────────────────────────────────────────────────────

    def get_logits_f32(self) -> np.ndarray:
        """Get last logits from F32 oracle context."""
        ptr = self.lib.llama_get_logits_ith(self.ctx_f32, -1)
        if not ptr:
            raise RuntimeError("llama_get_logits_ith(ctx_f32, -1) returned NULL")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    def get_logits_nvfp4(self) -> np.ndarray:
        """Get last logits from NVFP4 candidate context."""
        ptr = self.lib.llama_get_logits_ith(self.ctx_nvfp4, -1)
        if not ptr:
            raise RuntimeError("llama_get_logits_ith(ctx_nvfp4, -1) returned NULL")
        return np.ctypeslib.as_array(ptr, shape=(self.n_vocab,)).copy()

    # ── Cleanup ──────────────────────────────────────────────────────────

    def close(self):
        if hasattr(self, "ctx_f32") and self.ctx_f32:
            self.lib.llama_free(self.ctx_f32)
            self.ctx_f32 = None
        if hasattr(self, "ctx_nvfp4") and self.ctx_nvfp4:
            self.lib.llama_free(self.ctx_nvfp4)
            self.ctx_nvfp4 = None
        if hasattr(self, "model") and self.model:
            self.lib.llama_model_free(self.model)
            self.model = None

    def __del__(self):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PROMPTS (mixed: short context anchors + varied tokens)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = (
    "The capital of France is Paris, a city known for its"
)

LONGER_PROMPT = (
    "The transformer architecture, introduced in the paper "
    "'Attention Is All You Need', revolutionized natural language processing. "
    "Its key innovation is the self-attention mechanism, which allows each token "
    "to attend to all other tokens in the sequence. This enables the model to "
    "capture long-range dependencies without the sequential bottleneck of "
    "recurrent neural networks. The standard transformer consists of an encoder "
    "and a decoder, each composed of multiple identical layers. Each layer "
    "contains multi-head self-attention followed by a position-wise feed-forward "
    "network, with residual connections and layer normalization."
)

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY_KV GATE
# ─────────────────────────────────────────────────────────────────────────────


# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def test_accuracy_kv(
    model_path: str,
    n_tokens: int = 200,
    ngl: int = 99,
    n_threads: int = 6,
    expert_stage: bool = True,
    prompt: str = None,
    n_prompt_tokens: int = 0,
) -> dict:
    """
    In-process NVFP4 KV accuracy gate.

    Loads model once, creates F32 oracle + NVFP4 candidate contexts,
    runs shared greedy decode loop, compares logits at every position.

    Returns dict with all metrics, thresholds, and pass/fail status.
    """
    if prompt is None:
        prompt = LONGER_PROMPT

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  ACCURACY_KV GATE — NVFP4 KV Cache vs F32 Oracle{RESET}")
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"  Model        : {Path(model_path).name}")
    print(f"  Tokens       : {n_tokens}")
    print(f"  GPU layers   : {ngl}")
    print(f"  Threads      : {n_threads}")
    print(f"  Expert stage : {expert_stage}")
    print(f"  Prompt       : {prompt[:80]}...")
    print(f"{'='*70}")

    m = None
    try:
        # Estimate context size: prompt + generation + headroom
        ctx_size = max(512, n_tokens + 512)
        if n_prompt_tokens > 0:
            ctx_size = max(ctx_size, n_prompt_tokens + n_tokens + 128)

        print(f"\n  {CYAN}Loading model + creating dual contexts...{RESET}")
        m = DualContextKVModel(
            model_path,
            n_ctx=ctx_size,
            ngl=ngl,
            n_threads=n_threads,
            expert_stage=expert_stage,
        )
        print(f"  Vocab size: {m.n_vocab}")

        # Tokenize and decode prompt on both contexts
        print(f"  {CYAN}Decoding prompt ({len(prompt)} chars)...{RESET}")
        prompt_tokens = m.tokenize(prompt, add_special=True)
        if n_prompt_tokens > 0 and len(prompt_tokens) > n_prompt_tokens:
            prompt_tokens = prompt_tokens[:n_prompt_tokens]
        print(f"  Prompt tokens: {len(prompt_tokens)}")

        m.decode_both(prompt_tokens)

        # Get initial logits after prompt
        lf32 = m.get_logits_f32()
        lnv = m.get_logits_nvfp4()

        # Accumulators
        klds = []
        cosines = []
        top1_matches = 0
        total_positions = 0

        # Track worst-case position for diagnostics
        worst_step = -1
        worst_kld = -1.0

        print(f"\n  {CYAN}Running shared decode loop ({n_tokens} tokens)...{RESET}")
        print(f"  {'Step':>6}  {'KLD':>10}  {'Cosine':>10}  {'Top-1':>6}  {'Token'}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*10}")

        for step in range(n_tokens):
            # Compute metrics at current position
            pf32 = softmax(lf32.astype(np.float64))  # double precision for KLD stability
            pnv = softmax(lnv.astype(np.float64))

            eps = 1e-12
            # KL(P_f32 || P_nvfp4) — how much extra surprise using NVFP4 approx
            kld = float(np.sum(pf32 * np.log((pf32 + eps) / (pnv + eps))))
            cos = cos_sim(lf32, lnv)
            top1_match = int(np.argmax(lf32)) == int(np.argmax(lnv))

            klds.append(kld)
            cosines.append(cos)
            if top1_match:
                top1_matches += 1
            total_positions += 1

            if kld > worst_kld:
                worst_kld = kld
                worst_step = step

            # Greedy next token from F32 oracle
            next_token = int(np.argmax(lf32))

            # Periodic progress
            if step % 20 == 0 or step == n_tokens - 1:
                marker = " *" if not top1_match else ""
                print(f"  {step:>6}  {kld:>10.6f}  {cos:>10.6f}  {'Y' if top1_match else 'N':>6}{marker}  {next_token}")

            # Decode same token on both contexts
            m.decode_both([next_token])

            # Get next logits
            lf32 = m.get_logits_f32()
            lnv = m.get_logits_nvfp4()

        # ── Compute summary metrics ──────────────────────────────────────
        klds = np.array(klds)
        cosines = np.array(cosines)

        median_kld = float(np.median(klds))
        p999_kld = float(np.percentile(klds, 99.9)) if len(klds) >= 1000 else float(np.max(klds))
        p99_kld = float(np.percentile(klds, 99.0))
        p95_kld = float(np.percentile(klds, 95.0))
        mean_kld = float(np.mean(klds))
        max_kld = float(np.max(klds))

        mean_cos = float(np.mean(cosines))
        min_cos = float(np.min(cosines))
        median_cos = float(np.median(cosines))

        top1_rate = top1_matches / total_positions if total_positions > 0 else 0.0

        # ── Threshold checks ─────────────────────────────────────────────
        passed_kld_median = median_kld < 0.001
        passed_kld_tail = p999_kld < 0.1
        passed_cos_mean = mean_cos >= 0.9995
        passed_cos_min = min_cos >= 0.99
        passed_top1 = top1_rate >= 0.95  # informational

        hard_metrics = [passed_kld_median, passed_kld_tail, passed_cos_mean, passed_cos_min]
        all_hard_passed = all(hard_metrics)

        # ── Build result dict ────────────────────────────────────────────
        result = {
            "gate": "ACCURACY_KV",
            "passed": all_hard_passed,
            "metrics": {
                "median_kld": {
                    "value": median_kld, "passed": passed_kld_median,
                    "threshold": "< 0.001", "display": f"{median_kld:.6f}",
                },
                "p99.9_kld": {
                    "value": p999_kld, "passed": passed_kld_tail,
                    "threshold": "< 0.1", "display": f"{p999_kld:.6f}",
                },
                "mean_cos": {
                    "value": mean_cos, "passed": passed_cos_mean,
                    "threshold": ">= 0.9995", "display": f"{mean_cos:.6f}",
                },
                "min_cos": {
                    "value": min_cos, "passed": passed_cos_min,
                    "threshold": ">= 0.99", "display": f"{min_cos:.6f}",
                },
                "top1_rate": {
                    "value": top1_rate, "passed": passed_top1,
                    "threshold": ">= 0.95 (info)", "display": f"{top1_rate:.4f}",
                },
            },
            "extra": {
                "mean_kld": mean_kld,
                "max_kld": max_kld,
                "p99_kld": p99_kld,
                "p95_kld": p95_kld,
                "median_cos": median_cos,
                "worst_step": worst_step,
                "worst_kld": worst_kld,
            },
            "n_prompt_tokens": len(prompt_tokens),
            "n_tokens": n_tokens,
            "n_positions": total_positions,
        }

    except Exception as e:
        import traceback
        result = {
            "gate": "ACCURACY_KV",
            "passed": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    finally:
        if m is not None:
            try:
                m.close()
            except Exception:
                pass

    _print_kv_result(result)
    return result


def _print_kv_result(result: dict):
    """Print formatted KV accuracy results table."""
    if result.get("error"):
        print(f"\n  [{RED}FAIL{RESET}] ERROR: {result['error']}")
        if result.get("traceback"):
            print(f"\n  {RED}Traceback:{RESET}")
            for line in result["traceback"].splitlines()[-8:]:
                print(f"    {line}")
        return

    m = result["metrics"]
    extra = result.get("extra", {})

    # ── Results table ───────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  RESULTS{RESET}")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} {'Value':<14} {'Threshold':<18} {'Status'}")
    print(f"  {'-'*20} {'-'*14} {'-'*18} {'-'*8}")

    for name in ["median_kld", "p99.9_kld", "mean_cos", "min_cos", "top1_rate"]:
        d = m[name]
        status = f"{GREEN}PASS{RESET}" if d["passed"] else f"{RED}FAIL{RESET}"
        val_str = d.get("display", f"{d['value']:.6f}")
        print(f"  {name:<20} {val_str:<14} {d['threshold']:<18} [{status}]")

    print(f"  {'-'*20} {'-'*14} {'-'*18} {'-'*8}")

    # Additional diagnostics
    print(f"\n  {CYAN}Diagnostics:{RESET}")
    print(f"    Positions evaluated : {result['n_positions']}")
    print(f"    Prompt tokens       : {result['n_prompt_tokens']}")
    print(f"    Mean KLD            : {extra.get('mean_kld', 'N/A'):.6f}")
    print(f"    Max KLD             : {extra.get('max_kld', 'N/A'):.6f}")
    print(f"    P99 KLD             : {extra.get('p99_kld', 'N/A'):.6f}")
    print(f"    P95 KLD             : {extra.get('p95_kld', 'N/A'):.6f}")
    print(f"    Median cosine       : {extra.get('median_cos', 'N/A'):.6f}")
    print(f"    Worst KLD at step   : {extra.get('worst_step', 'N/A')} ({extra.get('worst_kld', 'N/A'):.6f})")

    # ── Overall ──────────────────────────────────────────────────────────
    overall = f"{GREEN}PASS{RESET}" if result["passed"] else f"{RED}FAIL{RESET}"
    print(f"\n  {BOLD}Overall: [{overall}]{RESET}")

    if not result["passed"]:
        # Identify which metrics failed
        failed = [name for name, d in m.items() if not d["passed"]]
        print(f"  {RED}Failed metrics: {', '.join(failed)}{RESET}")
    else:
        print(f"  {GREEN}ALL HARD GATES PASSED — NVFP4 KV meets accuracy thresholds{RESET}")

    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def _find_model() -> str:
    """Auto-discover a suitable model for testing."""
    candidates = [
        r"I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
        r"I:\models\AEON-7_Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8\AEON-7_Gemma-4-12B-it-AEON-Abliterated-K4-NVFP4-FP8.gguf",
        r"I:\models\ornith-1.0-9b-NVFP4.gguf",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="gate_accuracy_kv.py — In-process NVFP4 KV accuracy gate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/gate_accuracy_kv.py --model I:\\models\\ornith-35b-NVFP4.gguf
  python tools/gate_accuracy_kv.py --model I:\\models\\ornith-35b-NVFP4.gguf --tokens 500 --ngl 0
  python tools/gate_accuracy_kv.py --model I:\\models\\ornith-9b-NVFP4.gguf --tokens 200 --prompt "Once upon a time"
        """,
    )
    parser.add_argument("--model", default=None,
                        help="Path to model (NVFP4 or any quant). Auto-discovered if omitted.")
    parser.add_argument("--tokens", type=int, default=200,
                        help="Number of tokens to generate for comparison (default: 200)")
    parser.add_argument("--ngl", type=int, default=99,
                        help="GPU layers to offload. Use 0 for CPU-only (default: 99)")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads (default: 6)")
    parser.add_argument("--no-expert-stage", action="store_true",
                        help="Disable expert_stage context flag (for dense models)")
    parser.add_argument("--prompt", default=None,
                        help="Custom prompt text (default: transformer architecture paragraph)")
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="Truncate prompt to N tokens (0 = use all)")

    args = parser.parse_args()

    # Resolve model
    model_path = args.model
    if not model_path:
        model_path = _find_model()
        if not model_path:
            print(f"{RED}ERROR: No model specified and auto-discovery failed.{RESET}")
            print("  Specify --model or place a model at one of:")
            for c in [
                r"I:\models\ornith-1.0-35b-APEX-I-Mini-MTP.gguf",
                r"I:\models\ornith-1.0-9b-NVFP4.gguf",
            ]:
                print(f"    {c}")
            sys.exit(1)
        print(f"{YELLOW}Auto-discovered model: {model_path}{RESET}")

    if not os.path.isfile(model_path):
        print(f"{RED}ERROR: Model not found: {model_path}{RESET}")
        sys.exit(1)

    expert_stage = not args.no_expert_stage

    # Run gate
    result = test_accuracy_kv(
        model_path=model_path,
        n_tokens=args.tokens,
        ngl=args.ngl,
        n_threads=args.threads,
        expert_stage=expert_stage,
        prompt=args.prompt,
        n_prompt_tokens=args.prompt_tokens,
    )

    # Exit code
    sys.exit(0 if result.get("passed", False) else 1)


if __name__ == "__main__":
    main()
