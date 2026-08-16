# Model Pipeline Plan — Ornith-35B: abliteration + native MTP

Goal: **uncensored + native-MTP (1.3×)** Ornith-35B, served on llama.cpp at 16GB.

## Inputs (on disk)

| File | Size | Role |
|---|---|---|
| `I:\models\Ornith-35B-F16.gguf` | 64.6G | F16 source — the training/abliteration base |
| `I:\models\Ornith-1.0-35B-Heretic-MTP-APEX-I-Mini.gguf` | 13.4G | current ship (already abliterated, Qwen3.5 MTP head) |

## Pipeline

```
F16 source ──► [1] abliterate ──► [2] train MTP head (EAGLE-3) ──► [3] graft ──► [4] quantize ──► [5] serve
```

---

### 1. Abliteration (refusal-direction projection)

Only needed for censored bases (Heretic already done; LordNeel/Dreadbyte/BTL-4 are not).

- Framework: PyTorch (not llama.cpp). Load F16, run refusal-direction projection, dump abliterated F16.
- Reference recipe: standard abliteration — collect "refuse" vs "comply" activations on a probe set, compute the refusal direction (mean-diff of residual streams at the last layers), subtract `direction * scale` from the relevant `mlp.down_proj` / residual weights.
- Scale sweep 0.5–2.0, test refusal on slurs/harm probes until neutral, verify benchmark scores don't collapse.
- Output: `Ornith-35B-F16-abliterated.gguf`.

### 2. MTP head training (EAGLE-3 style)

Train a **~0.8B draft head** on the abliterated trunk's hidden states. Frozen trunk, shared `embed_tokens` + `lm_head` (frozen), only the head layers train.

- Framework: **AngelSpec** (Tencent) — torch-native, one pipeline for MTP + EAGLE-3 + DSpark.
  `git clone https://github.com/Tencent/AngelSpec`
- Data: **on-policy** (the head must imitate THIS target, not generic text).
  1. Generate responses with the abliterated F16 itself (teacher-forced, ~1–5M tokens of chat/code/agent traces).
  2. Pre-extract hidden states from the F16 → train the head on those (no full verifier in VRAM → fits 16GB).
- Loss: position-decay weighted, `alpha_k = beta^(k-1)/Σ`, `beta = 0.6`.
- Output: `ornith-mtp-head.safetensors` (0.8B).

### 3. Graft

Two paths:

- **Own head** → graft onto the abliterated body via `gguf_mtp_graft.py` (base = body, donor = head), `block_count 40→41`, `nextn_predict_layers = 1`.
- **Dreadbyte head-v3** (`ornith-1.0-35b-MTP-head-v3-Q8_0.gguf`, 858MB) — pre-trained standalone head; graft onto a non-MTP abliterated body (Heretic already has a head, so use the abliterated body from step 1).
- **AEON-7 DFlash** (`AEON-DFlash-Qwen3.6-35B-A3B`, 948MB safetensors, 45.1% accept on Ornith) — needs `convert_hf_to_gguf.py` support for its 8L block-7 arch; serves via `--spec-type draft-dflash`.

### 4. Quantize

Re-quantize the grafted model to Q3_K/Q4_K to fit 16GB. GDN/SSM stays BF16 (hard rule).

### 5. Serve

```powershell
$env:DEN_GDN_FAST_EXP=1; $env:DEN_NVFP4_KV_CACHE=0
llama-server -m <abliterated-native-mtp>.gguf -ngl 99 -ctk kvarn6 -ctv kvarn6 `
  --spec-type draft-mtp -md <head>.gguf --reasoning-format deepseek --jinja
```

## Decision points (do NOT revisit)

1. **Built-in Ornith MTP = Qwen3.5 graft (~10%)** — not worth it. Use native/EAGLE-3/DFlash.
2. **Native MTP (LordNeel) = 1.3×** but censored + no standalone head → train-own (this plan) or Dreadbyte head.
3. **DFlash = +18%** (vanilla), Ornith-validated (45.1% accept), conversion-gap only — cheapest +18%.
4. **DSpark = regression on this trunk (0.91–0.96×)** — skip.
5. All shipped models must be abliterated ([[refusal-free-models-rule]]).

## Priority

1. **DFlash conversion** (cheapest +18%, engine-safe) — download AEON-7 drafter, extend converter, benchmark.
2. **Abliteration** of LordNeel's native-MTP (if we want the 1.3×) OR train-own EAGLE-3 head on the abliterated F16.
3. MTP training only if DFlash conversion + LordNeel abliteration both fail.
