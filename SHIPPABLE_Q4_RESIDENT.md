# Shippable #1 — 196k q8_0/q4_0 + adaptive-KV streaming (merged 2972b7ef8)
Verified 2026-08-29 on RTX 5070 Ti 16GB:
- Config: GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 llama-server -m YMQ-XS-Pro.gguf -c 196000 -fa on -ctk q8_0 -ctv q4_0 -ngl all -b 256 -ub 256 -np 1 --kv-stream-stage-mib 2048
- Result: 196k decode 38.1 t/s, prompt 16.9 t/s, tool+agentic 3/3 PASS, health 200
- Note: q4_0/q4_0 rejected (requires K Q8_0 V Q4_0)
