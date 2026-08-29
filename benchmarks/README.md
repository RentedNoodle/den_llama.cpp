# Adaptive KV streaming benchmark

`benchmark_kv_stream.py` performs the complete context-matched benchmark and
creates its graph. The only required inputs are the GGUF model and the largest
context capacity to test:

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K
```

The default server is `build/bin/llama-server`. Use `--server` when the
binary is elsewhere. Matplotlib is the only Python dependency:

```bash
python3 -m pip install matplotlib
```

## What the script does

For every configured context capacity from 8K through `--max-context`, in 8K
steps, the script:

1. Starts a fresh adaptive KV streaming server with a small probe pool.
2. Measures free VRAM after model initialization and a warm-up request.
3. Assigns all measured free VRAM to the pool, rounded down to 32 MiB.
4. Runs a full prompt and 256-token decode to validate that pool.
5. If the candidate fails, reduces it by 64 MiB and retries.
6. Records prefill speed, decode speed, selected pool size, and VRAM telemetry.
7. Updates the CSV and Matplotlib graph after every successful point.

There is no fixed VRAM safety reserve. Actual server execution is the
validation: allocation failures are handled by automatic pool backoff.

The prompt length at each point is the configured context capacity minus the
256 decode tokens. For example, the 192K point starts the server with
`--ctx-size 196608`, prefills 196352 tokens, and then decodes 256 tokens.
If the maximum is not a multiple of 8K, the exact maximum is appended as the
last point.

The driver uses the configuration currently supported and validated by this
branch:

- Flash Attention enabled
- K cache `q8_0`
- V cache `q4_0`
- all model layers on the GPU
- one server slot
- 256-token batch and micro-batch
- ordinary CUDA allocation, without UVM

## Results and resuming

By default, a timestamped directory is created under
`benchmarks/results/adaptive-kv-sweep-*`. It contains:

- `results.jsonl`: metadata, pool probes, retries, and measurements
- `results.csv`: one successful measurement per context capacity
- `kv-stream-sweep.png` and `kv-stream-sweep.svg`: decode, prefill, and pool
  size plots
- `logs/`: one server log per probe and benchmark attempt

Use an explicit output directory to resume an interrupted sweep:

```bash
python3 benchmarks/benchmark_kv_stream.py \
  --model /path/to/model.gguf \
  --max-context 192K \
  --output-dir benchmarks/results/my-sweep
```

Re-run the same command after an interruption. Completed contexts are skipped.
The script rejects a resume if the model or benchmark settings differ, avoiding
mixed data in one result set.

Run `python3 benchmarks/benchmark_kv_stream.py --help` for optional GPU,
timeout, pool-step, output, and server arguments.

Do not run another GPU workload during the sweep. Its allocations would change
the automatically selected pool and invalidate comparisons between points.
