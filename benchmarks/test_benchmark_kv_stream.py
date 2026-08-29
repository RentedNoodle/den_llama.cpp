#!/usr/bin/env python3
"""Unit tests for the automatic adaptive KV benchmark driver."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).with_name("benchmark_kv_stream.py")
SPEC = importlib.util.spec_from_file_location("benchmark_kv_stream", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkKvStreamTest(unittest.TestCase):
    def test_parse_token_count(self) -> None:
        self.assertEqual(BENCHMARK.parse_token_count("192K"), 192 * 1024)
        self.assertEqual(BENCHMARK.parse_token_count("262144"), 262144)
        with self.assertRaises(argparse.ArgumentTypeError):
            BENCHMARK.parse_token_count("bad")

    def test_context_capacities_include_non_aligned_maximum(self) -> None:
        self.assertEqual(
            BENCHMARK.context_capacities(20000),
            [8192, 16384, 20000],
        )

    def test_pool_estimate_uses_all_free_memory_and_rounds_down(self) -> None:
        self.assertEqual(BENCHMARK.estimate_pool_mib(64, 3500, 32), 3552)
        self.assertEqual(
            BENCHMARK.estimate_pool_mib(64, 3500, 32, max_pool_mib=2048),
            2048,
        )

    def test_clean_server_env_removes_memory_policy_overrides(self) -> None:
        inherited = {
            "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1",
            "GGML_CUDA_PREFER_MODEL_WEIGHTS": "1",
            "GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS": "8",
            "KEEP_ME": "yes",
        }
        with mock.patch.dict(os.environ, inherited, clear=True):
            env = BENCHMARK.clean_server_env("2")
        self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", env)
        self.assertNotIn("GGML_CUDA_PREFER_MODEL_WEIGHTS", env)
        self.assertNotIn("GGML_CUDA_KV_STREAM_FIXED_RING_SLOTS", env)
        self.assertEqual(env["KEEP_ME"], "yes")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")

    def test_server_command_uses_tested_configuration(self) -> None:
        args = argparse.Namespace(
            server=Path("/tmp/llama-server"),
            model=Path("/tmp/model.gguf"),
            port=12355,
            extra_server_arg=["--verbosity", "3"],
        )
        command = BENCHMARK.server_command(args, 131072, 2304)
        self.assertEqual(command[0], "/tmp/llama-server")
        self.assertIn("131072", command)
        self.assertIn("2304", command)
        self.assertEqual(command[command.index("-ctk") + 1], "q8_0")
        self.assertEqual(command[command.index("-ctv") + 1], "q4_0")
        self.assertEqual(command[command.index("-np") + 1], "1")
        self.assertEqual(command[-2:], ["--verbosity", "3"])

    def test_resume_rejects_changed_settings(self) -> None:
        signature = {"model": "/tmp/model.gguf", "max_context": 16384}
        BENCHMARK.validate_resume(signature.copy(), signature, Path("results.jsonl"))
        with self.assertRaisesRegex(SystemExit, "different settings"):
            BENCHMARK.validate_resume(
                {"model": "/tmp/other.gguf", "max_context": 16384},
                signature,
                Path("results.jsonl"),
            )

    def test_csv_and_plot_accept_partial_sweep(self) -> None:
        rows = {
            8192: {
                "context_capacity": 8192,
                "prompt_tokens": 7936,
                "decode_tokens": 256,
                "pool_mib": 3552,
                "prefill_tps": 1400.0,
                "decode_tps": 50.0,
            },
            16384: {
                "context_capacity": 16384,
                "prompt_tokens": 16128,
                "decode_tokens": 256,
                "pool_mib": 3520,
                "prefill_tps": 1300.0,
                "decode_tps": 45.0,
            },
        }
        try:
            plt = BENCHMARK.require_matplotlib()
        except SystemExit:
            self.skipTest("Matplotlib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            BENCHMARK.write_csv(output / "results.csv", rows)
            BENCHMARK.plot_results(output, rows, plt)
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "kv-stream-sweep.png").is_file())
            self.assertTrue((output / "kv-stream-sweep.svg").is_file())


if __name__ == "__main__":
    unittest.main()
