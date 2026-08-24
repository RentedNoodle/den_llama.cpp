from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import torch

from conversion.base import ModelBase, gguf
from conversion.qwen import Qwen3_5TextModel


class _CaptureWriter:
    def __init__(self):
        self.tensors = []

    def add_tensor(self, name, data, **kwargs):
        self.tensors.append((name, np.asarray(data).copy(), kwargs))


def _model_for_escha_tests():
    weights = {
        "model.embed_tokens.weight_int8": torch.tensor([[1, -2], [3, -4]], dtype=torch.int8),
        "model.embed_tokens.weight_scale": torch.tensor([[0.5], [1.5]], dtype=torch.float16),
        "lm_head.weight_int8": torch.tensor([[5, -6], [7, -8]], dtype=torch.int8),
        "lm_head.weight_scale": torch.tensor([[2.5], [3.5]], dtype=torch.float16),
    }

    model = Qwen3_5TextModel.__new__(Qwen3_5TextModel)
    model.hparams = {
        "quantization_config": {"quant_method": "escha"},
        "num_hidden_layers": 0,
    }
    model.model_tensors = {name: (lambda tensor=tensor: tensor) for name, tensor in weights.items()}
    model.ftype = gguf.LlamaFileType.MOSTLY_F16
    model.dir_model = Path("__missing_escha_test_model__")
    model._is_nvfp4 = False
    model._is_mxfp4 = False
    model._fp8_as_q8 = False
    model._fp8_dequantized = set()
    model.fuse_gate_up_exps = False
    model._gate_exp_buffer = {}
    model._up_exp_buffer = {}
    model.tensor_map = type("TensorMap", (), {"mapping": {}})()
    model.map_tensor_name = lambda name, try_suffixes=(".weight", ".bias"): {
        "model.embed_tokens.weight": "token_embd.weight",
        "lm_head.weight": "output.weight",
    }[name]
    model.match_model_tensor_name = lambda name, key, bid, suffix=".weight": False
    model.gguf_writer = _CaptureWriter()
    return model, weights


def _write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        raw = tensor.numpy().tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": {
                torch.int8: "I8",
                torch.float16: "F16",
            }[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + payload)


def _escha_group(base, include_bias=True):
    tensors = {
        f"{base}.escha_code": torch.tensor([1, 2], dtype=torch.int16),
        f"{base}.escha_config": torch.tensor([3, 4], dtype=torch.int32),
        f"{base}.escha_rin": torch.tensor([5], dtype=torch.float16),
        f"{base}.escha_rout": torch.tensor([6], dtype=torch.float16),
        f"{base}.escha_s_in": torch.tensor([7], dtype=torch.float32),
        f"{base}.escha_s_out": torch.tensor([8], dtype=torch.float32),
    }
    if include_bias:
        tensors[f"{base}.bias"] = torch.tensor([9], dtype=torch.float16)
    return tensors


def _model_for_escha_group_test(tensors):
    model, _ = _model_for_escha_tests()
    base_names = {
        name.split(".escha_", 1)[0] if ".escha_" in name else name.removesuffix(".bias")
        for name in tensors
    }
    model.tensor_map = type("TensorMap", (), {"mapping": {}})()
    model.map_tensor_name = lambda name, try_suffixes=(".weight", ".bias"): {
        f"{base}.weight": f"mapped.{i}.weight"
        for i, base in enumerate(sorted(base_names))
    }[name]
    model.model_tensors = {name: (lambda tensor=tensor: tensor) for name, tensor in tensors.items()}
    model.set_escha_lut(torch.tensor([1], dtype=torch.float16))
    return model


class EschaEndpointConversionTests(unittest.TestCase):
    def test_lazy_safetensors_endpoint_conversion_maps_int8_to_i8(self):
        model, weights = _model_for_escha_tests()

        with tempfile.TemporaryDirectory() as tmp:
            model.dir_model = Path(tmp)
            model.lazy = True
            model.gguf_writer = gguf.GGUFWriter(
                path=None,
                arch=gguf.MODEL_ARCH_NAMES[model.model_arch],
                dry_run=True,
            )
            _write_safetensors(model.dir_model / "model.safetensors", weights)
            model.model_tensors = Qwen3_5TextModel.index_tensors(model)

            model.prepare_tensors()

        tensor_info = model.gguf_writer.tensors[0]
        self.assertEqual(tensor_info["token_embd.weight"].dtype, gguf.GGMLQuantizationType.I8)
        self.assertEqual(tensor_info["output.weight"].dtype, gguf.GGMLQuantizationType.I8)
        self.assertEqual(tensor_info["token_embd.weight_scale"].dtype, gguf.GGMLQuantizationType.F16)
        self.assertEqual(tensor_info["output.weight_scale"].dtype, gguf.GGMLQuantizationType.F16)

    def test_endpoint_pairs_remain_i8_and_f16_under_canonical_names(self):
        model, weights = _model_for_escha_tests()

        model.prepare_tensors()

        tensors = {name: data for name, data, _ in model.gguf_writer.tensors}
        self.assertEqual(set(tensors), {
            "token_embd.weight",
            "token_embd.weight_scale",
            "output.weight",
            "output.weight_scale",
        })
        self.assertEqual(tensors["token_embd.weight"].dtype, np.int8)
        self.assertEqual(tensors["output.weight"].dtype, np.int8)
        self.assertEqual(tensors["token_embd.weight_scale"].dtype, np.float16)
        self.assertEqual(tensors["output.weight_scale"].dtype, np.float16)
        np.testing.assert_array_equal(tensors["token_embd.weight"], weights["model.embed_tokens.weight_int8"].numpy())
        np.testing.assert_array_equal(tensors["output.weight"], weights["lm_head.weight_int8"].numpy())
        np.testing.assert_array_equal(
            tensors["token_embd.weight_scale"], weights["model.embed_tokens.weight_scale"].numpy()
        )
        np.testing.assert_array_equal(tensors["output.weight_scale"], weights["lm_head.weight_scale"].numpy())


    def test_dequantization_does_not_consume_endpoint_pairs(self):
        model, weights = _model_for_escha_tests()

        ModelBase.dequant_model(model)

        self.assertEqual(set(model.model_tensors), set(weights))
        self.assertEqual(model.model_tensors["model.embed_tokens.weight_int8"]().dtype, torch.int8)
        self.assertEqual(model.model_tensors["model.embed_tokens.weight_scale"]().dtype, torch.float16)
        self.assertEqual(model.model_tensors["lm_head.weight_int8"]().dtype, torch.int8)
        self.assertEqual(model.model_tensors["lm_head.weight_scale"]().dtype, torch.float16)


class EschaCompanionOrderingTests(unittest.TestCase):
    def _converted_names(self, tensors):
        model = _model_for_escha_group_test(tensors)
        model.prepare_tensors()
        return [name for name, _, _ in model.gguf_writer.tensors if name != "escha_lut"]

    def test_bias_before_sidecars_is_emitted(self):
        base = "model.layers.0.mlp.down_proj"
        tensors = _escha_group(base)
        ordered = {f"{base}.bias": tensors.pop(f"{base}.bias"), **tensors}

        self.assertEqual(
            self._converted_names(ordered),
            [f"mapped.0.weight.escha_{suffix}" for suffix in ("code", "config", "rin", "rout", "s_in", "s_out", "bias")],
        )

    def test_bias_after_sidecars_is_emitted(self):
        base = "model.layers.0.mlp.down_proj"
        tensors = _escha_group(base)

        self.assertEqual(
            self._converted_names(tensors),
            [f"mapped.0.weight.escha_{suffix}" for suffix in ("code", "config", "rin", "rout", "s_in", "s_out", "bias")],
        )

    def test_complete_sidecars_without_bias_are_valid(self):
        base = "model.layers.0.mlp.down_proj"

        self.assertEqual(
            self._converted_names(_escha_group(base, include_bias=False)),
            [f"mapped.0.weight.escha_{suffix}" for suffix in ("code", "config", "rin", "rout", "s_in", "s_out")],
        )

    def test_incomplete_sidecars_fail_in_sorted_order_at_end(self):
        first = "model.layers.1.mlp.down_proj"
        second = "model.layers.0.mlp.down_proj"
        tensors = {
            **{name: value for name, value in _escha_group(first).items() if name.endswith(".escha_code")},
            **{name: value for name, value in _escha_group(second).items() if name.endswith(".escha_code") or name.endswith(".escha_rin")},
        }
        model = _model_for_escha_group_test(tensors)

        with self.assertRaisesRegex(
            ValueError,
            r"Incomplete Escha companion group\(s\): model\.layers\.0\.mlp\.down_proj \(missing config, rout, s_in, s_out\); model\.layers\.1\.mlp\.down_proj \(missing config, rin, rout, s_in, s_out\)",
        ):
            model.prepare_tensors()
