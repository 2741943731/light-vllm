"""Configuration checks without loading checkpoints or importing CUDA kernels."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import warnings


try:
    import torch
except ImportError:
    # Metadata checks can also run on a Python installation without PyTorch.
    torch = ModuleType("torch")
    torch.float16, torch.bfloat16, torch.float32 = object(), object(), object()


class AutoConfig:
    @classmethod
    def from_pretrained(cls, path):
        raise AssertionError("Tests must provide a local configuration")


transformers = ModuleType("transformers")
transformers.AutoConfig = AutoConfig
path = Path(__file__).resolve().parents[1] / "nanovllm/config.py"
spec = importlib.util.spec_from_file_location("_qwen35_config_tests", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
with patch.dict(sys.modules, {"transformers": transformers, "torch": torch}):
    spec.loader.exec_module(module)
Config = module.Config


def make_text_config(**changes):
    values = dict(
        model_type="qwen3_5_text",
        max_position_embeddings=8192,
        dtype="bfloat16",
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    values.update(changes)
    return SimpleNamespace(**values)


class ConfigTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def config(self, hf_config, **kwargs):
        with patch.object(AutoConfig, "from_pretrained", return_value=hf_config):
            return Config(self.directory.name, **kwargs)

    def test_nested_text_config_and_default_resident_limit(self):
        text = make_text_config()
        outer = SimpleNamespace(model_type="qwen3_5", text_config=text)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self.config(outer, max_model_len=16384)
        self.assertTrue(any(issubclass(w.category, UserWarning) and "eager" in str(w.message) for w in caught))
        self.assertTrue(config.is_qwen3_5)
        self.assertIs(config.hf_config, outer)
        self.assertIs(config.model_config, text)
        self.assertIs(text.dtype, torch.bfloat16)
        self.assertEqual(config.max_model_len, 8192)
        self.assertEqual(config.max_num_delta_states, 16)
        self.assertEqual(config.max_num_seqs, 512)
        self.assertTrue(config.enforce_eager)

    def test_direct_text_config_and_explicit_eager_are_supported(self):
        text = make_text_config(dtype=torch.float16)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self.config(text, max_num_seqs=3, enforce_eager=True)
        self.assertEqual(caught, [])
        self.assertIs(config.model_config, text)
        self.assertEqual(config.max_num_delta_states, 3)
        self.assertEqual(config.max_model_len, 4096)

    def test_explicit_resident_limit_and_legacy_dtype(self):
        text = make_text_config(dtype=None, torch_dtype="float16")
        config = self.config(text, max_num_seqs=2, max_num_delta_states=7, enforce_eager=True)
        self.assertEqual(config.max_num_delta_states, 7)
        self.assertEqual(config.max_num_seqs, 2)
        self.assertIs(config.model_config.dtype, torch.float16)

    def test_qwen3_configuration_preserves_existing_behavior(self):
        hf = SimpleNamespace(model_type="qwen3", max_position_embeddings=2048, dtype=torch.float32)
        config = self.config(hf, tensor_parallel_size=2)
        self.assertFalse(config.is_qwen3_5)
        self.assertFalse(config.enforce_eager)
        self.assertIs(config.model_config, hf)
        self.assertEqual(config.tensor_parallel_size, 2)
        self.assertEqual(config.max_model_len, 2048)
        self.assertIsNone(config.max_num_delta_states)
        self.assertIs(hf.dtype, torch.float32)

    def test_unsupported_parallelism_dtype_and_layer_layout(self):
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            self.config(make_text_config(), tensor_parallel_size=2)
        for dtype in (torch.float32, "float32", "invalid", None):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "float16 or bfloat16"):
                self.config(make_text_config(dtype=dtype))
        for layers in (None, [], ["full_attention"] * 4, ["linear_attention"] * 4,
                       ["linear_attention", "full_attention"], ["invalid"] * 4):
            with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, "layer_types"):
                self.config(make_text_config(layer_types=layers))

    def test_invalid_resource_limits_are_rejected(self):
        for changes in (
            dict(max_num_delta_states=0), dict(max_num_delta_states=True),
            dict(max_num_delta_states=-1), dict(max_num_delta_states=1.5),
            dict(max_num_seqs=0), dict(max_num_batched_tokens=0), dict(max_model_len=-1),
            dict(gpu_memory_utilization=0), dict(gpu_memory_utilization=1.1),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(make_text_config(), **changes)

    def test_unsupported_transformers_error_is_actionable(self):
        error = ValueError("Transformers does not recognize model type qwen3_5")
        with patch.object(AutoConfig, "from_pretrained", side_effect=error):
            with self.assertRaisesRegex(ValueError, "Transformers version") as caught:
                Config(self.directory.name)
        self.assertIs(caught.exception.__cause__, error)

    def test_unrelated_configuration_errors_are_preserved(self):
        error = ValueError("invalid local checkpoint")
        with patch.object(AutoConfig, "from_pretrained", side_effect=error):
            with self.assertRaises(ValueError) as caught:
                Config(self.directory.name)
        self.assertIs(caught.exception, error)

    def test_real_transformers_nested_and_text_config_roundtrip(self):
        try:
            from transformers import AutoConfig as HFAutoConfig, Qwen3_5Config, Qwen3_5TextConfig
        except ImportError:
            self.skipTest("Transformers with Qwen3.5 support is not installed")
        values = dict(dtype="bfloat16", num_hidden_layers=4, max_position_embeddings=2048)
        for hf in (Qwen3_5Config(text_config=values), Qwen3_5TextConfig(**values)):
            with self.subTest(model_type=hf.model_type):
                hf.save_pretrained(self.directory.name)
                with patch.object(module, "AutoConfig", HFAutoConfig):
                    config = Config(self.directory.name, enforce_eager=True)
                self.assertTrue(config.is_qwen3_5)
                self.assertEqual(config.max_model_len, 2048)
                self.assertEqual(config.model_config.num_hidden_layers, 4)
                self.assertIs(config.model_config.dtype, torch.bfloat16)

    def test_real_transformers_qwen3_config_roundtrip(self):
        try:
            from transformers import AutoConfig as HFAutoConfig, Qwen3Config
        except ImportError:
            self.skipTest("Transformers with Qwen3 support is not installed")
        Qwen3Config(dtype="float16", max_position_embeddings=2048).save_pretrained(self.directory.name)
        with patch.object(module, "AutoConfig", HFAutoConfig):
            config = Config(self.directory.name, tensor_parallel_size=2)
        self.assertFalse(config.is_qwen3_5)
        self.assertIs(config.model_config, config.hf_config)
        self.assertIs(config.model_config.dtype, torch.float16)
        self.assertFalse(config.enforce_eager)


if __name__ == "__main__":
    unittest.main()
