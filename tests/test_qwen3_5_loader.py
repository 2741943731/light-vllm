"""CPU checks for real safetensors files and weight preflight."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from test_qwen3_5_model import Qwen3_5ForCausalLM, make_config, CPUAttention, package

loader = importlib.import_module(package.__name__ + ".loader")


def write_shard(path, tensors):
    save_file({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()}, str(path))


def write_index(path, weight_map):
    path.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))


def make_model(seed):
    torch.manual_seed(seed)
    return Qwen3_5ForCausalLM(make_config(), lambda _: CPUAttention(8 ** -0.5)).eval()


def checkpoint_weights(model):
    result = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if name == "lm_head.weight":
            continue
        if name.startswith("model."):
            name = "model.language_model." + name[len("model."):]
        result[name] = param.detach().clone()
    return result


class Qwen3_5LoaderTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = make_model(100)
        self.target = make_model(999)
        self.tensors = checkpoint_weights(self.source)

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self):
        return {name: param.detach().clone() for name, param in self.target.named_parameters(remove_duplicate=False)}

    def assert_unchanged(self, before):
        for name, param in self.target.named_parameters(remove_duplicate=False):
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)

    def assert_models_equal(self):
        actual = dict(self.target.named_parameters(remove_duplicate=False))
        for name, expected in self.source.named_parameters(remove_duplicate=False):
            torch.testing.assert_close(actual[name], expected, rtol=0, atol=0)
        self.assertIs(self.target.lm_head.weight, self.target.model.embed_tokens.weight)

    def expect_failure_unchanged(self, error):
        before = self.snapshot()
        with self.assertRaises(error):
            loader.load_model(self.target, self.root)
        self.assert_unchanged(before)

    @torch.inference_mode()
    def test_official_prefix_roundtrip_and_skip_vision_mtp(self):
        extra = dict(self.tensors)
        extra["model.visual.fake.weight"] = torch.randn(2, 2)
        extra["mtp.fake.weight"] = torch.randn(2, 2)
        write_shard(self.root / "model.safetensors", extra)
        result = loader.load_model(self.target, self.root)
        self.assert_models_equal()
        self.assertEqual(result.skipped_tensors, 2)
        self.assertTrue(result.loaded_parameters)
        ids = torch.tensor([1, 2, 5, 6])
        positions = torch.arange(4)
        source_hidden, _ = self.source(ids, positions)
        target_hidden, _ = self.target(ids, positions)
        torch.testing.assert_close(target_hidden, source_hidden, rtol=0, atol=0)
        torch.testing.assert_close(self.target.compute_logits(target_hidden), self.source.compute_logits(source_hidden), rtol=0, atol=0)

    def test_index_shards_and_ignores_unlisted_file(self):
        items = list(self.tensors.items())
        first, second = dict(items[::2]), dict(items[1::2])
        write_shard(self.root / "z-1.safetensors", first)
        write_shard(self.root / "z-2.safetensors", second)
        write_shard(self.root / "unlisted.safetensors", {"unknown.weight": torch.ones(1)})
        mapping = {name: "z-1.safetensors" for name in first}
        mapping.update({name: "z-2.safetensors" for name in second})
        write_index(self.root / "model.safetensors.index.json", mapping)
        loader.load_model(self.target, self.root)
        self.assert_models_equal()

    def test_text_only_names_and_multiple_shards_without_index(self):
        names = {name.replace("model.language_model.", "model.", 1): tensor
                 for name, tensor in self.tensors.items()}
        items = list(names.items())
        write_shard(self.root / "b.safetensors", dict(items[::2]))
        write_shard(self.root / "a.safetensors", dict(items[1::2]))
        loader.load_model(self.target, self.root)
        self.assert_models_equal()

    def test_tied_alias_can_be_single_or_identical_pair(self):
        embedding = "model.language_model.embed_tokens.weight"
        for mode in ("embed", "head", "both"):
            with self.subTest(mode=mode):
                self.target = make_model(999)
                weights = dict(self.tensors)
                if mode in ("head", "both"):
                    weights["lm_head.weight"] = weights[embedding].clone()
                if mode == "head":
                    del weights[embedding]
                write_shard(self.root / "model.safetensors", weights)
                result = loader.load_model(self.target, self.root)
                self.assert_models_equal()
                self.assertEqual(result.tied_aliases, int(mode == "both"))

    def test_tied_alias_conflict_or_dtype_mismatch_preserves_parameters(self):
        embedding = "model.language_model.embed_tokens.weight"
        for mismatch in ("value", "dtype"):
            with self.subTest(mismatch=mismatch):
                weights = dict(self.tensors)
                weights["lm_head.weight"] = weights[embedding].clone()
                if mismatch == "value":
                    weights["lm_head.weight"][0, 0] += 1
                else:
                    weights["lm_head.weight"] = weights[embedding].to(torch.float16)
                write_shard(self.root / "model.safetensors", weights)
                self.expect_failure_unchanged(ValueError)

    def test_floating_source_conversion_to_target_dtype(self):
        for source_dtype, target_dtype in ((torch.float16, torch.float32),
                                           (torch.bfloat16, torch.float32),
                                           (torch.float32, torch.float32),
                                           (torch.float32, torch.bfloat16),
                                           (torch.bfloat16, torch.float16)):
            with self.subTest(source_dtype=source_dtype, target_dtype=target_dtype):
                self.target = make_model(999).to(target_dtype)
                weights = {name: value.to(source_dtype) for name, value in self.tensors.items()}
                write_shard(self.root / "model.safetensors", weights)
                loader.load_model(self.target, self.root)
                source_params = dict(self.source.named_parameters())
                for name, actual in self.target.named_parameters():
                    expected = source_params[name].detach().to(source_dtype).to(actual.dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unknown_missing_shape_and_integer_weights_preserve_parameters(self):
        key = next(name for name in self.tensors if name != "model.language_model.embed_tokens.weight")
        for problem in ("unknown", "missing", "shape", "integer"):
            with self.subTest(problem=problem):
                weights = dict(self.tensors)
                if problem == "unknown":
                    weights["model.language_model.unknown.weight"] = torch.ones(1)
                elif problem == "missing":
                    del weights[key]
                elif problem == "shape":
                    weights[key] = torch.ones(1)
                else:
                    weights[key] = weights[key].to(torch.int32)
                write_shard(self.root / "model.safetensors", weights)
                self.expect_failure_unchanged(ValueError)

    def test_duplicate_source_or_mapped_target_preserves_parameters(self):
        key = next(name for name in self.tensors if name != "model.language_model.embed_tokens.weight")
        write_shard(self.root / "a.safetensors", self.tensors)
        write_shard(self.root / "b.safetensors", {key: self.tensors[key]})
        self.expect_failure_unchanged(ValueError)
        (self.root / "b.safetensors").unlink()
        weights = dict(self.tensors)
        weights["model." + key[len("model.language_model."):]] = weights[key].clone()
        write_shard(self.root / "a.safetensors", weights)
        self.expect_failure_unchanged(ValueError)

    def test_index_missing_shard_and_mismatched_physical_keys_preserve_parameters(self):
        mapping = {name: "part.safetensors" for name in self.tensors}
        write_index(self.root / "model.safetensors.index.json", mapping)
        self.expect_failure_unchanged(FileNotFoundError)
        items = list(self.tensors.items())
        for problem in ("omitted", "unlisted"):
            with self.subTest(problem=problem):
                weights = dict(items[1:] if problem == "omitted" else items)
                if problem == "unlisted":
                    weights["unlisted.weight"] = torch.ones(1)
                write_shard(self.root / "part.safetensors", weights)
                self.expect_failure_unchanged(ValueError)

    def test_malformed_safetensors_preserves_parameters(self):
        (self.root / "model.safetensors").write_bytes(b"not a safetensors file")
        self.expect_failure_unchanged(Exception)


if __name__ == "__main__":
    unittest.main()
