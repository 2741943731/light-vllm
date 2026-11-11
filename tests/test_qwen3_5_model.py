"""CPU checks: python -m unittest discover -s tests -p 'test_qwen3_5_model.py'."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


# Avoid importing the CUDA engine in nanovllm.__init__.
package = ModuleType("_qwen35_model_tests")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "nanovllm/models/qwen3_5")]
sys.modules[package.__name__] = package
model_module = importlib.import_module(package.__name__ + ".model")
Qwen3_5DecoderLayer = model_module.Qwen3_5DecoderLayer
Qwen3_5Model = model_module.Qwen3_5Model
Qwen3_5ForCausalLM = model_module.Qwen3_5ForCausalLM

# The reference backend stores a separate KV history for each packed sequence.
from test_qwen3_5_attention import CPUAttention


def make_config(**changes):
    values = dict(
        hidden_size=12,
        intermediate_size=20,
        hidden_act="silu",
        vocab_size=32,
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class ModelTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(29)
        self.backends = {}

        def make_backend(layer_idx):
            backend = CPUAttention(8 ** -0.5)
            self.backends[layer_idx] = backend
            return backend

        self.factory = make_backend
        self.model = Qwen3_5ForCausalLM(make_config(), attention_backend_factory=self.factory).eval()
        self.ids = torch.tensor([1, 7, 2, 6, 5, 9, 4, 3, 8], dtype=torch.long)
        self.positions = torch.arange(self.ids.numel())

    def clear_backends(self, bounds=None, decode=False):
        for backend in self.backends.values():
            backend.cache.clear()
            backend.bounds = bounds
            backend.decode = decode

    def assert_states_close(self, actual, expected):
        self.assertEqual(set(actual), {0, 1, 2})
        self.assertEqual(set(actual), set(expected))
        for index in actual:
            torch.testing.assert_close(actual[index].conv_state, expected[index].conv_state, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(actual[index].recurrent_state, expected[index].recurrent_state, rtol=3e-5, atol=3e-6)

    @torch.inference_mode()
    def test_decoder_residual_order_for_both_layer_types(self):
        x, positions, bounds = torch.randn(4, 12), torch.arange(4), [0, 4]
        for index in (0, 3):
            with self.subTest(layer=index):
                backend = CPUAttention(8 ** -0.5) if index == 3 else None
                layer = Qwen3_5DecoderLayer(make_config(), index, backend)
                result, state = layer(positions, x, bounds)
                normalized = layer.input_layernorm(x)
                if index == 0:
                    mixed, expected_state = layer.linear_attn(normalized, bounds)
                    torch.testing.assert_close(state.conv_state, expected_state.conv_state)
                    torch.testing.assert_close(state.recurrent_state, expected_state.recurrent_state)
                else:
                    self.assertIsNone(state)
                    backend.cache.clear()
                    mixed = layer.self_attn(positions, normalized)
                residual = x + mixed
                expected = residual + layer.mlp(layer.post_attention_layernorm(residual))
                torch.testing.assert_close(result, expected)

    def test_factory_builds_one_backend_per_full_layer(self):
        types = make_config().layer_types * 2
        model = Qwen3_5Model(make_config(num_hidden_layers=8, layer_types=types), self.factory)
        self.assertEqual(set(self.backends), {3, 7})
        self.assertIsNot(self.backends[3], self.backends[7])
        self.assertEqual(set(model.init_state(1)), {0, 1, 2, 4, 5, 6})

    @torch.inference_mode()
    def test_mixed_layers_and_weight_tying(self):
        self.assertEqual(set(self.backends), {3})
        self.assertIs(self.model.lm_head.weight, self.model.model.embed_tokens.weight)
        states = self.model.init_state(2)
        self.assertEqual(set(states), {0, 1, 2})
        self.assertTrue(all(s.conv_state.shape[0] == 2 for s in states.values()))
        hidden, returned = self.model(self.ids, self.positions)
        self.assertEqual(hidden.shape, (9, 12))
        self.assertEqual(set(returned), {0, 1, 2})
        self.assertTrue(all(s.recurrent_state.shape[0] == 1 for s in returned.values()))
        # Both the nested and direct text configuration routes must work.
        other_backends = []
        def factory(index):
            other_backends.append(index)
            return CPUAttention(8 ** -0.5)
        nested = Qwen3_5Model(SimpleNamespace(text_config=make_config()), attention_backend_factory=factory)
        direct = Qwen3_5Model(make_config(), attention_backend_factory=factory)
        self.assertEqual(other_backends, [3, 3])
        self.assertEqual(set(nested.init_state(1)), set(direct.init_state(1)))

    @torch.inference_mode()
    def test_packed_sequences_equal_independent_sequences(self):
        bounds = [0, 3, 9]
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 5])
        self.clear_backends(bounds)
        packed, packed_states = self.model(self.ids, positions, bounds)
        for seq, (start, end) in enumerate(zip(bounds, bounds[1:])):
            self.clear_backends()
            single, single_states = self.model(self.ids[start:end], positions[start:end])
            torch.testing.assert_close(packed[start:end], single, rtol=3e-5, atol=3e-6)
            for index in packed_states:
                torch.testing.assert_close(packed_states[index].conv_state[seq:seq + 1], single_states[index].conv_state, rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(packed_states[index].recurrent_state[seq:seq + 1], single_states[index].recurrent_state, rtol=3e-5, atol=3e-6)

    @torch.inference_mode()
    def test_full_chunked_and_decode_match_with_both_cache_types(self):
        self.clear_backends()
        full, full_states = self.model(self.ids, self.positions)
        for chunks in ([2, 3, 4], [1] * len(self.ids)):
            self.clear_backends()
            states, start, outputs = None, 0, []
            for length in chunks:
                # KV history lives in the full-attention backend; linear states
                # are explicitly passed back to the model on every step.
                for backend in self.backends.values():
                    backend.decode = length == 1
                output, states = self.model(self.ids[start:start + length], self.positions[start:start + length], states=states)
                outputs.append(output)
                start += length
            torch.testing.assert_close(torch.cat(outputs), full, rtol=3e-5, atol=3e-6)
            self.assert_states_close(states, full_states)
            self.assertEqual(self.backends[3].cache[0][0].shape[0], len(self.ids))

    @torch.inference_mode()
    def test_last_token_logits(self):
        bounds = [0, 3, 9]
        self.clear_backends(bounds)
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 5])
        hidden, _ = self.model(self.ids, positions, bounds)
        all_logits = self.model.compute_logits(hidden)
        selected = self.model.compute_logits(hidden, bounds)
        self.assertEqual(all_logits.shape, (9, 32))
        self.assertEqual(selected.shape, (2, 32))
        torch.testing.assert_close(selected, all_logits[[2, 8]])
        torch.testing.assert_close(all_logits, F.linear(hidden, self.model.model.embed_tokens.weight))

    def test_invalid_config_and_missing_states(self):
        invalid = [
            dict(layer_types=["linear_attention"]),
            dict(layer_types=["linear_attention", "linear_attention", "linear_attention", "unknown"]),
            dict(hidden_act="gelu"),
            dict(rope_parameters={"rope_type": "linear", "rope_theta": 10000000, "partial_rotary_factor": 0.25}),
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Qwen3_5Model(make_config(**changes), attention_backend_factory=self.factory)
        states = self.model.init_state(1)
        del states[1]
        with self.assertRaises(ValueError):
            self.model(self.ids, self.positions, states=states)


if __name__ == "__main__":
    unittest.main()
