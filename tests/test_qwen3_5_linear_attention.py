"""CPU checks: python -m unittest discover -s tests -p 'test_qwen3_5_linear_attention.py'."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

import torch


# Load the independent model package without importing the CUDA engine in nanovllm.__init__.
package = ModuleType("_qwen35_cpu_tests")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "nanovllm/models/qwen3_5")]
sys.modules[package.__name__] = package
module = importlib.import_module(package.__name__ + ".linear_attention")
Qwen3_5GatedDeltaNet = module.Qwen3_5GatedDeltaNet
GatedDeltaNetState = module.GatedDeltaNetState


class GatedDeltaNetTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = Qwen3_5GatedDeltaNet(12, 2, 4, 3, 5).eval()

    def assert_state_close(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual.conv_state, expected.conv_state, **kwargs)
        torch.testing.assert_close(actual.recurrent_state, expected.recurrent_state, **kwargs)
        self.assertEqual(actual.recurrent_state.dtype, torch.float32)

    @torch.inference_mode()
    def test_recurrence_matches_matrix_transition(self):
        q, k = torch.randn(7, 2, 3), torch.randn(7, 2, 3)
        v = torch.randn(7, 2, 5)
        g, beta = -torch.rand(7, 2), torch.rand(7, 2)
        initial = torch.randn(2, 3, 5)
        actual, final = module.recurrent_gated_delta_rule(q, k, v, g, beta, initial)
        state, expected = initial.clone(), []
        # Independent matrix form: S_t = alpha (I - beta kk^T) S_{t-1} + beta kv^T.
        for t in range(7):
            kt = k[t] / torch.sqrt(k[t].square().sum(-1, keepdim=True) + 1e-6)
            qt = q[t] / torch.sqrt(q[t].square().sum(-1, keepdim=True) + 1e-6) / (3 ** 0.5)
            transition = torch.eye(3) - beta[t, :, None, None] * kt.unsqueeze(-1) @ kt.unsqueeze(-2)
            state = g[t].exp()[:, None, None] * (transition @ state)
            state = state + beta[t, :, None, None] * (kt.unsqueeze(-1) @ v[t].unsqueeze(-2))
            expected.append((qt.unsqueeze(-2) @ state).squeeze(-2))
        torch.testing.assert_close(actual, torch.stack(expected), rtol=2e-5, atol=1e-6)
        torch.testing.assert_close(final, state, rtol=2e-5, atol=1e-6)

    @torch.inference_mode()
    def test_chunked_prefill_and_decode_match_full_sequence(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = self.model.to(dtype)
            x = torch.randn(17, 12).to(dtype)
            initial = model.init_state(1)
            initial.conv_state.normal_()
            initial.recurrent_state.normal_()
            full_output, full_state = model(x, state=initial)
            for chunks in ([2, 1, 3, 11], [1] * 17):
                outputs, state, start = [], initial, 0
                for length in chunks:
                    output, state = model(x[start:start + length], state=state)
                    outputs.append(output)
                    start += length
                tolerance = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=0.02, atol=0.002)
                torch.testing.assert_close(torch.cat(outputs), full_output, **tolerance)
                self.assert_state_close(state, full_state, **tolerance)

    @torch.inference_mode()
    def test_packed_sequences_are_isolated_and_can_be_reordered(self):
        x, bounds = torch.randn(10, 12), [0, 1, 5, 10]
        initial = self.model.init_state(3)
        initial.conv_state.normal_()
        initial.recurrent_state.normal_()
        before = GatedDeltaNetState(*(tensor.clone() for tensor in initial))
        packed, state = self.model(x, bounds, initial)
        for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
            single_state = GatedDeltaNetState(*(tensor[i:i + 1] for tensor in initial))
            single, final = self.model(x[start:end], state=single_state)
            torch.testing.assert_close(packed[start:end], single)
            self.assert_state_close(GatedDeltaNetState(*(tensor[i:i + 1] for tensor in state)), final)
        self.assert_state_close(initial, before, rtol=0, atol=0)
        changed = x.clone()
        changed[1:5] += 10
        changed_output, _ = self.model(changed, bounds, initial)
        torch.testing.assert_close(changed_output[:1], packed[:1], rtol=0, atol=0)
        torch.testing.assert_close(changed_output[5:], packed[5:], rtol=0, atol=0)
        order = torch.tensor([2, 0, 1])
        tokens = torch.randn(3, 12)
        output, final = self.model(tokens, [0, 1, 2, 3], state)
        reordered = GatedDeltaNetState(*(tensor[order] for tensor in state))
        reordered_output, reordered_final = self.model(tokens[order], [0, 1, 2, 3], reordered)
        torch.testing.assert_close(reordered_output, output[order])
        self.assert_state_close(reordered_final, GatedDeltaNetState(*(tensor[order] for tensor in final)))

    @torch.inference_mode()
    def test_convolution_is_causal_and_retains_raw_history(self):
        x = torch.randn(7, 12)
        output, state = self.model(x)
        changed = x.clone()
        changed[4:] += 10
        changed_output, _ = self.model(changed)
        torch.testing.assert_close(output[:4], changed_output[:4], rtol=0, atol=0)
        expected = self.model.in_proj_qkv(x)[-4:].transpose(0, 1).unsqueeze(0)
        torch.testing.assert_close(state.conv_state, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_equal_heads_and_unit_convolution_kernel(self):
        model = Qwen3_5GatedDeltaNet(8, 2, 2, 3, 5, conv_kernel_size=1)
        x = torch.randn(3, 8)
        full, expected_state = model(x)
        first, state = model(x[:1])
        rest, state = model(x[1:], state=state)
        torch.testing.assert_close(torch.cat((first, rest)), full)
        self.assert_state_close(state, expected_state)

    def test_invalid_inputs_and_state_are_rejected(self):
        x = torch.randn(3, 12)
        for bounds in ([1, 3], [0, 2], [0, 2, 2, 3], [0, -1, 3], [0, 1.5, 3], [0]):
            with self.assertRaises(ValueError):
                self.model(x, bounds)
        with self.assertRaises(ValueError):
            self.model(torch.empty(0, 12))
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 3, 12))
        with self.assertRaises(ValueError):
            self.model(x, state=self.model.init_state(2))
        with self.assertRaises(ValueError):
            self.model(x, state=GatedDeltaNetState(self.model.init_state(1).conv_state, torch.zeros(1, 4, 3, 5).bfloat16()))
        with self.assertRaises(ValueError):
            Qwen3_5GatedDeltaNet(12, 3, 4, 3, 5)


if __name__ == "__main__":
    unittest.main()
