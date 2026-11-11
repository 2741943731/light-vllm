"""CPU checks: python -m unittest discover -s tests -p 'test_qwen3_5_attention.py'."""
import importlib
from pathlib import Path
import sys
from types import ModuleType
import unittest

import torch
import torch.nn.functional as F
from torch import nn


package = ModuleType("_qwen35_attention_tests")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "nanovllm/models/qwen3_5")]
sys.modules[package.__name__] = package
Qwen3_5Attention = importlib.import_module(package.__name__ + ".attention").Qwen3_5Attention


class CPUAttention(nn.Module):
    """Test backend with per-sequence KV history and the engine's Q/K/V interface."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.bounds = None
        self.cache = {}
        self.decode = False

    def forward(self, q, k, v):
        self.last_qkv = (q, k, v)
        outputs = []
        bounds = self.bounds or [0, q.size(0)]
        for i, (start, end) in enumerate(zip(bounds, bounds[1:])):
            ki, vi = k[start:end], v[start:end]
            past = 0
            if i in self.cache:
                old_k, old_v = self.cache[i]
                past = old_k.size(0)
                ki, vi = torch.cat((old_k, ki)), torch.cat((old_v, vi))
            self.cache[i] = (ki, vi)
            repeats = q.size(1) // k.size(1)
            ki, vi = ki.repeat_interleave(repeats, 1), vi.repeat_interleave(repeats, 1)
            scores = q[start:end].transpose(0, 1) @ ki.transpose(0, 1).transpose(-1, -2) * self.scale
            mask = torch.arange(ki.size(0))[None, :] > past + torch.arange(end - start)[:, None]
            scores = scores.masked_fill(mask, float("-inf"))
            weights = scores.float().softmax(-1).to(q.dtype)
            outputs.append((weights @ vi.transpose(0, 1)).transpose(0, 1))
        output = torch.cat(outputs)
        return output.unsqueeze(1) if self.decode else output


def reference_forward(model, positions, x):
    """Independent per-head projection and causal attention, without the tested helpers."""
    n, h, d = x.size(0), model.num_heads, model.head_dim
    weights = model.q_proj.weight.reshape(h, 2, d, model.hidden_size)
    q = F.linear(x, weights[:, 0].reshape(h * d, -1)).reshape(n, h, d)
    gate = F.linear(x, weights[:, 1].reshape(h * d, -1)).reshape(n, h, d)
    k = model.k_proj(x).reshape(n, model.num_kv_heads, d)
    v = model.v_proj(x).reshape(n, model.num_kv_heads, d)
    def normalize(tensor, weight):
        t = tensor.float()
        return (t / torch.sqrt(t.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.float())).to(x.dtype)
    q, k = normalize(q, model.q_norm.weight), normalize(k, model.k_norm.weight)
    r = model.rotary_emb.rotary_dim
    angles = positions.float()[:, None] / model.rotary_emb.base ** (torch.arange(0, r, 2).float() / r)
    cos, sin = angles.cos().to(x.dtype)[:, None], angles.sin().to(x.dtype)[:, None]
    def rotate(tensor):
        a, b, rest = tensor[..., :r // 2], tensor[..., r // 2:r], tensor[..., r:]
        return torch.cat((a * cos - b * sin, b * cos + a * sin, rest), -1)
    q, k = rotate(q), rotate(k)
    outputs = []
    for head in range(h):
        kv_head = head // (h // model.num_kv_heads)
        scores = q[:, head] @ k[:, kv_head].T * d ** -0.5
        scores = scores.masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), float("-inf"))
        outputs.append(scores.float().softmax(-1).to(x.dtype) @ v[:, kv_head])
    output = torch.stack(outputs, dim=1) * gate.sigmoid()
    return model.o_proj(output.reshape(n, h * d))


class Qwen3_5AttentionTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        self.backend = CPUAttention(16 ** -0.5)
        self.model = Qwen3_5Attention(12, 4, 2, 16, attention_backend=self.backend).eval()
        with torch.no_grad():
            self.model.q_norm.weight.normal_(0, 0.1)
            self.model.k_norm.weight.normal_(0, 0.1)

    @torch.inference_mode()
    def test_matches_independent_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            self.model.to(dtype)
            self.backend.cache.clear()
            x = torch.randn(7, 12).to(dtype)
            positions = torch.tensor([0, 1, 2, 15, 63, 4095, 262143])
            actual = self.model(positions, x)
            expected = reference_forward(self.model, positions, x)
            tolerance = dict(rtol=3e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=0.02, atol=0.002)
            torch.testing.assert_close(actual, expected, **tolerance)
            q, k, v = self.backend.last_qkv
            self.assertEqual(q.shape, (7, 4, 16))
            self.assertEqual(k.shape, (7, 2, 16))
            self.assertEqual(k.stride(1), 16)
            self.assertEqual(v.stride(1), 16)
            self.assertEqual(k.stride(-1), 1)
            self.assertEqual(v.stride(-1), 1)

    @torch.inference_mode()
    def test_cached_chunks_and_decode_match_full_prefill(self):
        x, positions = torch.randn(9, 12), torch.arange(9)
        expected = self.model(positions, x)
        self.backend.cache.clear()
        outputs, start = [], 0
        for length in (2, 3, 1, 1, 1, 1):
            self.backend.decode = length == 1
            outputs.append(self.model(positions[start:start + length], x[start:start + length]))
            start += length
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=3e-5, atol=2e-6)
        self.assertEqual(self.backend.cache[0][0].shape, (9, 2, 16))

    @torch.inference_mode()
    def test_packed_sequences_are_isolated(self):
        x = torch.randn(8, 12)
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
        self.backend.bounds = [0, 3, 8]
        packed = self.model(positions, x)
        for start, end in ((0, 3), (3, 8)):
            self.backend.bounds = None
            self.backend.cache.clear()
            single = self.model(positions[start:end], x[start:end])
            torch.testing.assert_close(packed[start:end], single)

    @torch.inference_mode()
    def test_gate_precedes_output_projection(self):
        class ConstantAttention(nn.Module):
            def forward(self, q, k, v):
                return torch.ones_like(q)
        model = Qwen3_5Attention(12, 4, 2, 16, qkv_bias=True, attention_backend=ConstantAttention())
        model.q_proj.weight.zero_()
        model.q_proj.bias.zero_()
        gate = torch.linspace(-3, 3, 64).reshape(4, 16)
        model.q_proj.bias.reshape(4, 32)[:, 16:].copy_(gate)
        model.o_proj.bias.fill_(1.25)
        actual = model(torch.arange(2), torch.randn(2, 12))
        expected = F.linear(gate.sigmoid().reshape(1, 64).expand(2, -1), model.o_proj.weight, model.o_proj.bias)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_configuration_and_shapes(self):
        with self.assertRaises(ValueError):
            Qwen3_5Attention(12, 3, 2, 16)
        with self.assertRaises(ValueError):
            Qwen3_5Attention(12, 4, 2, 16, partial_rotary_factor=0)
        with self.assertRaises(ValueError):
            self.model(torch.arange(2), torch.randn(3, 12))
        with self.assertRaises(ValueError):
            self.model(torch.arange(3), torch.randn(1, 3, 12))


if __name__ == "__main__":
    unittest.main()
