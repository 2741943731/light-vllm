"""Exercise the real Runner and hybrid model with a CPU paged-attention oracle.

These checks validate batching and cache ownership, not CUDA/FlashAttention.
Run with Python >= 3.10 and the ordinary torch/transformers dependencies.
"""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from test_qwen3_5_model import CPUAttention, Qwen3_5ForCausalLM, make_config, package


# Keep nanovllm.__init__ from eagerly importing the public GPU entry point.
# Restore only project modules; external packages can register C++ operators
# during import and must remain cached rather than be imported a second time.
namespace = ModuleType("nanovllm")
namespace.__path__ = [str(Path(__file__).resolve().parents[1] / "nanovllm")]
saved_modules = {name: module for name, module in sys.modules.items()
                 if name == "nanovllm" or name.startswith("nanovllm.")}
for name in saved_modules:
    del sys.modules[name]
sys.modules["nanovllm"] = namespace
try:
    runner_module = importlib.import_module("nanovllm.engine.model_runner")
    context_module = importlib.import_module("nanovllm.utils.context")
    sequence_module = importlib.import_module("nanovllm.engine.sequence")
    scheduler_module = importlib.import_module("nanovllm.engine.scheduler")
    engine_module = importlib.import_module("nanovllm.engine.llm_engine")
    sampling_module = importlib.import_module("nanovllm.sampling_params")
finally:
    for name in list(sys.modules):
        if name == "nanovllm" or name.startswith("nanovllm."):
            del sys.modules[name]
    sys.modules.update(saved_modules)

Qwen3_5Runtime = importlib.import_module(package.__name__ + ".runtime").Qwen3_5Runtime
ModelRunner = runner_module.ModelRunner
Sequence = sequence_module.Sequence
SamplingParams = sampling_module.SamplingParams


class PagedCPUAttention(nn.Module):
    """Reference attention reading the same physical block tables as the runner."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.k_cache = self.v_cache = torch.empty(0)

    def forward(self, q, k, v):
        context = context_module.get_context()
        if self.k_cache.numel():
            slots = context.slot_mapping.long()
            self.k_cache.flatten(0, 1)[slots] = k
            self.v_cache.flatten(0, 1)[slots] = v
        bounds = context.cu_seqlens_q.tolist() if context.is_prefill else list(range(q.size(0) + 1))
        outputs = []
        for row, (start, end) in enumerate(zip(bounds, bounds[1:])):
            if context.block_tables is None:
                ki, vi = k[start:end], v[start:end]
            else:
                if context.is_prefill:
                    length = int(context.cu_seqlens_k[row + 1] - context.cu_seqlens_k[row])
                else:
                    length = int(context.context_lens[row])
                block_count = (length + self.k_cache.size(1) - 1) // self.k_cache.size(1)
                blocks = context.block_tables[row, :block_count].long()
                ki = self.k_cache[blocks].flatten(0, 1)[:length]
                vi = self.v_cache[blocks].flatten(0, 1)[:length]
            past = ki.size(0) - (end - start)
            repeats = q.size(1) // k.size(1)
            ki, vi = ki.repeat_interleave(repeats, 1), vi.repeat_interleave(repeats, 1)
            scores = q[start:end].transpose(0, 1) @ ki.transpose(0, 1).transpose(-1, -2) * self.scale
            future = torch.arange(ki.size(0))[None, :] > past + torch.arange(end - start)[:, None]
            weights = scores.masked_fill(future, float("-inf")).float().softmax(-1).to(q.dtype)
            outputs.append((weights @ vi.transpose(0, 1)).transpose(0, 1))
        result = torch.cat(outputs)
        return result if context.is_prefill else result.unsqueeze(1)


class RecordingSampler:

    def __call__(self, logits, temperatures):
        self.logits = logits.clone()
        return logits.argmax(-1)


class RunnerTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(151)
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 2
        self.addCleanup(setattr, Sequence, "block_size", self.old_block_size)
        self.addCleanup(context_module.reset_context)
        # Only transport is replaced. Packing, Context construction, numerical
        # model operations, state management and sampling dispatch remain real.
        real_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            if kwargs.pop("pin_memory", False):
                self.assertEqual(kwargs.get("device"), "cpu")
            return real_tensor(*args, **kwargs)

        self.tensor_patch = patch.object(torch, "tensor", side_effect=cpu_tensor)
        self.cuda_patch = patch.object(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
        self.tensor_patch.start()
        self.cuda_patch.start()
        self.addCleanup(self.tensor_patch.stop)
        self.addCleanup(self.cuda_patch.stop)
        self.text_config = make_config(dtype=torch.float32)
        self.backends = []

        def backend_factory(_):
            backend = PagedCPUAttention(self.text_config.head_dim ** -0.5)
            self.backends.append(backend)
            return backend

        self.model = Qwen3_5ForCausalLM(self.text_config, backend_factory).eval()
        self.runner = ModelRunner.__new__(ModelRunner)
        self.runner.model = self.model
        self.runner.qwen3_5_runtime = Qwen3_5Runtime(self.model, 3)
        self.runner.config = SimpleNamespace(
            is_qwen3_5=True, model_config=self.text_config, hf_config=self.text_config,
            max_num_delta_states=3, max_num_seqs=3, max_num_batched_tokens=8,
            max_model_len=8, kvcache_block_size=2, num_kvcache_blocks=16,
            gpu_memory_utilization=0.75,
        )
        self.runner.block_size = 2
        self.runner.world_size = 1
        self.runner.rank = 0
        self.runner.enforce_eager = True
        self.runner.sampler = RecordingSampler()
        for backend in self.backends:
            backend.k_cache = torch.empty(16, 2, 2, 8)
            backend.v_cache = torch.empty_like(backend.k_cache)

    def sequence(self, tokens, blocks, scheduled=None):
        seq = Sequence(tokens, SamplingParams(max_tokens=3, ignore_eos=True))
        seq.block_table = blocks
        seq.num_scheduled_tokens = len(tokens) if scheduled is None else scheduled
        return seq

    @torch.inference_mode()
    def reference(self, tokens):
        backend = CPUAttention(self.text_config.head_dim ** -0.5)
        model = Qwen3_5ForCausalLM(self.text_config, lambda _: backend).eval()
        model.load_state_dict(self.model.state_dict())
        hidden, states = model(torch.tensor(tokens), torch.arange(len(tokens)))
        return model.compute_logits(hidden[-1:]), states

    def assert_logits(self, tokens_by_row):
        expected = torch.cat([self.reference(tokens)[0] for tokens in tokens_by_row])
        torch.testing.assert_close(self.runner.sampler.logits, expected, rtol=3e-5, atol=3e-6)

    def test_prefill_and_decode_packing_use_matching_context(self):
        a = self.sequence([1, 2, 3], [4, 1])
        b = self.sequence([4, 5], [7])
        ids, positions, bounds = self.runner.prepare_prefill([a, b])
        context = context_module.get_context()
        self.assertEqual(ids.tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(positions.tolist(), [0, 1, 2, 0, 1])
        self.assertEqual(bounds, [0, 3, 5])
        self.assertEqual(context.cu_seqlens_q.tolist(), bounds)
        self.assertEqual(context.slot_mapping.tolist(), [8, 9, 2, 14, 15])
        self.assertIsNone(context.block_tables)
        a.append_token(6)
        b.append_token(7)
        b.block_table.append(2)
        ids, positions, bounds = self.runner.prepare_decode([b, a])
        context = context_module.get_context()
        self.assertEqual(ids.tolist(), [7, 6])
        self.assertEqual(positions.tolist(), [2, 3])
        self.assertEqual(bounds, [0, 1, 2])
        self.assertEqual(context.slot_mapping.tolist(), [4, 3])
        self.assertEqual(context.block_tables.tolist(), [[7, 2], [4, 1]])
        self.assertEqual(context.context_lens.tolist(), [3, 4])

    @torch.inference_mode()
    def test_packed_prefill_and_reordered_decode_match_independent_models(self):
        a = self.sequence([1, 2, 3], [4, 1])
        b = self.sequence([4, 5], [7])
        self.runner.run([a, b], True)
        self.assert_logits([a.token_ids, b.token_ids])
        self.assertIsNone(context_module.get_context().slot_mapping)
        for seq, token in ((a, 6), (b, 7)):
            seq.num_cached_tokens = len(seq)
            seq.append_token(token)
            seq.num_scheduled_tokens = 1
        b.block_table.append(2)
        self.runner.run([b, a], False)
        self.assert_logits([b.token_ids, a.token_ids])
        for seq in (a, b):
            expected = self.reference(seq.token_ids)[1]
            actual = self.runner.qwen3_5_runtime.pool.gather([seq.seq_id])
            for layer in expected:
                for actual_tensor, expected_tensor in zip(actual[layer], expected[layer]):
                    torch.testing.assert_close(actual_tensor, expected_tensor, rtol=3e-5, atol=3e-6)

    @torch.inference_mode()
    def test_chunked_prefill_reuses_kv_and_delta_state(self):
        seq = self.sequence([1, 2, 3, 4, 5], [5, 0, 6], scheduled=2)
        self.runner.run([seq], True)
        self.assert_logits([seq.token_ids[:2]])
        seq.num_cached_tokens = 2
        seq.num_scheduled_tokens = 3
        ids, positions, bounds = self.runner.prepare_prefill([seq])
        context = context_module.get_context()
        self.assertEqual(ids.tolist(), [3, 4, 5])
        self.assertEqual(positions.tolist(), [2, 3, 4])
        self.assertEqual(bounds, [0, 3])
        self.assertEqual(context.cu_seqlens_k.tolist(), [0, 5])
        self.assertEqual(context.block_tables.tolist(), [[5, 0, 6]])
        self.runner.run([seq], True)
        self.assert_logits([seq.token_ids])

    @torch.inference_mode()
    def test_warmup_leaves_no_request_slots(self):
        for backend in self.backends:
            backend.k_cache = backend.v_cache = torch.empty(0)
        seq = self.sequence([1, 2, 3], [])
        pool = self.runner.qwen3_5_runtime.pool
        self.runner.run([seq], True, warmup=True)
        self.assertEqual(pool.num_free_slots, 3)
        self.assertNotIn(seq.seq_id, pool)
        self.assert_logits([seq.token_ids])

    @torch.inference_mode()
    def test_release_and_reused_slot_start_with_fresh_memory(self):
        seq = self.sequence([1, 2], [4])
        self.runner.run([seq], True)
        self.runner.release_states([seq.seq_id])
        self.assertEqual(self.runner.qwen3_5_runtime.pool.num_free_slots, 3)
        other = self.sequence([3, 4], [4])
        self.runner.run([other], True)
        self.assert_logits([other.token_ids])

    def test_context_resets_when_model_or_sampler_fails(self):
        seq = self.sequence([1, 2], [4])
        with patch.object(self.runner, "run_model", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.runner.run([seq], True)
        self.assertIsNone(context_module.get_context().slot_mapping)
        self.runner.sampler = Mock(side_effect=RuntimeError("sampler failed"))
        with self.assertRaisesRegex(RuntimeError, "sampler failed"):
            self.runner.run([seq], True)
        self.assertIsNone(context_module.get_context().slot_mapping)

    def test_legacy_eager_forward_keeps_tensor_interface(self):
        model = Mock()
        hidden = torch.randn(3, 12)
        expected = torch.randn(2, 32)
        model.return_value = hidden
        model.compute_logits.return_value = expected
        self.runner.model = model
        self.runner.qwen3_5_runtime = None
        ids, positions = torch.tensor([1, 2, 3]), torch.arange(3)
        actual = self.runner.run_model(ids, positions, True)
        self.assertIs(actual, expected)
        model.assert_called_once_with(ids, positions)
        model.compute_logits.assert_called_once_with(hidden)

    def test_kv_allocation_counts_only_full_attention_layers(self):
        # Persistent DeltaNet storage is already included in used/current.
        # The allocator should reserve the observed temporary peak once.
        stats = {"allocated_bytes.all.peak": 6000, "allocated_bytes.all.current": 4000}
        with patch.object(torch.cuda, "mem_get_info", return_value=(16000, 20000)), \
                patch.object(torch.cuda, "memory_stats", return_value=stats):
            self.runner.allocate_kv_cache()
        block_bytes = 2 * 1 * 2 * 2 * 8 * 4
        expected_blocks = (int(20000 * 0.75) - 4000 - 6000 + 4000) // block_bytes
        self.assertEqual(self.runner.config.num_kvcache_blocks, expected_blocks)
        self.assertEqual(tuple(self.runner.kv_cache.shape), (2, 1, expected_blocks, 2, 2, 8))
        self.assertEqual(self.backends[0].k_cache.data_ptr(), self.runner.kv_cache[0, 0].data_ptr())
        self.assertEqual(self.backends[0].v_cache.data_ptr(), self.runner.kv_cache[1, 0].data_ptr())

    @torch.inference_mode()
    def test_engine_steps_keep_chunked_requests_and_release_finished_slots(self):
        config = self.runner.config
        config.max_num_delta_states = 1
        config.max_num_batched_tokens = 2
        config.eos = -1
        scheduler = scheduler_module.Scheduler(config)
        first = Sequence([1, 2, 3], SamplingParams(max_tokens=2, ignore_eos=True))
        second = Sequence([4, 5], SamplingParams(max_tokens=2, ignore_eos=True))
        scheduler.add(first)
        scheduler.add(second)
        engine = engine_module.LLMEngine.__new__(engine_module.LLMEngine)
        engine.scheduler = scheduler
        engine.model_runner = self.runner
        completed = {}
        for _ in range(12):
            outputs, _ = engine.step()
            completed.update(outputs)
            for seq_id, _ in outputs:
                self.assertNotIn(seq_id, self.runner.qwen3_5_runtime.pool)
            if scheduler.is_finished():
                break
        self.assertTrue(scheduler.is_finished())
        self.assertEqual(set(completed), {first.seq_id, second.seq_id})
        self.assertEqual(len(completed[first.seq_id]), 2)
        self.assertEqual(len(completed[second.seq_id]), 2)
        self.assertEqual(self.runner.qwen3_5_runtime.pool.num_free_slots, 3)


if __name__ == "__main__":
    unittest.main()
