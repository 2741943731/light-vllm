"""CPU checks for hybrid cache admission and request lifecycle."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch.multiprocessing
import transformers


# Load the actual scheduler/engine without importing the CUDA ModelRunner.
root = Path(__file__).resolve().parents[1] / "nanovllm"
package = ModuleType("nanovllm")
package.__path__ = [str(root)]
runner_module = ModuleType("nanovllm.engine.model_runner")
runner_module.ModelRunner = object
with patch.dict(sys.modules, {"nanovllm": package, "nanovllm.engine.model_runner": runner_module}):
    Scheduler = importlib.import_module("nanovllm.engine.scheduler").Scheduler
    sequence_module = importlib.import_module("nanovllm.engine.sequence")
    Sequence, SequenceStatus = sequence_module.Sequence, sequence_module.SequenceStatus
    BlockManager = importlib.import_module("nanovllm.engine.block_manager").BlockManager
    SamplingParams = importlib.import_module("nanovllm.sampling_params").SamplingParams
    LLMEngine = importlib.import_module("nanovllm.engine.llm_engine").LLMEngine


def make_config(**changes):
    values = dict(max_num_seqs=2, max_num_batched_tokens=16, eos=99,
                  kvcache_block_size=4, num_kvcache_blocks=16,
                  is_qwen3_5=True, max_num_delta_states=2, max_model_len=4096)
    values.update(changes)
    return SimpleNamespace(**values)


def make_seq(tokens, max_tokens=8):
    return Sequence(tokens, SamplingParams(max_tokens=max_tokens))


class SchedulerTests(unittest.TestCase):

    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_state_capacity_queues_new_requests_but_allows_decode(self):
        scheduler = Scheduler(make_config(max_num_delta_states=1))
        a, b = make_seq([1, 2], 2), make_seq([3, 4], 1)
        scheduler.add(a)
        scheduler.add(b)
        seqs, prefill = scheduler.schedule()
        self.assertEqual((seqs, prefill), ([a], True))
        self.assertEqual(scheduler.resident_seq_ids, {a.seq_id})
        self.assertEqual(b.block_table, [])
        scheduler.postprocess(seqs, [5], prefill)

        seqs, prefill = scheduler.schedule()
        self.assertEqual((seqs, prefill), ([a], False))
        scheduler.postprocess(seqs, [6], prefill)
        self.assertEqual(scheduler.pop_released_seq_ids(), [a.seq_id])
        self.assertEqual(scheduler.pop_released_seq_ids(), [])
        self.assertEqual(scheduler.resident_seq_ids, set())

        seqs, prefill = scheduler.schedule()
        self.assertEqual((seqs, prefill), ([b], True))
        scheduler.postprocess(seqs, [7], prefill)
        self.assertEqual(scheduler.pop_released_seq_ids(), [b.seq_id])
        self.assertTrue(scheduler.is_finished())

    def test_chunked_prefill_and_unscheduled_requests_keep_residency(self):
        scheduler = Scheduler(make_config(max_num_batched_tokens=2, max_num_seqs=1))
        a, b, c = make_seq([1, 2, 3, 4, 5, 6]), make_seq([7, 8]), make_seq([9])
        for seq in (a, b, c):
            scheduler.add(seq)
        for cached in (2, 4, 6):
            seqs, prefill = scheduler.schedule()
            self.assertEqual((seqs, prefill), ([a], True))
            self.assertEqual(a.num_scheduled_tokens, 2)
            scheduler.postprocess(seqs, [10], prefill)
            self.assertEqual(a.num_cached_tokens, cached)
            self.assertEqual(scheduler.resident_seq_ids, {a.seq_id})
            self.assertEqual(scheduler.pop_released_seq_ids(), [])
        self.assertEqual(a.num_completion_tokens, 1)

        seqs, prefill = scheduler.schedule()
        self.assertEqual((seqs, prefill), ([b], True))
        scheduler.postprocess(seqs, [11], prefill)
        self.assertEqual(scheduler.resident_seq_ids, {a.seq_id, b.seq_id})
        seqs, prefill = scheduler.schedule()
        self.assertEqual((seqs, prefill), ([a], False))
        self.assertEqual(scheduler.resident_seq_ids, {a.seq_id, b.seq_id})
        self.assertEqual(c.block_table, [])

    def test_preemption_releases_state_and_restarts_history(self):
        scheduler = Scheduler(make_config())
        seq = make_seq([1, 2, 3])
        scheduler.add(seq)
        scheduled, prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [4], prefill)
        self.assertIs(scheduler.running.popleft(), seq)
        scheduler.preempt(seq)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertEqual(seq.block_table, [])
        self.assertEqual(scheduler.pop_released_seq_ids(), [seq.seq_id])
        self.assertEqual(scheduler.resident_seq_ids, set())
        scheduled, prefill = scheduler.schedule()
        self.assertEqual((scheduled, prefill), ([seq], True))
        self.assertEqual(seq.num_scheduled_tokens, 4)
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertEqual(scheduler.resident_seq_ids, {seq.seq_id})

    def test_engine_releases_preempted_state_before_run_and_finished_state_after(self):
        Sequence.block_size = 2
        scheduler = Scheduler(make_config(kvcache_block_size=2, num_kvcache_blocks=3))
        a, b = make_seq([1, 2], 2), make_seq([3, 4], 2)
        scheduler.add(a)
        scheduler.add(b)
        calls = []

        class Runner:
            def call(self, method, *args):
                if method == "run":
                    calls.append((method, [seq.seq_id for seq in args[0]]))
                    return [9] * len(args[0])
                calls.append((method, args[0]))

        engine = object.__new__(LLMEngine)
        engine.scheduler, engine.model_runner = scheduler, Runner()
        engine.step()
        calls.clear()
        output, num_tokens = engine.step()
        self.assertEqual(calls, [("release_states", [b.seq_id]),
                                 ("run", [a.seq_id]),
                                 ("release_states", [a.seq_id])])
        self.assertEqual(output, [(a.seq_id, [9, 9])])
        self.assertEqual(num_tokens, -1)
        self.assertEqual(b.num_cached_tokens, 0)
        self.assertEqual(b.block_table, [])
        self.assertEqual(scheduler.resident_seq_ids, set())
        self.assertEqual(list(scheduler.waiting), [b])

    def test_oversized_history_reports_cache_exhaustion(self):
        scheduler = Scheduler(make_config(num_kvcache_blocks=1))
        seq = make_seq([1, 2, 3, 4])
        scheduler.add(seq)
        scheduled, prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [5], prefill)
        engine = object.__new__(LLMEngine)
        engine.scheduler, engine.model_runner = scheduler, Mock()
        with self.assertRaisesRegex(RuntimeError, "fits in the KV cache"):
            engine.step()
        engine.model_runner.call.assert_called_once_with("release_states", [seq.seq_id])
        self.assertEqual(scheduler.pop_released_seq_ids(), [])
        self.assertEqual(scheduler.resident_seq_ids, set())

    def test_decode_respects_batched_token_limit(self):
        scheduler = Scheduler(make_config(max_num_batched_tokens=1))
        a, b = make_seq([1]), make_seq([2])
        for seq in (a, b):
            scheduler.add(seq)
            scheduled, prefill = scheduler.schedule()
            self.assertEqual((scheduled, prefill), ([seq], True))
            scheduler.postprocess(scheduled, [3], prefill)
        self.assertEqual(scheduler.schedule(), ([a], False))
        self.assertEqual(scheduler.resident_seq_ids, {a.seq_id, b.seq_id})

    def test_prompt_and_completion_fit_context_limit(self):
        scheduler = Scheduler(make_config(max_model_len=4))
        with self.assertRaisesRegex(ValueError, "max_model_len"):
            scheduler.add(make_seq([1, 2, 3, 4]))
        seq = make_seq([1, 2, 3])
        scheduler.add(seq)
        scheduled, prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [4], prefill)
        self.assertTrue(seq.is_finished)
        self.assertEqual(scheduler.pop_released_seq_ids(), [seq.seq_id])

    def test_hybrid_prefix_reuse_is_disabled_and_qwen3_reuse_is_preserved(self):
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid):
                scheduler = Scheduler(make_config(is_qwen3_5=hybrid))
                a = make_seq(list(range(8)), 1)
                scheduler.add(a)
                scheduled, prefill = scheduler.schedule()
                scheduler.postprocess(scheduled, [9], prefill)
                b = make_seq(list(range(8)), 1)
                scheduler.add(b)
                scheduled, prefill = scheduler.schedule()
                self.assertEqual((scheduled, prefill), ([b], True))
                self.assertEqual(b.num_cached_tokens, 0 if hybrid else 4)
                self.assertEqual(b.num_scheduled_tokens, 8 if hybrid else 4)
                if hybrid:
                    self.assertEqual(scheduler.block_manager.hash_to_block_id, {})
                else:
                    self.assertEqual(scheduler.pop_released_seq_ids(), [])

    def test_disabled_prefix_mode_checks_physical_block_capacity(self):
        blocks = BlockManager(2, 4, enable_prefix_cache=False)
        a, b = make_seq([1, 2, 3, 4]), make_seq(list(range(8)))
        self.assertEqual(blocks.can_allocate(a), 0)
        blocks.allocate(a, 0)
        self.assertEqual(blocks.can_allocate(b), -1)
        blocks.deallocate(a)
        self.assertEqual(blocks.can_allocate(b), 0)


if __name__ == "__main__":
    unittest.main()
