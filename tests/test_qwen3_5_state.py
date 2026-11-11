"""State pool correctness using a small real mixed Qwen3.5 model on CPU."""
import importlib
import unittest

import torch

from test_qwen3_5_model import Qwen3_5Model, CPUAttention, make_config, package

state_module = importlib.import_module(package.__name__ + ".state")
GatedDeltaNetStatePool = state_module.GatedDeltaNetStatePool
GatedDeltaNetState = importlib.import_module(package.__name__ + ".linear_attention").GatedDeltaNetState


def make_model():
    backend = CPUAttention(8 ** -0.5)
    model = Qwen3_5Model(make_config(), lambda _: backend).eval()
    return model, backend


def assert_states_close(actual, expected, *, exact=False):
    assert set(actual) == set(expected)
    kwargs = dict(rtol=0, atol=0) if exact else dict(rtol=3e-5, atol=3e-6)
    for layer in actual:
        torch.testing.assert_close(actual[layer].conv_state, expected[layer].conv_state, **kwargs)
        torch.testing.assert_close(actual[layer].recurrent_state, expected[layer].recurrent_state, **kwargs)


class GatedDeltaNetStatePoolTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(233)
        self.model, self.backend = make_model()
        self.pool = GatedDeltaNetStatePool(self.model, 3)

    def test_memory_count_and_state_dtype(self):
        self.model.to(torch.bfloat16)
        pool = GatedDeltaNetStatePool(self.model, 2)
        per_seq = self.model.init_state(1)
        expected = sum(t.numel() * t.element_size() for state in per_seq.values() for t in state)
        self.assertEqual(pool.bytes_per_sequence(self.model), expected)
        pool.allocate(1)
        for state in pool.gather([1]).values():
            self.assertEqual(state.conv_state.dtype, torch.bfloat16)
            self.assertEqual(state.recurrent_state.dtype, torch.float32)
            self.assertEqual(state.conv_state.device.type, "cpu")

    def test_reorder_copy_and_reused_slot_isolation(self):
        self.assertEqual([self.pool.allocate(seq) for seq in (101, 202, 303)], [0, 1, 2])
        states = self.pool.gather([101, 202, 303])
        for layer, state in states.items():
            for row in range(3):
                state.conv_state[row].fill_(10 * layer + row + 1)
                state.recurrent_state[row].fill_(100 * layer + row + 1)
        # Gather must be independent: modifying it cannot change the pool.
        self.assertTrue(all(torch.count_nonzero(t) == 0 for state in self.pool.gather([101, 202, 303]).values() for t in state))
        self.pool.scatter([101, 202, 303], states)
        reordered = self.pool.gather([303, 101, 202])
        for layer in states:
            torch.testing.assert_close(reordered[layer].conv_state, states[layer].conv_state[[2, 0, 1]], rtol=0, atol=0)
            torch.testing.assert_close(reordered[layer].recurrent_state, states[layer].recurrent_state[[2, 0, 1]], rtol=0, atol=0)
        before = self.pool.gather([101, 303])
        self.pool.release(202)
        self.assertNotIn(202, self.pool)
        self.assertEqual(self.pool.num_free_slots, 1)
        self.assertEqual(self.pool.allocate(404), 1)
        self.assertTrue(all(torch.count_nonzero(t) == 0 for state in self.pool.gather([404]).values() for t in state))
        assert_states_close(self.pool.gather([101, 303]), before, exact=True)

    def test_preflight_error_does_not_partially_modify_other_layers(self):
        self.pool.allocate(1)
        self.pool.allocate(2)
        original = self.pool.gather([1, 2])
        for problem in ("missing_layer", "extra_layer", "shape", "conv_dtype", "recurrent_dtype", "device"):
            with self.subTest(problem=problem):
                update = self.pool.gather([1, 2])
                for state in update.values():
                    state.conv_state.fill_(7)
                    state.recurrent_state.fill_(8)
                layer = sorted(update)[-1]
                conv, recurrent = update[layer]
                if problem == "missing_layer":
                    del update[layer]
                elif problem == "extra_layer":
                    update[999] = update[layer]
                elif problem == "shape":
                    update[layer] = GatedDeltaNetState(conv[:1], recurrent)
                elif problem == "conv_dtype":
                    update[layer] = GatedDeltaNetState(conv.to(torch.float64), recurrent)
                elif problem == "recurrent_dtype":
                    update[layer] = GatedDeltaNetState(conv, recurrent.to(torch.bfloat16))
                else:
                    update[layer] = GatedDeltaNetState(conv.to("meta"), recurrent)
                with self.assertRaises(ValueError):
                    self.pool.scatter([1, 2], update)
                assert_states_close(self.pool.gather([1, 2]), original, exact=True)

    def test_lifecycle_and_invalid_batches(self):
        self.pool.allocate(1)
        with self.assertRaises(ValueError):
            self.pool.allocate(1)
        self.pool.allocate(2)
        self.pool.allocate(3)
        with self.assertRaises(RuntimeError):
            self.pool.allocate(4)
        with self.assertRaises(KeyError):
            self.pool.release(99)
        for ids, exc in (([], ValueError), ([1, 1], ValueError), ([1, 99], KeyError)):
            with self.subTest(ids=ids):
                with self.assertRaises(exc):
                    self.pool.gather(ids)
                with self.assertRaises(exc):
                    self.pool.scatter(ids, self.model.init_state(max(1, len(ids))))
        self.pool.release(1)
        with self.assertRaises(KeyError):
            self.pool.release(1)

    @torch.inference_mode()
    def test_mixed_model_reorder_chunk_decode_and_reuse_matches_independent(self):
        # Paged/full attention has its own cache. Reorder its sequence rows at
        # each step in lockstep with the state pool, just as a caller must.
        self.pool.allocate(10)
        self.pool.allocate(20)
        self.backend.bounds = [0, 3, 5]
        first, state = self.model(torch.tensor([1, 2, 3, 4, 5]), torch.tensor([0, 1, 2, 0, 1]),
                                  [0, 3, 5], self.pool.gather([10, 20]))
        self.pool.scatter([10, 20], state)
        outputs = {10: [first[:3]], 20: [first[3:]], 30: []}
        self.backend.cache = {0: self.backend.cache[1], 1: self.backend.cache[0]}
        self.backend.bounds, self.backend.decode = [0, 1, 2], True
        next_hidden, state = self.model(torch.tensor([6, 7]), torch.tensor([2, 3]), [0, 1, 2], self.pool.gather([20, 10]))
        self.pool.scatter([20, 10], state)
        outputs[20].append(next_hidden[:1])
        outputs[10].append(next_hidden[1:])
        final_10 = self.pool.gather([10])
        self.pool.release(10)
        self.pool.allocate(30)
        saved_20_cache = self.backend.cache[0]
        self.backend.cache, self.backend.bounds, self.backend.decode = {}, [0, 2], False
        chunk, state = self.model(torch.tensor([8, 9]), torch.tensor([0, 1]), [0, 2], self.pool.gather([30]))
        self.pool.scatter([30], state)
        outputs[30].append(chunk)
        self.backend.cache = {0: saved_20_cache, 1: self.backend.cache[0]}
        self.backend.bounds, self.backend.decode = [0, 1, 2], True
        final, state = self.model(torch.tensor([10, 11]), torch.tensor([3, 2]), [0, 1, 2], self.pool.gather([20, 30]))
        self.pool.scatter([20, 30], state)
        outputs[20].append(final[:1])
        outputs[30].append(final[1:])
        seqs = {10: [1, 2, 3, 7], 20: [4, 5, 6, 10], 30: [8, 9, 11]}
        for seq_id, tokens in seqs.items():
            reference, _ = make_model()
            reference.load_state_dict(self.model.state_dict())
            expected, expected_state = reference(torch.tensor(tokens), torch.arange(len(tokens)))
            torch.testing.assert_close(torch.cat(outputs[seq_id]), expected, rtol=3e-5, atol=3e-6)
            final_state = final_10 if seq_id == 10 else self.pool.gather([seq_id])
            assert_states_close(final_state, expected_state)


if __name__ == "__main__":
    unittest.main()
