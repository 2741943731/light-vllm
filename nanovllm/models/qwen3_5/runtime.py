from __future__ import annotations

import torch

from .model import Qwen3_5ForCausalLM, validate_cu_seqlens
from .state import GatedDeltaNetStatePool


class Qwen3_5Runtime:
    """Adapt packed engine batches to the text model's explicit recurrent state.

    The runner sets the full-attention Context before calling forward. Sequence
    IDs, token boundaries and KV block tables must describe the same batch.
    The scheduler owns admission and sends release events on completion or
    preemption; a sequence absent from the current batch still owns its slot.
    """

    def __init__(self, model: Qwen3_5ForCausalLM, capacity: int):
        self.model = model
        self.pool = GatedDeltaNetStatePool(model.model, capacity)

    def release(self, seq_ids: list[int]):
        for seq_id in seq_ids:
            # A request may be preempted before its first model call.
            if seq_id in self.pool:
                self.pool.release(seq_id)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: list[int],
        cu_seqlens: list[int],
        num_cached_tokens: list[int],
        warmup: bool = False,
    ) -> torch.Tensor:
        validate_cu_seqlens(cu_seqlens, input_ids.numel())
        num_seqs = len(cu_seqlens) - 1
        if len(seq_ids) != num_seqs or len(set(seq_ids)) != num_seqs or len(num_cached_tokens) != num_seqs:
            raise ValueError("Sequence IDs, cached lengths and token boundaries must describe the same batch")
        if any(n < 0 for n in num_cached_tokens):
            raise ValueError("Cached token counts must be nonnegative")
        if warmup:
            # Profile temporary input/output states without admitting fake IDs.
            states = self.model.init_state(num_seqs)
        else:
            new_ids = []
            for seq_id, cached in zip(seq_ids, num_cached_tokens):
                if seq_id not in self.pool:
                    if cached:
                        raise ValueError("Missing DeltaNet state for a cached sequence; replay its full history")
                    new_ids.append(seq_id)
                elif cached == 0:
                    raise ValueError("Release the previous DeltaNet state before restarting a sequence")
            if len(new_ids) > self.pool.num_free_slots:
                raise RuntimeError("DeltaNet state pool is full; the scheduler must limit resident requests")
            for seq_id in new_ids:
                self.pool.allocate(seq_id)
            states = self.pool.gather(seq_ids)

        hidden_states, new_states = self.model(input_ids, positions, cu_seqlens, states)
        logits = self.model.compute_logits(hidden_states, cu_seqlens)
        if not warmup:
            self.pool.scatter(seq_ids, new_states)
        return logits
