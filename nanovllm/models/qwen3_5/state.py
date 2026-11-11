from __future__ import annotations

import torch

from .linear_attention import GatedDeltaNetState
from .model import Qwen3_5Model


class GatedDeltaNetStatePool:
    """Keep each live sequence's linear-attention state in a fixed slot.

    Create the pool after moving the model to its inference device and dtype.
    The caller owns the request lifecycle and the full-attention KV cache.
    Run pool operations and the corresponding model forward in order on the
    same device stream. Pass the same sequence order to gather and scatter.
    """

    def __init__(self, model: Qwen3_5Model, capacity: int) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("State pool capacity must be a positive integer")
        if not model.linear_layer_ids:
            raise ValueError("State pool requires at least one linear-attention layer")
        self.capacity = capacity
        self._device = model.embed_tokens.weight.device
        self._states = model.init_state(capacity)
        self._slots: dict[int, int] = {}
        self._free_slots = list(range(capacity - 1, -1, -1))

    @staticmethod
    def bytes_per_sequence(model: Qwen3_5Model) -> int:
        """Size of one persistent slot across all linear-attention layers."""
        total = 0
        for layer_id in model.linear_layer_ids:
            attn = model.layers[layer_id].linear_attn
            conv_bytes = attn.conv_dim * attn.conv_kernel_size * attn.in_proj_qkv.weight.element_size()
            recurrent_bytes = attn.num_v_heads * attn.head_k_dim * attn.head_v_dim * torch.float32.itemsize
            total += conv_bytes + recurrent_bytes
        return total

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def __contains__(self, seq_id: int) -> bool:
        return seq_id in self._slots

    @torch.no_grad()
    def allocate(self, seq_id: int) -> int:
        if seq_id in self._slots:
            raise ValueError(f"Sequence {seq_id} already has a state slot")
        if not self._free_slots:
            raise RuntimeError("DeltaNet state pool is full")
        slot = self._free_slots[-1]
        for state in self._states.values():
            state.conv_state[slot].zero_()
            state.recurrent_state[slot].zero_()
        self._slots[seq_id] = self._free_slots.pop()
        return slot

    def release(self, seq_id: int) -> None:
        self._free_slots.append(self._slots.pop(seq_id))

    def _batch_indices(self, seq_ids: list[int]) -> torch.Tensor:
        if not seq_ids or len(set(seq_ids)) != len(seq_ids):
            raise ValueError("Expected a nonempty batch of distinct sequence IDs")
        slots = [self._slots[seq_id] for seq_id in seq_ids]
        return torch.tensor(slots, dtype=torch.long, device=self._device)

    @torch.no_grad()
    def gather(self, seq_ids: list[int]) -> dict[int, GatedDeltaNetState]:
        """Copy the requested slots into a batch ordered by seq_ids."""
        indices = self._batch_indices(seq_ids)
        return {
            layer_id: GatedDeltaNetState(
                state.conv_state.index_select(0, indices),
                state.recurrent_state.index_select(0, indices),
            )
            for layer_id, state in self._states.items()
        }

    @torch.no_grad()
    def scatter(self, seq_ids: list[int], states: dict[int, GatedDeltaNetState]) -> None:
        """Write each returned batch row back to its sequence's slot."""
        indices = self._batch_indices(seq_ids)
        if not isinstance(states, dict) or states.keys() != self._states.keys():
            raise ValueError("State layer indices do not match the pool")
        # Check the entire batch before changing any layer's buffers.
        for layer_id, target in self._states.items():
            source = states[layer_id]
            if not isinstance(source, GatedDeltaNetState):
                raise ValueError(f"Expected GatedDeltaNetState for layer {layer_id}")
            for name, value, buffer in zip(GatedDeltaNetState._fields, source, target):
                shape = (len(seq_ids), *buffer.shape[1:])
                if not isinstance(value, torch.Tensor) or value.shape != shape:
                    raise ValueError(f"Expected {name} shape {shape} for layer {layer_id}")
                if value.dtype != buffer.dtype or value.device != buffer.device:
                    raise ValueError(f"{name} dtype or device does not match the pool for layer {layer_id}")

        for layer_id, target in self._states.items():
            source = states[layer_id]
            target.conv_state.index_copy_(0, indices, source.conv_state)
            target.recurrent_state.index_copy_(0, indices, source.recurrent_state)
