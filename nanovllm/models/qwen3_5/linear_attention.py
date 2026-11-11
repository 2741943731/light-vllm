from __future__ import annotations

from typing import NamedTuple
import torch
import torch.nn.functional as F
from torch import nn

from .layernorm import Qwen3_5RMSNormGated


class GatedDeltaNetState(NamedTuple):
    conv_state: torch.Tensor       # [num_seqs, conv_dim, kernel_size], raw QKV history
    recurrent_state: torch.Tensor  # [num_seqs, num_value_heads, key_head_dim, value_head_dim], FP32


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan one sequence: Q/K/V are [tokens, heads, head_dim], g/beta [tokens, heads]."""
    orig_dtype = query.dtype
    query, key, value = query.float(), key.float(), value.float()
    query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    query = query / (query.size(-1) ** 0.5)
    state = initial_state.float()
    outputs = []
    for t in range(query.size(0)):
        # Forget, correct the current key's prediction, then read with the query.
        state = state * g[t].float().exp()[:, None, None]
        prediction = (state * key[t].unsqueeze(-1)).sum(dim=-2)
        delta = (value[t] - prediction) * beta[t].float().unsqueeze(-1)
        state = state + key[t].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * query[t].unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outputs).to(orig_dtype), state


class Qwen3_5GatedDeltaNet(nn.Module):
    """Single-device eager baseline with explicit state and packed, unpadded tokens.

    cu_seqlens is a CPU list of sequence boundaries, e.g. [0, 3, 8]. Each
    state row belongs to the corresponding sequence in this call. Callers must
    reorder state rows when reordering sequences. Input state is never modified;
    use the returned state for the next chunk or decode step. Run inference
    under torch.inference_mode() to avoid retaining the token scan's graph.
    """

    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_key_heads, num_value_heads, key_head_dim, value_head_dim, conv_kernel_size) <= 0:
            raise ValueError("hidden size, head counts, head dimensions and kernel size must be positive")
        if num_value_heads % num_key_heads:
            raise ValueError("num_value_heads must be a multiple of num_key_heads")
        self.hidden_size = hidden_size
        self.num_k_heads = num_key_heads
        self.num_v_heads = num_value_heads
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = num_key_heads * key_head_dim
        self.value_dim = num_value_heads * value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = conv_kernel_size

        # Keep checkpoint parameter names and the contiguous Q, K, V layout.
        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, conv_kernel_size, groups=self.conv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_value_heads))
        self.A_log = nn.Parameter(torch.empty(num_value_heads).uniform_(0.01, 16).log())
        self.norm = Qwen3_5RMSNormGated(value_head_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def init_state(self, num_seqs: int) -> GatedDeltaNetState:
        if num_seqs <= 0:
            raise ValueError("num_seqs must be positive")
        weight = self.in_proj_qkv.weight
        conv = weight.new_zeros(num_seqs, self.conv_dim, self.conv_kernel_size)
        recurrent = torch.zeros(
            num_seqs, self.num_v_heads, self.head_k_dim, self.head_v_dim,
            dtype=torch.float32, device=weight.device,
        )
        return GatedDeltaNetState(conv, recurrent)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: list[int] | None = None,
        state: GatedDeltaNetState | None = None,
    ) -> tuple[torch.Tensor, GatedDeltaNetState]:
        if hidden_states.ndim != 2 or hidden_states.size(1) != self.hidden_size:
            raise ValueError("hidden_states must have shape [num_tokens, hidden_size]")
        num_tokens = hidden_states.size(0)
        if cu_seqlens is None:
            cu_seqlens = [0, num_tokens]
        if (not isinstance(cu_seqlens, list) or len(cu_seqlens) < 2
                or any(not isinstance(x, int) for x in cu_seqlens)
                or cu_seqlens[0] != 0 or cu_seqlens[-1] != num_tokens
                or any(a >= b for a, b in zip(cu_seqlens, cu_seqlens[1:]))):
            raise ValueError("cu_seqlens must be a strictly increasing CPU list from 0 to num_tokens")
        num_seqs = len(cu_seqlens) - 1
        if state is None:
            state = self.init_state(num_seqs)
        if state.conv_state.shape != (num_seqs, self.conv_dim, self.conv_kernel_size):
            raise ValueError("conv_state shape does not match the batch or model")
        if state.recurrent_state.shape != (num_seqs, self.num_v_heads, self.head_k_dim, self.head_v_dim):
            raise ValueError("recurrent_state shape does not match the batch or model")
        if state.conv_state.dtype != hidden_states.dtype or state.recurrent_state.dtype != torch.float32:
            raise ValueError("conv_state must match input dtype and recurrent_state must be float32")
        if state.conv_state.device != hidden_states.device or state.recurrent_state.device != hidden_states.device:
            raise ValueError("state and hidden_states must be on the same device")

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(num_tokens, self.num_v_heads, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        outputs, conv_states, recurrent_states = [], [], []
        for i, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
            # Store raw projections, not the convolved/activated QKV.
            history = torch.cat((state.conv_state[i], mixed_qkv[start:end].transpose(0, 1)), dim=-1)
            convolved = self.conv1d(history.unsqueeze(0))[0, :, -(end - start):]
            convolved = F.silu(convolved).transpose(0, 1)
            query, key, value = convolved.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            query = query.reshape(-1, self.num_k_heads, self.head_k_dim)
            key = key.reshape(-1, self.num_k_heads, self.head_k_dim)
            value = value.reshape(-1, self.num_v_heads, self.head_v_dim)
            repeats = self.num_v_heads // self.num_k_heads
            if repeats > 1:
                query = query.repeat_interleave(repeats, dim=1)
                key = key.repeat_interleave(repeats, dim=1)
            output, recurrent = recurrent_gated_delta_rule(
                query, key, value, g[start:end], beta[start:end], state.recurrent_state[i],
            )
            outputs.append(output)
            conv_states.append(history[:, -self.conv_kernel_size:])
            recurrent_states.append(recurrent)

        output = self.norm(torch.cat(outputs, dim=0), z).reshape(num_tokens, self.value_dim)
        output = self.out_proj(output)
        new_state = GatedDeltaNetState(torch.stack(conv_states), torch.stack(recurrent_states))
        return output, new_state
