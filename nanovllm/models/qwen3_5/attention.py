from __future__ import annotations

import torch
from torch import nn

from .layernorm import Qwen3_5RMSNorm
from .rotary_embedding import Qwen3_5RotaryEmbedding


class Qwen3_5Attention(nn.Module):
    """Single-device gated attention over packed tokens.

    The default backend uses the engine's Context and paged KV cache. An
    alternative backend may be supplied for numerical checks on the CPU.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000000,
        partial_rotary_factor: float = 0.25,
        attention_backend: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if min(hidden_size, num_heads, num_kv_heads, head_dim) <= 0:
            raise ValueError("hidden size, head counts and head dimension must be positive")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if not 0 < partial_rotary_factor <= 1:
            raise ValueError("partial_rotary_factor must be between 0 and 1")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5

        # Each query head occupies [query, gate] in the checkpoint projection.
        self.q_proj = nn.Linear(hidden_size, self.q_size * 2, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=qkv_bias)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=qkv_bias)
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=int(head_dim * partial_rotary_factor),
            base=rope_theta,
        )
        if attention_backend is None:
            from nanovllm.layers.attention import Attention
            attention_backend = Attention(num_heads, head_dim, self.scaling, num_kv_heads)
        self.attn = attention_backend

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.size(1) != self.hidden_size:
            raise ValueError("hidden_states must have shape [num_tokens, hidden_size]")
        num_tokens = hidden_states.size(0)
        if positions.ndim != 1 or positions.size(0) != num_tokens:
            raise ValueError("positions must have shape [num_tokens]")
        if positions.device != hidden_states.device:
            raise ValueError("positions and hidden_states must be on the same device")
        q_gate = self.q_proj(hidden_states).reshape(num_tokens, self.num_heads, self.head_dim * 2)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(hidden_states).reshape(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(num_tokens, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        # The backend stores the normalized, rotated K and the unmodified V.
        output = self.attn(q, k, v).reshape(num_tokens, self.q_size)
        output = output * gate.reshape(num_tokens, self.q_size).sigmoid()
        return self.o_proj(output)
