from __future__ import annotations

from collections.abc import Callable
import torch
import torch.nn.functional as F
from torch import nn

from .attention import Qwen3_5Attention
from .layernorm import Qwen3_5RMSNorm
from .linear_attention import GatedDeltaNetState, Qwen3_5GatedDeltaNet


def validate_cu_seqlens(cu_seqlens: list[int], num_tokens: int):
    if (not isinstance(cu_seqlens, list) or len(cu_seqlens) < 2
            or any(not isinstance(x, int) for x in cu_seqlens)
            or cu_seqlens[0] != 0 or cu_seqlens[-1] != num_tokens
            or any(a >= b for a, b in zip(cu_seqlens, cu_seqlens[1:]))):
        raise ValueError("cu_seqlens must be a strictly increasing CPU list from 0 to num_tokens")


class Qwen3_5MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError("Qwen3.5 MLP requires hidden_act='silu'")
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
        attention_backend: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel_size=config.linear_conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
            )
        elif self.block_type == "full_attention":
            rope_parameters = config.rope_parameters
            if rope_parameters.get("rope_type", "default") != "default":
                raise ValueError("Only default text RoPE is supported")
            if not getattr(config, "attn_output_gate", True):
                raise ValueError("Qwen3.5 attention requires attn_output_gate=True")
            self.self_attn = Qwen3_5Attention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                qkv_bias=getattr(config, "attention_bias", False),
                rope_theta=rope_parameters["rope_theta"],
                partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),
                attention_backend=attention_backend,
            )
        else:
            raise ValueError(f"Unsupported layer type: {self.block_type}")
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3_5MLP(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        cu_seqlens: list[int],
        state: GatedDeltaNetState | None = None,
    ) -> tuple[torch.Tensor, GatedDeltaNetState | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states, state = self.linear_attn(hidden_states, cu_seqlens, state)
        else:
            if state is not None:
                raise ValueError("Full attention uses backend KV cache, not GatedDeltaNetState")
            hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, state


class Qwen3_5Model(nn.Module):
    """Text decoder with explicit DeltaNet states, keyed by decoder layer index.

    Full-attention KV caches live in the individual backends. The caller must
    configure those backends (the engine Context by default) for the same packed
    batch described by cu_seqlens, and keep both kinds of state between calls.
    states=None starts fresh DeltaNet memory; full-attention caches must also be
    fresh for those requests. Always pass the returned states on continuation.
    """

    def __init__(
        self,
        config,
        attention_backend_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        config = getattr(config, "text_config", config)
        self.config = config
        if config.num_hidden_layers <= 0 or len(config.layer_types) != config.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per decoder layer")
        if any(t not in ("linear_attention", "full_attention") for t in config.layer_types):
            raise ValueError("layer_types must be linear_attention or full_attention")
        if config.hidden_act != "silu":
            raise ValueError("Qwen3.5 requires hidden_act='silu'")
        self.linear_layer_ids = tuple(i for i, t in enumerate(config.layer_types) if t == "linear_attention")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, getattr(config, "pad_token_id", None))
        layers = []
        for i, layer_type in enumerate(config.layer_types):
            backend = None
            if layer_type == "full_attention" and attention_backend_factory is not None:
                backend = attention_backend_factory(i)
            layers.append(Qwen3_5DecoderLayer(config, i, backend))
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def init_state(self, num_seqs: int) -> dict[int, GatedDeltaNetState]:
        return {i: self.layers[i].linear_attn.init_state(num_seqs) for i in self.linear_layer_ids}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: list[int] | None = None,
        states: dict[int, GatedDeltaNetState] | None = None,
    ) -> tuple[torch.Tensor, dict[int, GatedDeltaNetState]]:
        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty packed token vector")
        if positions.shape != input_ids.shape or positions.device != input_ids.device:
            raise ValueError("positions must match the shape and device of input_ids")
        if input_ids.dtype not in (torch.int32, torch.int64) or positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids and positions must contain integers")
        if cu_seqlens is None:
            cu_seqlens = [0, input_ids.numel()]
        validate_cu_seqlens(cu_seqlens, input_ids.numel())
        if states is None:
            states = self.init_state(len(cu_seqlens) - 1)
        if not isinstance(states, dict) or set(states) != set(self.linear_layer_ids):
            raise ValueError("states must contain exactly the linear-attention layer indices")
        if any(not isinstance(state, GatedDeltaNetState) for state in states.values()):
            raise ValueError("Each linear-attention layer requires a GatedDeltaNetState")

        hidden_states = self.embed_tokens(input_ids)
        new_states = {}
        for i, layer in enumerate(self.layers):
            hidden_states, state = layer(positions, hidden_states, cu_seqlens, states.get(i))
            if state is not None:
                new_states[i] = state
        return self.norm(hidden_states), new_states


class Qwen3_5ForCausalLM(nn.Module):
    """Single-device text model; forward returns hidden states and updated memory."""

    def __init__(
        self,
        config,
        attention_backend_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        config = getattr(config, "text_config", config)
        self.model = Qwen3_5Model(config, attention_backend_factory)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def init_state(self, num_seqs: int) -> dict[int, GatedDeltaNetState]:
        return self.model.init_state(num_seqs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: list[int] | None = None,
        states: dict[int, GatedDeltaNetState] | None = None,
    ) -> tuple[torch.Tensor, dict[int, GatedDeltaNetState]]:
        return self.model(input_ids, positions, cu_seqlens, states)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.size(1) != self.model.config.hidden_size:
            raise ValueError("hidden_states must have shape [num_tokens, hidden_size]")
        if cu_seqlens is not None:
            validate_cu_seqlens(cu_seqlens, hidden_states.size(0))
            last_indices = torch.tensor([end - 1 for end in cu_seqlens[1:]], device=hidden_states.device)
            hidden_states = hidden_states[last_indices]
        return self.lm_head(hidden_states)
