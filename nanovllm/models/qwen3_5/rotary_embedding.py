import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rot = x_rot * cos + rotated * sin
    return torch.cat((x_rot, x_pass), dim=-1)


class Qwen3_5RotaryEmbedding(nn.Module):
    """Partial RoPE for packed, text-only query and key tensors."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        base: float,
    ) -> None:
        super().__init__()
        if not 0 < rotary_dim <= head_size or rotary_dim % 2:
            raise ValueError("rotary_dim must be even and between 1 and head_size")
        if base <= 0:
            raise ValueError("base must be positive")
        self.rotary_dim = rotary_dim
        self.base = base

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=positions.device) / self.rotary_dim
        ))
        freqs = positions.float().unsqueeze(-1) * inv_freq
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        query = apply_rotary_emb(query, cos.to(query.dtype), sin.to(query.dtype), self.rotary_dim)
        key = apply_rotary_emb(key, cos.to(key.dtype), sin.to(key.dtype), self.rotary_dim)
        return query, key
