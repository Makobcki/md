from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        dtype = x.dtype
        y = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return y.to(dtype=dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        y = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y.to(dtype=dtype) * self.weight.to(dtype=dtype))


class AdaLNZero(nn.Module):
    def __init__(self, hidden_dim: int, chunks: int = 6) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.chunks = int(chunks)
        self.linear = nn.Linear(hidden_dim, hidden_dim * chunks)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        parts = self.linear(cond).chunk(self.chunks, dim=-1)
        return parts


def build_norm(hidden_dim: int, *, rms_norm: bool) -> nn.Module:
    if rms_norm:
        return RMSNorm(hidden_dim)
    return FP32LayerNorm(hidden_dim)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

