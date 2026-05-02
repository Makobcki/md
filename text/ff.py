from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2, bias=True)
        self.out = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(hidden * F.gelu(gate))
