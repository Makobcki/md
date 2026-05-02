from __future__ import annotations

import math

import torch


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    sinusoidal embedding: [B] -> [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / float(half)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb
