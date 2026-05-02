from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom
