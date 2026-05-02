from __future__ import annotations

import math

import torch


def get_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        return torch.zeros((height * width, dim), device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    return pos.to(dtype=dtype)


def add_2d_pos_embed(tokens: torch.Tensor, grid_hw: tuple[int, int], mode: str) -> torch.Tensor:
    if mode == "none":
        return tokens
    if mode == "rope_2d":
        raise NotImplementedError(
            "rope_2d must be applied to q/k inside attention, not added to tokens."
        )
    if mode != "sincos_2d":
        raise ValueError(f"Unsupported positional embedding mode: {mode}")
    h, w = grid_hw
    pos = get_2d_sincos_pos_embed(h, w, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
    return tokens + pos.unsqueeze(0)
