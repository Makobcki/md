from __future__ import annotations

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_dim: int = 1024, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(
            int(in_channels),
            int(hidden_dim),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("PatchEmbed input must have shape [B,C,H,W].")
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"height and width must be divisible by patch_size={self.patch_size}; "
                f"got {tuple(x.shape[-2:])}."
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2).contiguous()


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    p = int(patch_size)
    if h % p != 0 or w % p != 0:
        raise ValueError("height and width must be divisible by patch_size.")
    x = x.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def unpatchify(tokens: torch.Tensor, *, channels: int, height: int, width: int, patch_size: int) -> torch.Tensor:
    b, n, d = tokens.shape
    p = int(patch_size)
    gh = height // p
    gw = width // p
    expected_d = int(channels) * p * p
    if height % p != 0 or width % p != 0:
        raise ValueError("height and width must be divisible by patch_size.")
    if n != gh * gw:
        raise ValueError(f"token count mismatch: got {n}, expected {gh * gw}.")
    if d != expected_d:
        raise ValueError(f"token dim mismatch: got {d}, expected {expected_d}.")
    x = tokens.reshape(b, gh, gw, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.reshape(b, channels, height, width)

