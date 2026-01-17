from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(c: int) -> int:
    for g in (32, 16, 8, 4, 2):
        if c % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float, use_scale_shift: bool):
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch, eps=1e-6)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.use_scale_shift = use_scale_shift
        self.tproj = nn.Linear(time_dim, out_ch * 2 if use_scale_shift else out_ch)
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        t = self.tproj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1)
        if self.use_scale_shift:
            scale, shift = t.chunk(2, dim=1)
            h = self.norm2(h)
            h = h * (1.0 + scale) + shift
            h = self.act(h)
            h = self.conv2(self.drop(h))
        else:
            h = h + t
            h = self.conv2(self.drop(self.act(self.norm2(h))))
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)
