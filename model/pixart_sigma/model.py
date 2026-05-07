from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PixArtSigmaConfig:
    latent_channels: int = 4
    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    caption_channels: int = 4096
    cross_attention_dim: int = 1152
    max_text_tokens: int = 300


class PixArtSigmaRFModel(nn.Module):
    """Small project-native PixArt-Sigma-style RF transformer."""

    def __init__(self, cfg: PixArtSigmaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        patch_dim = int(cfg.latent_channels) * int(cfg.patch_size) * int(cfg.patch_size)
        hidden = int(cfg.hidden_size)
        self.patch = nn.Conv2d(
            int(cfg.latent_channels),
            hidden,
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
        )
        self.text_proj = nn.Linear(int(cfg.caption_channels), hidden)
        self.time = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg.num_heads),
            dim_feedforward=max(hidden, int(hidden * float(cfg.mlp_ratio))),
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=int(cfg.depth))
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, patch_dim)

    def forward(self, *, x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        p = int(self.cfg.patch_size)
        patches = self.patch(x).flatten(2).transpose(1, 2)
        text_tokens = self.text_proj(text[:, : int(self.cfg.max_text_tokens)])
        time = self.time(t.to(device=x.device, dtype=x.dtype).view(b, 1)).unsqueeze(1)
        seq = torch.cat([patches + time, text_tokens], dim=1)
        seq = self.blocks(seq)
        image_tokens = self.norm(seq[:, : patches.shape[1]])
        out = self.out(image_tokens)
        hp = h // p
        wp = w // p
        out = out.view(b, hp, wp, int(self.cfg.latent_channels), p, p)
        return out.permute(0, 3, 1, 4, 2, 5).reshape(b, int(self.cfg.latent_channels), h, w)


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
    )
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
