from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal embedding: [B] -> [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def _gn(ch: int) -> nn.GroupNorm:
    # безопасно для любых ch: groups <= ch и делит ch
    # для типичных ch (64/128/192/256) GroupNorm(32, ch) ок
    groups = 32
    while ch % groups != 0:
        groups //= 2
        if groups <= 1:
            groups = 1
            break
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.tproj = nn.Linear(tdim, out_ch)

        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = SiLU()

    def forward(self, x, t_emb=None):
        h = self.conv1(self.act(self.norm1(x)))

        if t_emb is not None:
            # t_emb: [B, emb_dim] → [B, C, 1, 1]
            h = h + self.tproj(t_emb)[:, :, None, None]

        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class SelfAttention2d(nn.Module):
    """
    Multi-head self-attention over spatial tokens (H*W).
    """
    def __init__(self, channels: int, heads: int = 4, head_dim: int = 32) -> None:
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim

        self.norm = _gn(channels)
        self.to_qkv = nn.Conv2d(channels, inner * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner, channels, 1)

        # стабилизация
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w

        x_in = x
        x = self.norm(x)

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # [B, heads, N, D]
        q = q.view(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)
        k = k.view(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)
        v = v.view(b, self.heads, self.head_dim, n).permute(0, 1, 3, 2)

        # Flash / SDPA (no NxN matrix)
        # dropout_p=0.0 важно для train-стабильности
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            is_causal=False
        )  # [B, heads, N, D]

        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(b, self.heads * self.head_dim, h, w)

        return x_in + self.to_out(out)


class UNet(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        grad_checkpoint: bool = False,
        attn_resolutions: Tuple[int, ...] = (48,),
        attn_heads: int = 4,
        attn_head_dim: int = 32,
    ) -> None:
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.attn_resolutions = set(attn_resolutions)

        tdim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, tdim),
            SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.in_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        # --- Down ---
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.down_attn = nn.ModuleList()

        ch = base_channels
        levels = []
        for mult in channel_mults:
            levels.append(base_channels * mult)

        for li, out_ch in enumerate(levels):
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, tdim, dropout))
                self.down_attn.append(SelfAttention2d(out_ch, attn_heads, attn_head_dim))
                ch = out_ch
            if li != len(levels) - 1:
                self.downsamples.append(Downsample(ch))

        # --- Middle ---
        self.mid1 = ResBlock(ch, ch, tdim, dropout)
        self.mid_attn = SelfAttention2d(ch, attn_heads, attn_head_dim)
        self.mid2 = ResBlock(ch, ch, tdim, dropout)

        # --- Up ---
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()

        for li, out_ch in enumerate(reversed(levels)):
            if li != 0:
                self.upsamples.append(Upsample(ch))
            for _ in range(num_res_blocks):
                # skip всегда имеет out_ch каналов на этом уровне
                self.up_blocks.append(ResBlock(ch + out_ch, out_ch, tdim, dropout))
                self.up_attn.append(SelfAttention2d(out_ch, attn_heads, attn_head_dim))
                ch = out_ch

        self.out_norm = _gn(ch)
        self.out_conv = nn.Conv2d(ch, image_channels, 3, padding=1)
        self.act = SiLU()

        # Для корректного управления порядком samples
        self._levels = levels
        self._num_res_blocks = num_res_blocks

    def _maybe_ckpt(self, module: nn.Module, *args):
        if self.grad_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # time emb
        t_emb = timestep_embedding(t, self.in_conv.out_channels)
        t_emb = self.time_mlp(t_emb)

        h = self.in_conv(x)

        # хранить skip ТОЛЬКО после ResBlock (это критично)
        skips: List[torch.Tensor] = []

        # --- Down ---
        dbi = 0
        dsi = 0
        for li, out_ch in enumerate(self._levels):
            for _ in range(self._num_res_blocks):
                h = self._maybe_ckpt(self.down_blocks[dbi], h, t_emb)

                # attention только если текущий spatial есть в списке
                hh, ww = h.shape[-2], h.shape[-1]
                if hh in self.attn_resolutions and ww in self.attn_resolutions:
                    h = self._maybe_ckpt(self.down_attn[dbi], h)

                skips.append(h)  # только тут
                dbi += 1

            if li != len(self._levels) - 1:
                h = self.downsamples[dsi](h)
                dsi += 1

        # --- Middle ---
        h = self._maybe_ckpt(self.mid1, h, t_emb)
        hh, ww = h.shape[-2], h.shape[-1]
        if hh in self.attn_resolutions and ww in self.attn_resolutions:
            h = self._maybe_ckpt(self.mid_attn, h)
        h = self._maybe_ckpt(self.mid2, h, t_emb)

        # --- Up ---
        ubi = 0
        usi = 0
        for li, out_ch in enumerate(reversed(self._levels)):
            if li != 0:
                h = self.upsamples[usi](h)
                usi += 1

            for _ in range(self._num_res_blocks):
                skip = skips.pop()  # ✅ порядок строго обратный
                # На всякий пожарный: привести spatial (может быть 1px разница из-за stride/ceil)
                if skip.shape[-2:] != h.shape[-2:]:
                    skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")

                h = torch.cat([h, skip], dim=1)
                h = self._maybe_ckpt(self.up_blocks[ubi], h, t_emb)

                hh, ww = h.shape[-2], h.shape[-1]
                if hh in self.attn_resolutions and ww in self.attn_resolutions:
                    h = self._maybe_ckpt(self.up_attn[ubi], h)

                ubi += 1

        return self.out_conv(self.act(self.out_norm(h)))
