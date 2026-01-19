from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import _gn_groups


class SelfAttention2d(nn.Module):
    def __init__(self, in_ch: int, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner = inner

        self.norm = nn.GroupNorm(_gn_groups(in_ch), in_ch, eps=1e-6)
        self.to_qkv = nn.Conv2d(in_ch, 3 * inner, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner, in_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        hw = h * w
        q = q.view(b, self.heads, self.head_dim, hw).transpose(2, 3)  # [B, heads, HW, d]
        k = k.view(b, self.heads, self.head_dim, hw).transpose(2, 3)
        v = v.view(b, self.heads, self.head_dim, hw).transpose(2, 3)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).contiguous().view(b, self.inner, h, w)
        out = self.to_out(out)
        return x_in + out


class CrossAttention2d(nn.Module):
    def __init__(self, in_ch: int, ctx_dim: int, heads: int, head_dim: int):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner = inner

        self.norm = nn.GroupNorm(_gn_groups(in_ch), in_ch, eps=1e-6)

        self.to_q = nn.Conv2d(in_ch, inner, kernel_size=1, bias=False)
        self.to_kv = nn.Linear(ctx_dim, 2 * inner, bias=False)
        self.to_out = nn.Conv2d(inner, in_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        q = self.to_q(x)  # [B, inner, H, W]
        hw = h * w
        q = q.view(b, self.heads, self.head_dim, hw).transpose(2, 3)  # [B, heads, HW, d]

        kv = self.to_kv(ctx)  # [B, T, 2*inner]
        k, v = kv.chunk(2, dim=-1)
        k = k.view(b, -1, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, T, d]
        v = v.view(b, -1, self.heads, self.head_dim).transpose(1, 2)

        # boolean mask for SDPA: True = mask
        attn_mask = (~ctx_mask).unsqueeze(1).unsqueeze(1)  # [B,1,1,T]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(2, 3).contiguous().view(b, self.inner, h, w)
        out = self.to_out(out)
        return x_in + out


class NoOpCrossAttention(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return x
