from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import _gn_groups


def _zero_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        nn.init.zeros_(param)
    return module


class SelfAttention2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        heads: int,
        head_dim: int,
        *,
        attention_type: str = "global",
        window_size: int = 8,
        hybrid_window_threshold: int = 0,
        zero_init: bool = False,
    ):
        super().__init__()
        if attention_type not in {"global", "windowed", "hybrid"}:
            raise ValueError("attention_type must be one of: global, windowed, hybrid.")
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner = inner
        self.attention_type = attention_type
        self.window_size = int(window_size)
        self.hybrid_window_threshold = int(hybrid_window_threshold)

        self.norm = nn.GroupNorm(_gn_groups(in_ch), in_ch, eps=1e-6)
        self.to_qkv = nn.Conv2d(in_ch, 3 * inner, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(inner, in_ch, kernel_size=1, bias=True)
        if zero_init:
            _zero_module(self.to_out)

    def _global_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    def _windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        if self.window_size <= 0:
            raise RuntimeError("window_size must be positive for windowed attention.")
        if self.window_size >= h and self.window_size >= w:
            return self._global_attention(q, k, v)

        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws

        def to_windows(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
            b, heads, hw, d = x.shape
            x = x.transpose(2, 3).contiguous().view(b, heads, d, h, w)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))
            hp, wp = h + pad_h, w + pad_w
            x = x.view(b, heads, d, hp // ws, ws, wp // ws, ws)
            x = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
            return x.view(b * (hp // ws) * (wp // ws), heads, ws * ws, d), hp, wp

        q_win, hp, wp = to_windows(q)
        k_win, _, _ = to_windows(k)
        v_win, _, _ = to_windows(v)
        attn_mask = None
        if pad_h or pad_w:
            valid = torch.ones((1, 1, 1, h, w), dtype=torch.bool, device=q.device)
            valid = F.pad(valid, (0, pad_w, 0, pad_h), value=False)
            valid = valid.view(1, 1, 1, hp // ws, ws, wp // ws, ws)
            valid = valid.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
            valid = valid.view((hp // ws) * (wp // ws), ws * ws)
            valid = valid.repeat(q.shape[0], 1)
            attn_mask = valid.unsqueeze(1).unsqueeze(1)
        out = self._global_attention(q_win, k_win, v_win, attn_mask=attn_mask)

        b = q.shape[0]
        out = out.view(b, hp // ws, wp // ws, self.heads, ws, ws, self.head_dim)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        out = out.view(b, self.heads, self.head_dim, hp, wp)
        out = out[..., :h, :w].contiguous().view(b, self.heads, self.head_dim, h * w)
        return out.transpose(2, 3)

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

        attention_type = self.attention_type
        if attention_type == "hybrid":
            threshold = self.hybrid_window_threshold
            if threshold <= 0:
                threshold = max(h, w)
            attention_type = "windowed" if max(h, w) >= threshold else "global"

        if attention_type == "windowed":
            out = self._windowed_attention(q, k, v, h, w)
        else:
            out = self._global_attention(q, k, v)
        out = out.transpose(2, 3).contiguous().view(b, self.inner, h, w)
        out = self.to_out(out)
        return x_in + out


class CrossAttention2d(nn.Module):
    def __init__(self, in_ch: int, ctx_dim: int, heads: int, head_dim: int, *, zero_init: bool = False):
        super().__init__()
        inner = heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner = inner

        self.norm = nn.GroupNorm(_gn_groups(in_ch), in_ch, eps=1e-6)

        self.to_q = nn.Conv2d(in_ch, inner, kernel_size=1, bias=False)
        self.to_kv = nn.Linear(ctx_dim, 2 * inner, bias=False)
        self.to_out = nn.Conv2d(inner, in_ch, kernel_size=1, bias=True)
        if zero_init:
            _zero_module(self.to_out)

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

        # Boolean mask for SDPA: True = token can participate.
        attn_mask = ctx_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,T]

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
