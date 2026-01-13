from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(c: int) -> int:
    for g in (32, 16, 8, 4, 2):
        if c % g == 0:
            return g
    return 1


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


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask_f.unsqueeze(-1)).sum(dim=1) / denom


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class GEGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2, bias=True)
        self.out = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(hidden * F.gelu(gate))


class SDPATransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim must be divisible by n_heads (dim={dim}, n_heads={n_heads}).")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = float(dropout)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.ff = GEGLUFeedForward(dim, hidden_dim=dim * 4)
        self.drop = nn.Dropout(self.dropout)
        self.attn_gate = nn.Parameter(torch.ones(1))
        self.ff_gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.norm1(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        key_padding = ~attn_mask
        sdpa_mask = key_padding.unsqueeze(1).unsqueeze(2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        x = x + self.drop(self.to_out(attn_out)) * self.attn_gate

        h = self.norm2(x)
        x = x + self.drop(self.ff(h)) * self.ff_gate
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, max_len: int):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [SDPATransformerBlock(dim=dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)

    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        pos = torch.arange(0, t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.norm(x)


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


@dataclass(frozen=True)
class UNetConfig:
    image_channels: int = 3
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    dropout: float = 0.1
    attn_resolutions: Tuple[int, ...] = (32, 16)
    attn_heads: int = 4
    attn_head_dim: int = 32
    vocab_size: int = 50_000
    text_dim: int = 256
    text_layers: int = 4
    text_heads: int = 4
    text_max_len: int = 64
    use_text_conditioning: bool = True
    self_conditioning: bool = False
    use_scale_shift_norm: bool = False
    grad_checkpointing: bool = False


class UNet(nn.Module):
    """
    UNet predicts v (v-prediction) with text conditioning (cross-attention).
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.use_text_conditioning = bool(cfg.use_text_conditioning)
        self.self_conditioning = bool(cfg.self_conditioning)

        time_dim = cfg.base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.text_enc: Optional[TextEncoder] = None
        self.text_pool_proj: Optional[nn.Linear] = None
        if self.use_text_conditioning:
            self.text_enc = TextEncoder(
                vocab_size=cfg.vocab_size,
                dim=cfg.text_dim,
                n_layers=cfg.text_layers,
                n_heads=cfg.text_heads,
                max_len=cfg.text_max_len,
            )
            self.text_pool_proj = nn.Linear(cfg.text_dim, time_dim)

        chs = [cfg.base_channels * m for m in cfg.channel_mults]
        in_channels = cfg.image_channels * 2 if self.self_conditioning else cfg.image_channels
        self.in_conv = nn.Conv2d(in_channels, chs[0], kernel_size=3, padding=1)

        # Down
        self.down_blocks = nn.ModuleList()
        self.down_sa = nn.ModuleList()
        self.down_ca = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        cur = chs[0]
        for level, ch in enumerate(chs):
            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(
                    ResBlock(cur, ch, time_dim=time_dim, dropout=cfg.dropout, use_scale_shift=cfg.use_scale_shift_norm)
                )
                cur = ch
                self.down_sa.append(SelfAttention2d(cur, cfg.attn_heads, cfg.attn_head_dim))
                if self.use_text_conditioning:
                    self.down_ca.append(CrossAttention2d(cur, cfg.text_dim, cfg.attn_heads, cfg.attn_head_dim))
                else:
                    self.down_ca.append(NoOpCrossAttention())
            if level != len(chs) - 1:
                self.downsamples.append(Downsample(cur))

        # Mid
        self.mid1 = ResBlock(cur, cur, time_dim=time_dim, dropout=cfg.dropout, use_scale_shift=cfg.use_scale_shift_norm)
        self.mid_sa = SelfAttention2d(cur, cfg.attn_heads, cfg.attn_head_dim)
        self.mid_ca = (
            CrossAttention2d(cur, cfg.text_dim, cfg.attn_heads, cfg.attn_head_dim)
            if self.use_text_conditioning
            else NoOpCrossAttention()
        )
        self.mid2 = ResBlock(cur, cur, time_dim=time_dim, dropout=cfg.dropout, use_scale_shift=cfg.use_scale_shift_norm)

        # Up
        self.up_blocks = nn.ModuleList()
        self.up_sa = nn.ModuleList()
        self.up_ca = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(len(chs))):
            ch = chs[level]
            for _ in range(cfg.num_res_blocks):
                self.up_blocks.append(
                    ResBlock(cur + ch, ch, time_dim=time_dim, dropout=cfg.dropout, use_scale_shift=cfg.use_scale_shift_norm)
                )
                cur = ch
                self.up_sa.append(SelfAttention2d(cur, cfg.attn_heads, cfg.attn_head_dim))
                if self.use_text_conditioning:
                    self.up_ca.append(CrossAttention2d(cur, cfg.text_dim, cfg.attn_heads, cfg.attn_head_dim))
                else:
                    self.up_ca.append(NoOpCrossAttention())
            if level != 0:
                self.upsamples.append(Upsample(cur))

        self.out_norm = nn.GroupNorm(_gn_groups(cur), cur, eps=1e-6)
        self.out_conv = nn.Conv2d(cur, cfg.image_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def _maybe_attend(
        self,
        x: torch.Tensor,
        sa: nn.Module,
        ca: nn.Module,
        ctx: Optional[torch.Tensor],
        ctx_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        if (h in self.cfg.attn_resolutions) or (w in self.cfg.attn_resolutions):
            x = sa(x)
            x = ca(x, ctx, ctx_mask)
        return x

    def _checkpointed(self, fn, *args):
        if self.cfg.grad_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        txt_ids: Optional[torch.Tensor],
        txt_mask: Optional[torch.Tensor],
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        te = timestep_embedding(t, self.cfg.base_channels)
        te = self.time_mlp(te)

        ctx: Optional[torch.Tensor] = None
        ctx_mask: Optional[torch.Tensor] = None
        if self.self_conditioning:
            if self_cond is None:
                self_cond = torch.zeros_like(x)
            if self_cond.shape != x.shape:
                raise RuntimeError("self_cond shape mismatch with input.")
            x = torch.cat([x, self_cond], dim=1)
        if self.use_text_conditioning:
            if txt_ids is None or txt_mask is None:
                raise RuntimeError("txt_ids/txt_mask required when use_text_conditioning is enabled.")
            if self.text_enc is None:
                raise RuntimeError("Text encoder is not initialized.")
            ctx = self.text_enc(txt_ids, txt_mask)
            ctx_mask = txt_mask
            if self.text_pool_proj is None:
                raise RuntimeError("Text pooling projection is not initialized.")
            pooled = _masked_mean(ctx, ctx_mask)
            te = te + self.text_pool_proj(pooled)

        h = self.in_conv(x)
        skips = []

        bi = 0
        dsi = 0
        for level in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_res_blocks):
                h = self._checkpointed(self.down_blocks[bi], h, te)
                h = self._checkpointed(self._maybe_attend, h, self.down_sa[bi], self.down_ca[bi], ctx, ctx_mask)
                skips.append(h)
                bi += 1
            if level != len(self.cfg.channel_mults) - 1:
                h = self._checkpointed(self.downsamples[dsi], h)
                dsi += 1

        h = self._checkpointed(self.mid1, h, te)
        h = self._checkpointed(self.mid_sa, h)
        h = self._checkpointed(self.mid_ca, h, ctx, ctx_mask)
        h = self._checkpointed(self.mid2, h, te)

        ui = 0
        usi = 0
        for level in reversed(range(len(self.cfg.channel_mults))):
            for _ in range(self.cfg.num_res_blocks):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = self._checkpointed(self.up_blocks[ui], h, te)
                h = self._checkpointed(self._maybe_attend, h, self.up_sa[ui], self.up_ca[ui], ctx, ctx_mask)
                ui += 1
            if level != 0:
                h = self._checkpointed(self.upsamples[usi], h)
                usi += 1

        h = self.out_conv(self.act(self.out_norm(h)))
        return h
