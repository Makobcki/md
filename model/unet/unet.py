from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.text.encoder import TextEncoder
from model.text.utils import masked_mean

from .attention import CrossAttention2d, NoOpCrossAttention, SelfAttention2d
from .blocks import Downsample, ResBlock, Upsample, _gn_groups
from .embeddings import timestep_embedding


@dataclass(frozen=True)
class UNetConfig:
    image_channels: int = 4
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    dropout: float = 0.1
    attn_resolutions: Tuple[int, ...] = (32, 16)
    cross_attn_resolutions: Tuple[int, ...] = ()
    attn_heads: int = 4
    attn_head_dim: int = 32
    self_attn_type: str = "global"
    self_attn_window_size: int = 8
    cross_attn_dim: int = 0
    mid_blocks: int = 1
    vocab_size: int = 50_000
    text_dim: int = 256
    text_layers: int = 4
    text_heads: int = 4
    text_max_len: int = 64
    text_spatial_conditioning: bool = False
    use_text_conditioning: bool = True
    self_conditioning: bool = False
    use_scale_shift_norm: bool = False
    grad_checkpointing: bool = False
    checkpoint_resblocks: bool = False
    checkpoint_attention: bool = False
    checkpoint_text_encoder: bool = False
    zero_init_residual: bool = False


class UNet(nn.Module):
    """
    UNet predicts v (v-prediction) with text conditioning (cross-attention).
    """
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.use_text_conditioning = bool(cfg.use_text_conditioning)
        self.self_conditioning = bool(cfg.self_conditioning)
        self.cross_attn_dim = int(cfg.cross_attn_dim) if int(cfg.cross_attn_dim) > 0 else int(cfg.text_dim)
        self.cross_attn_resolutions = tuple(cfg.cross_attn_resolutions) or tuple(cfg.attn_resolutions)

        time_dim = cfg.base_channels * 4
        chs = [cfg.base_channels * m for m in cfg.channel_mults]
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.text_enc: Optional[TextEncoder] = None
        self.text_ctx_proj: Optional[nn.Linear] = None
        self.text_pool_proj: Optional[nn.Linear] = None
        self.text_spatial_proj: Optional[nn.Linear] = None
        if self.use_text_conditioning:
            self.text_enc = TextEncoder(
                vocab_size=cfg.vocab_size,
                dim=cfg.text_dim,
                n_layers=cfg.text_layers,
                n_heads=cfg.text_heads,
                max_len=cfg.text_max_len,
                checkpoint_blocks=cfg.checkpoint_text_encoder or cfg.grad_checkpointing,
            )
            if self.cross_attn_dim != cfg.text_dim:
                self.text_ctx_proj = nn.Linear(cfg.text_dim, self.cross_attn_dim)
            self.text_pool_proj = nn.Linear(cfg.text_dim, time_dim)
            if cfg.text_spatial_conditioning:
                self.text_spatial_proj = nn.Linear(cfg.text_dim, chs[0])
                nn.init.zeros_(self.text_spatial_proj.weight)
                if self.text_spatial_proj.bias is not None:
                    nn.init.zeros_(self.text_spatial_proj.bias)

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
                    ResBlock(
                        cur,
                        ch,
                        time_dim=time_dim,
                        dropout=cfg.dropout,
                        use_scale_shift=cfg.use_scale_shift_norm,
                        zero_init=cfg.zero_init_residual,
                    )
                )
                cur = ch
                self.down_sa.append(
                    SelfAttention2d(
                        cur,
                        cfg.attn_heads,
                        cfg.attn_head_dim,
                        attention_type=cfg.self_attn_type,
                        window_size=cfg.self_attn_window_size,
                        zero_init=cfg.zero_init_residual,
                    )
                )
                if self.use_text_conditioning:
                    self.down_ca.append(
                        CrossAttention2d(
                            cur,
                            self.cross_attn_dim,
                            cfg.attn_heads,
                            cfg.attn_head_dim,
                            zero_init=cfg.zero_init_residual,
                        )
                    )
                else:
                    self.down_ca.append(NoOpCrossAttention())
            if level != len(chs) - 1:
                self.downsamples.append(Downsample(cur))

        # Mid
        self.mid1 = ResBlock(
            cur,
            cur,
            time_dim=time_dim,
            dropout=cfg.dropout,
            use_scale_shift=cfg.use_scale_shift_norm,
            zero_init=cfg.zero_init_residual,
        )
        self.mid_sa = SelfAttention2d(
            cur,
            cfg.attn_heads,
            cfg.attn_head_dim,
            attention_type=cfg.self_attn_type,
            window_size=cfg.self_attn_window_size,
            zero_init=cfg.zero_init_residual,
        )
        self.mid_ca = (
            CrossAttention2d(
                cur,
                self.cross_attn_dim,
                cfg.attn_heads,
                cfg.attn_head_dim,
                zero_init=cfg.zero_init_residual,
            )
            if self.use_text_conditioning
            else NoOpCrossAttention()
        )
        self.mid2 = ResBlock(
            cur,
            cur,
            time_dim=time_dim,
            dropout=cfg.dropout,
            use_scale_shift=cfg.use_scale_shift_norm,
            zero_init=cfg.zero_init_residual,
        )
        self.extra_mid_blocks = nn.ModuleList()
        self.extra_mid_sa = nn.ModuleList()
        self.extra_mid_ca = nn.ModuleList()
        for _ in range(max(int(cfg.mid_blocks) - 1, 0)):
            self.extra_mid_blocks.append(
                ResBlock(
                    cur,
                    cur,
                    time_dim=time_dim,
                    dropout=cfg.dropout,
                    use_scale_shift=cfg.use_scale_shift_norm,
                    zero_init=cfg.zero_init_residual,
                )
            )
            self.extra_mid_sa.append(
                SelfAttention2d(
                    cur,
                    cfg.attn_heads,
                    cfg.attn_head_dim,
                    attention_type=cfg.self_attn_type,
                    window_size=cfg.self_attn_window_size,
                    zero_init=cfg.zero_init_residual,
                )
            )
            self.extra_mid_ca.append(
                CrossAttention2d(
                    cur,
                    self.cross_attn_dim,
                    cfg.attn_heads,
                    cfg.attn_head_dim,
                    zero_init=cfg.zero_init_residual,
                )
                if self.use_text_conditioning
                else NoOpCrossAttention()
            )

        # Up
        self.up_blocks = nn.ModuleList()
        self.up_sa = nn.ModuleList()
        self.up_ca = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(len(chs))):
            ch = chs[level]
            for _ in range(cfg.num_res_blocks):
                self.up_blocks.append(
                    ResBlock(
                        cur + ch,
                        ch,
                        time_dim=time_dim,
                        dropout=cfg.dropout,
                        use_scale_shift=cfg.use_scale_shift_norm,
                        zero_init=cfg.zero_init_residual,
                    )
                )
                cur = ch
                self.up_sa.append(
                    SelfAttention2d(
                        cur,
                        cfg.attn_heads,
                        cfg.attn_head_dim,
                        attention_type=cfg.self_attn_type,
                        window_size=cfg.self_attn_window_size,
                        zero_init=cfg.zero_init_residual,
                    )
                )
                if self.use_text_conditioning:
                    self.up_ca.append(
                        CrossAttention2d(
                            cur,
                            self.cross_attn_dim,
                            cfg.attn_heads,
                            cfg.attn_head_dim,
                            zero_init=cfg.zero_init_residual,
                        )
                    )
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
        if (h in self.cross_attn_resolutions) or (w in self.cross_attn_resolutions):
            x = ca(x, ctx, ctx_mask)
        return x

    def _checkpointed(self, enabled: bool, fn, *args):
        if enabled and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _checkpoint_resblocks(self) -> bool:
        return bool(self.cfg.grad_checkpointing or self.cfg.checkpoint_resblocks)

    def _checkpoint_attention(self) -> bool:
        return bool(self.cfg.grad_checkpointing or self.cfg.checkpoint_attention)

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
            pooled = masked_mean(ctx, ctx_mask)
            te = te + self.text_pool_proj(pooled)
            if self.text_ctx_proj is not None:
                ctx = self.text_ctx_proj(ctx)

        h = self.in_conv(x)
        if self.use_text_conditioning and self.text_spatial_proj is not None:
            h = h + self.text_spatial_proj(pooled).unsqueeze(-1).unsqueeze(-1)
        skips = []

        bi = 0
        dsi = 0
        for level in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_res_blocks):
                h = self._checkpointed(self._checkpoint_resblocks(), self.down_blocks[bi], h, te)
                h = self._checkpointed(
                    self._checkpoint_attention(),
                    self._maybe_attend,
                    h,
                    self.down_sa[bi],
                    self.down_ca[bi],
                    ctx,
                    ctx_mask,
                )
                skips.append(h)
                bi += 1
            if level != len(self.cfg.channel_mults) - 1:
                h = self._checkpointed(bool(self.cfg.grad_checkpointing), self.downsamples[dsi], h)
                dsi += 1

        h = self._checkpointed(self._checkpoint_resblocks(), self.mid1, h, te)
        h = self._checkpointed(self._checkpoint_attention(), self.mid_sa, h)
        h = self._checkpointed(self._checkpoint_attention(), self.mid_ca, h, ctx, ctx_mask)
        h = self._checkpointed(self._checkpoint_resblocks(), self.mid2, h, te)
        for mid_block, mid_sa, mid_ca in zip(self.extra_mid_blocks, self.extra_mid_sa, self.extra_mid_ca):
            h = self._checkpointed(self._checkpoint_resblocks(), mid_block, h, te)
            h = self._checkpointed(self._checkpoint_attention(), mid_sa, h)
            h = self._checkpointed(self._checkpoint_attention(), mid_ca, h, ctx, ctx_mask)

        ui = 0
        usi = 0
        for level in reversed(range(len(self.cfg.channel_mults))):
            for _ in range(self.cfg.num_res_blocks):
                skip = skips.pop()
                if skip.shape[-2:] != h.shape[-2:]:
                    skip = F.interpolate(skip, size=h.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = self._checkpointed(self._checkpoint_resblocks(), self.up_blocks[ui], h, te)
                h = self._checkpointed(
                    self._checkpoint_attention(),
                    self._maybe_attend,
                    h,
                    self.up_sa[ui],
                    self.up_ca[ui],
                    ctx,
                    ctx_mask,
                )
                ui += 1
            if level != 0:
                h = self._checkpointed(bool(self.cfg.grad_checkpointing), self.upsamples[usi], h)
                usi += 1

        h = self.out_conv(self.act(self.out_norm(h)))
        return h
