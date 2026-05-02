from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.text.conditioning import TextConditioning

from .blocks import MMDiTDoubleBlock, MMDiTSingleBlock
from .config import MMDiTConfig
from .norms import AdaLNZero, build_norm, modulate
from .patch import PatchEmbed, unpatchify
from .pos_embed import add_2d_pos_embed
from .timestep import TimestepEmbedder


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int = 4, patch_size: int = 2, zero_init_final: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.adaln = AdaLNZero(hidden_dim, chunks=2)
        self.linear = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
        if zero_init_final:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, img_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(cond)
        return self.linear(modulate(self.norm(img_tokens), shift, scale))


class MMDiTFlowModel(nn.Module):
    def __init__(self, cfg: MMDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.hidden_dim)
        self.patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.patch_size)
        self.source_patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.patch_size)
        self.mask_patch_embed = PatchEmbed(1, d, cfg.patch_size)
        self.text_in = nn.Linear(cfg.text_dim, d)
        self.pooled_in = nn.Linear(cfg.pooled_dim, d)
        self.t_embed = TimestepEmbedder(d)
        self.type_text = nn.Parameter(torch.zeros(1, 1, d))
        self.type_target = nn.Parameter(torch.zeros(1, 1, d))
        self.type_source = nn.Parameter(torch.zeros(1, 1, d))
        self.type_mask = nn.Parameter(torch.zeros(1, 1, d))
        self.cond_norm = build_norm(d, rms_norm=cfg.rms_norm)
        self.double_blocks = nn.ModuleList(
            [
                MMDiTDoubleBlock(
                    d,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.qk_norm,
                    cfg.rms_norm,
                    cfg.swiglu,
                )
                for _ in range(cfg.double_stream_blocks)
            ]
        )
        self.single_blocks = nn.ModuleList(
            [
                MMDiTSingleBlock(
                    d,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    cfg.dropout,
                    cfg.attn_dropout,
                    cfg.qk_norm,
                    cfg.rms_norm,
                    cfg.swiglu,
                )
                for _ in range(cfg.single_stream_blocks)
            ]
        )
        self.final = FinalLayer(d, cfg.latent_channels, cfg.patch_size, cfg.zero_init_final)

    def _run_block(self, block: nn.Module, img: torch.Tensor, txt: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor]):
        if self.training and self.cfg.gradient_checkpointing:
            return checkpoint(block, img, txt, cond, mask, use_reentrant=False)
        return block(img, txt, cond, mask)

    def _condition_tokens(
        self,
        x: torch.Tensor,
        source_latent: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, int, tuple[int, int]]:
        target = self.patch_embed(x)
        h_grid = x.shape[-2] // self.cfg.patch_size
        w_grid = x.shape[-1] // self.cfg.patch_size
        target = add_2d_pos_embed(target, (h_grid, w_grid), self.cfg.pos_embed) + self.type_target
        chunks = []
        if source_latent is not None:
            src = self.source_patch_embed(source_latent.to(device=x.device, dtype=x.dtype))
            src = add_2d_pos_embed(src, (h_grid, w_grid), self.cfg.pos_embed) + self.type_source
            chunks.append(src)
        if mask is not None:
            m = mask
            if m.dim() == 3:
                m = m.unsqueeze(1)
            m = self.mask_patch_embed(m.to(device=x.device, dtype=x.dtype))
            m = add_2d_pos_embed(m, (h_grid, w_grid), self.cfg.pos_embed) + self.type_mask
            chunks.append(m)
        chunks.append(target)
        return torch.cat(chunks, dim=1), target.shape[1], (h_grid, w_grid)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: TextConditioning,
        source_latent: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        task: str = "txt2img",
    ) -> torch.Tensor:
        del task
        if t.dim() != 1:
            t = t.reshape(-1)
        img, target_tokens, _grid = self._condition_tokens(x, source_latent, mask)
        txt = self.text_in(text.tokens.to(device=x.device, dtype=self.text_in.weight.dtype)) + self.type_text
        txt = txt.to(dtype=img.dtype)
        pooled = self.pooled_in(text.pooled.to(device=x.device, dtype=self.pooled_in.weight.dtype)).to(dtype=img.dtype)
        cond = self.cond_norm(self.t_embed(t).to(dtype=img.dtype) + pooled)
        txt_mask = text.mask.to(device=x.device, dtype=torch.bool) if text.mask is not None else None

        for block in self.double_blocks:
            img, txt = self._run_block(block, img, txt, cond, txt_mask)
        for block in self.single_blocks:
            img, txt = self._run_block(block, img, txt, cond, txt_mask)

        target = img[:, -target_tokens:]
        patches = self.final(target, cond)
        return unpatchify(
            patches,
            channels=self.cfg.latent_channels,
            height=x.shape[-2],
            width=x.shape[-1],
            patch_size=self.cfg.patch_size,
        )

