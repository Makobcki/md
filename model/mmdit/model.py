from __future__ import annotations

from collections.abc import Sequence
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


_TASK_TO_ID = {
    "txt2img": 0,
    "img2img": 1,
    "inpaint": 2,
    "control": 3,
    "mixed": 4,
}


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, out_channels: int = 4, patch_size: int = 2, zero_init_final: bool = True) -> None:
        super().__init__()
        self.norm = build_norm(hidden_dim, rms_norm=False)
        self.adaln = AdaLNZero(hidden_dim, chunks=2)
        self.linear = nn.Linear(hidden_dim, out_channels * patch_size * patch_size)
        if zero_init_final:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, img_tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(cond)
        return self.linear(modulate(self.norm(img_tokens), shift, scale))


class MMDiTFlowModel(nn.Module):
    """Latent rectified-flow MMDiT.

    The model predicts flow velocity in latent space. Text tokens, optional
    source-image tokens, optional inpainting mask tokens, optional control
    latent tokens, and target noisy latent tokens are processed with joint
    text/image attention. Only target tokens are projected back to velocity.
    """

    task_to_id = dict(_TASK_TO_ID)

    def __init__(self, cfg: MMDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.hidden_dim)
        self.patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.patch_size)
        self.source_patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.patch_size)
        self.mask_patch_embed = PatchEmbed(1, d, cfg.patch_size)
        self.control_patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.patch_size)
        self.text_clip_in = nn.Linear(cfg.text_dim, d)
        self.text_t5_in = nn.Linear(cfg.text_dim, d)
        self.text_generic_in = nn.Linear(cfg.text_dim, d)
        # Backward-compatible attribute name used by older tests/checkpoints.
        self.text_in = self.text_generic_in
        self.pooled_in = nn.Linear(cfg.pooled_dim, d)
        self.t_embed = TimestepEmbedder(d)
        self.type_clip = nn.Parameter(torch.zeros(1, 1, d))
        self.type_t5 = nn.Parameter(torch.zeros(1, 1, d))
        self.type_text = nn.Parameter(torch.zeros(1, 1, d))
        self.type_image = nn.Parameter(torch.zeros(1, 1, d))
        self.type_target = nn.Parameter(torch.zeros(1, 1, d))
        self.type_source = nn.Parameter(torch.zeros(1, 1, d))
        self.type_mask = nn.Parameter(torch.zeros(1, 1, d))
        self.type_control = nn.Parameter(torch.zeros(1, 1, d))
        self.task_embed = nn.Embedding(len(_TASK_TO_ID), d)
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

    def _run_block(
        self,
        block: nn.Module,
        img: torch.Tensor,
        txt: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor],
        grid_hw: tuple[int, int],
    ):
        use_rope = self.cfg.pos_embed == "rope_2d"
        if self.training and self.cfg.gradient_checkpointing:
            return checkpoint(block, img, txt, cond, mask, grid_hw, use_rope, use_reentrant=False)
        return block(img, txt, cond, mask, grid_hw, use_rope)

    def _patch_image_like(
        self,
        embed: PatchEmbed,
        value: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        grid_hw: tuple[int, int],
        type_token: torch.Tensor,
        expected_hw: tuple[int, int],
    ) -> torch.Tensor:
        if value.shape[-2:] != expected_hw:
            raise ValueError(f"conditioning latent/mask spatial shape {tuple(value.shape[-2:])} must match target {expected_hw}.")
        tokens = embed(value.to(device=device, dtype=dtype))
        return add_2d_pos_embed(tokens, grid_hw, self.cfg.pos_embed) + type_token

    def _condition_tokens(
        self,
        x: torch.Tensor,
        source_latent: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        control_latents: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, int, tuple[int, int]]:
        target = self.patch_embed(x)
        h_grid = x.shape[-2] // self.cfg.patch_size
        w_grid = x.shape[-1] // self.cfg.patch_size
        grid_hw = (h_grid, w_grid)
        target = add_2d_pos_embed(target, grid_hw, self.cfg.pos_embed) + self.type_image + self.type_target
        chunks: list[torch.Tensor] = []
        expected_hw = (x.shape[-2], x.shape[-1])
        if source_latent is not None:
            chunks.append(
                self._patch_image_like(
                    self.source_patch_embed,
                    source_latent,
                    device=x.device,
                    dtype=x.dtype,
                    grid_hw=grid_hw,
                    type_token=self.type_source,
                    expected_hw=expected_hw,
                )
            )
        if mask is not None:
            m = mask.unsqueeze(1) if mask.dim() == 3 else mask
            chunks.append(
                self._patch_image_like(
                    self.mask_patch_embed,
                    m,
                    device=x.device,
                    dtype=x.dtype,
                    grid_hw=grid_hw,
                    type_token=self.type_mask,
                    expected_hw=expected_hw,
                )
            )
        if control_latents is not None:
            controls = control_latents
            if controls.dim() == 4:
                controls = controls.unsqueeze(1)
            if controls.dim() != 5:
                raise ValueError("control_latents must have shape [B,C,H,W] or [B,K,C,H,W].")
            if controls.shape[0] != x.shape[0]:
                raise ValueError("control_latents batch size must match x.")
            for idx in range(controls.shape[1]):
                chunks.append(
                    self._patch_image_like(
                        self.control_patch_embed,
                        controls[:, idx],
                        device=x.device,
                        dtype=x.dtype,
                        grid_hw=grid_hw,
                        type_token=self.type_control,
                        expected_hw=expected_hw,
                    )
                )
        chunks.append(target)
        return torch.cat(chunks, dim=1), target.shape[1], grid_hw

    def _project_text_tokens(self, text: TextConditioning, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raw = text.tokens.to(device=device)
        token_types = text.token_types
        if token_types is None:
            projected = self.text_generic_in(raw.to(dtype=self.text_generic_in.weight.dtype)) + self.type_text
            return projected.to(dtype=dtype)
        token_types = token_types.to(device=device, dtype=torch.long)
        if token_types.shape != raw.shape[:2]:
            raise ValueError(f"text token_types shape {tuple(token_types.shape)} must match token shape {tuple(raw.shape[:2])}.")
        clip = self.text_clip_in(raw.to(dtype=self.text_clip_in.weight.dtype)) + self.type_clip
        t5 = self.text_t5_in(raw.to(dtype=self.text_t5_in.weight.dtype)) + self.type_t5
        generic = self.text_generic_in(raw.to(dtype=self.text_generic_in.weight.dtype)) + self.type_text
        clip_mask = (token_types == 0).unsqueeze(-1)
        t5_mask = (token_types == 1).unsqueeze(-1)
        projected = torch.where(clip_mask, clip, torch.where(t5_mask, t5, generic))
        return projected.to(dtype=dtype)

    def _task_ids(self, task: str | Sequence[str], batch_size: int, device: torch.device) -> torch.Tensor:
        if isinstance(task, str):
            tasks = [task] * batch_size
        else:
            tasks = list(task)
            if len(tasks) != batch_size:
                raise ValueError(f"task list length {len(tasks)} must match batch size {batch_size}.")
        try:
            ids = [self.task_to_id[str(name)] for name in tasks]
        except KeyError as exc:
            allowed = ", ".join(sorted(self.task_to_id))
            raise ValueError(f"Unsupported MMDiT task {exc.args[0]!r}; allowed: {allowed}.") from exc
        return torch.tensor(ids, device=device, dtype=torch.long)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: TextConditioning,
        source_latent: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        control_latents: torch.Tensor | None = None,
        task: str | Sequence[str] = "txt2img",
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("x must have shape [B,C,H,W].")
        if t.dim() != 1:
            t = t.reshape(-1)
        if t.shape[0] != x.shape[0]:
            raise ValueError("t batch size must match x.")
        img, target_tokens, grid_hw = self._condition_tokens(x, source_latent, mask, control_latents)
        txt = self._project_text_tokens(text, device=x.device, dtype=img.dtype)
        pooled = self.pooled_in(text.pooled.to(device=x.device, dtype=self.pooled_in.weight.dtype)).to(dtype=img.dtype)
        task_cond = self.task_embed(self._task_ids(task, x.shape[0], x.device)).to(dtype=img.dtype)
        cond = self.cond_norm(self.t_embed(t).to(dtype=img.dtype) + pooled + task_cond)
        txt_mask = text.mask.to(device=x.device, dtype=torch.bool) if text.mask is not None else None

        for block in self.double_blocks:
            img, txt = self._run_block(block, img, txt, cond, txt_mask, grid_hw)
        for block in self.single_blocks:
            img, txt = self._run_block(block, img, txt, cond, txt_mask, grid_hw)

        target = img[:, -target_tokens:]
        patches = self.final(target, cond)
        return unpatchify(
            patches,
            channels=self.cfg.latent_channels,
            height=x.shape[-2],
            width=x.shape[-1],
            patch_size=self.cfg.patch_size,
        )
