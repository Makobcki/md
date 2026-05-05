from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from control.preprocess import CONTROL_TYPE_TO_ID as _CONTROL_TYPE_TO_ID
from model.text.conditioning import TextConditioning

from .blocks import ImageOnlyBlock, MMDiTDoubleBlock, MMDiTSingleBlock
from .config import MMDiTConfig
from .norms import AdaLNZero, build_norm, modulate
from .patch import PatchEmbed, unpatchify
from .pos_embed import add_2d_pos_embed
from .text_resampler import TextResampler
from .timestep import TimestepEmbedder

_TASK_TO_ID = {"txt2img": 0, "img2img": 1, "inpaint": 2, "control": 3, "mixed": 4}


class ControlAdapter(nn.Module):
    def __init__(self, hidden_dim: int, ratio: float = 0.25) -> None:
        super().__init__()
        inner = max(1, int(hidden_dim * float(ratio)))
        self.net = nn.Sequential(
            build_norm(hidden_dim, rms_norm=True),
            nn.Linear(hidden_dim, inner),
            nn.SiLU(),
            nn.Linear(inner, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


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
    """Latent rectified-flow MMDiT with Stage A/B/C/D conditioning options."""

    task_to_id = dict(_TASK_TO_ID)
    control_type_to_id = dict(_CONTROL_TYPE_TO_ID)

    def __init__(self, cfg: MMDiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.hidden_dim)
        self.patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.patch_size)
        self.source_patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.source_patch_size)
        self.source_mask_patch_embed = PatchEmbed(cfg.latent_channels + 1, d, cfg.source_patch_size)
        self.mask_patch_embed = PatchEmbed(1, d, cfg.mask_patch_size)
        self.control_patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.control_patch_size)
        self.coarse_patch_embed = PatchEmbed(cfg.latent_channels, d, cfg.coarse_patch_size)
        self.text_clip_in = nn.Linear(cfg.text_dim, d)
        self.text_t5_in = nn.Linear(cfg.text_dim, d)
        self.text_generic_in = nn.Linear(cfg.text_dim, d)
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
        self.type_coarse = nn.Parameter(torch.zeros(1, 1, d))
        self.control_type_embed = nn.Embedding(len(_CONTROL_TYPE_TO_ID), d)
        self.control_adapter = ControlAdapter(d, cfg.control_adapter_ratio) if cfg.control_adapter else nn.Identity()
        self.task_embed = nn.Embedding(len(_TASK_TO_ID), d)
        self.strength_in = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.cond_norm = build_norm(d, rms_norm=cfg.rms_norm)
        self.text_resampler = (
            TextResampler(
                d,
                cfg.num_heads,
                cfg.text_resampler_num_tokens,
                cfg.text_resampler_depth,
                cfg.text_resampler_mlp_ratio,
                cfg.dropout,
                cfg.attn_dropout,
                cfg.qk_norm,
                cfg.rms_norm,
                cfg.swiglu,
            )
            if cfg.text_resampler_enabled
            else None
        )
        double_blocks: list[nn.Module] = []
        for idx in range(cfg.double_stream_blocks):
            if self._block_mode(idx) == "image_only":
                double_blocks.append(ImageOnlyBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout, cfg.qk_norm, cfg.rms_norm, cfg.swiglu))
            else:
                double_blocks.append(MMDiTDoubleBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout, cfg.qk_norm, cfg.rms_norm, cfg.swiglu))
        single_blocks: list[nn.Module] = []
        for local_idx in range(cfg.single_stream_blocks):
            idx = cfg.double_stream_blocks + local_idx
            if self._block_mode(idx) == "image_only":
                single_blocks.append(ImageOnlyBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout, cfg.qk_norm, cfg.rms_norm, cfg.swiglu))
            else:
                single_blocks.append(MMDiTSingleBlock(d, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout, cfg.qk_norm, cfg.rms_norm, cfg.swiglu))
        self.double_blocks = nn.ModuleList(double_blocks)
        self.single_blocks = nn.ModuleList(single_blocks)
        self.final = FinalLayer(d, cfg.latent_channels, cfg.patch_size, cfg.zero_init_final)

    def _block_mode(self, idx: int) -> str:
        if self.cfg.attention_schedule != "hybrid":
            return "joint"
        early = int(self.cfg.early_joint_blocks)
        late_start = max(int(self.cfg.depth) - int(self.cfg.late_joint_blocks), early)
        return "joint" if idx < early or idx >= late_start else "image_only"

    def _run_block(
        self,
        block: nn.Module,
        img: torch.Tensor,
        txt: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor],
        img_mask: Optional[torch.Tensor],
        grid_hw: tuple[int, int],
        use_rope: bool,
        rope_sections: tuple[tuple[int, int, int, int, int, int], ...] | None = None,
    ):
        rope_base = tuple(self.cfg.rope_base_grid_hw)
        rope_scaling = str(self.cfg.rope_scaling)
        rope_theta = float(self.cfg.rope_theta)
        sections = rope_sections if use_rope else None
        if self.training and self.cfg.gradient_checkpointing:
            return checkpoint(
                block,
                img,
                txt,
                cond,
                mask,
                img_mask,
                grid_hw,
                use_rope,
                rope_base,
                rope_scaling,
                rope_theta,
                sections,
                use_reentrant=False,
            )
        return block(img, txt, cond, mask, img_mask, grid_hw, use_rope, rope_base, rope_scaling, rope_theta, sections)

    def _base_grid_for_patch(self, patch_size: int) -> tuple[int, int]:
        base_h, base_w = tuple(int(v) for v in self.cfg.rope_base_grid_hw)
        scale = float(self.cfg.patch_size) / float(patch_size)
        return max(1, int(round(base_h * scale))), max(1, int(round(base_w * scale)))

    def _patch_image_like_base(
        self,
        embed: PatchEmbed,
        value: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        expected_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if value.shape[-2:] != expected_hw:
            raise ValueError(f"conditioning latent/mask spatial shape {tuple(value.shape[-2:])} must match target {expected_hw}.")
        if value.shape[-2] % embed.patch_size != 0 or value.shape[-1] % embed.patch_size != 0:
            raise ValueError(f"conditioning spatial shape must be divisible by patch_size={embed.patch_size}.")
        grid_hw = (value.shape[-2] // embed.patch_size, value.shape[-1] // embed.patch_size)
        tokens = embed(value.to(device=device, dtype=dtype))
        return add_2d_pos_embed(tokens, grid_hw, self.cfg.pos_embed), grid_hw

    def _patch_image_like(
        self,
        embed: PatchEmbed,
        value: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        type_token: torch.Tensor,
        expected_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        tokens, grid_hw = self._patch_image_like_base(embed, value, device=device, dtype=dtype, expected_hw=expected_hw)
        return tokens + type_token, grid_hw

    def _control_type_ids(self, control_type: torch.Tensor | str | Sequence[str] | None, *, batch: int, streams: int, device: torch.device) -> torch.Tensor:
        if control_type is None:
            return torch.zeros((batch, streams), device=device, dtype=torch.long)
        if torch.is_tensor(control_type):
            ids = control_type.to(device=device, dtype=torch.long)
            if ids.dim() == 0:
                ids = ids.view(1, 1).expand(batch, streams)
            elif ids.dim() == 1:
                if ids.numel() == batch:
                    ids = ids.view(batch, 1).expand(batch, streams)
                elif ids.numel() == streams:
                    ids = ids.view(1, streams).expand(batch, streams)
                elif ids.numel() == 1:
                    ids = ids.view(1, 1).expand(batch, streams)
                else:
                    raise ValueError("control_type tensor must have batch, stream or scalar length.")
            elif ids.dim() == 2:
                if ids.shape == (batch, streams):
                    pass
                elif ids.shape == (batch, 1):
                    ids = ids.expand(batch, streams)
                elif ids.shape == (1, streams):
                    ids = ids.expand(batch, streams)
                else:
                    raise ValueError(f"control_type tensor shape {tuple(ids.shape)} must broadcast to {(batch, streams)}.")
            else:
                raise ValueError("control_type tensor must be scalar, 1D or 2D.")
            if int(ids.min().item()) < 0 or int(ids.max().item()) >= len(_CONTROL_TYPE_TO_ID):
                raise ValueError("control_type id out of range.")
            return ids
        if isinstance(control_type, str):
            names = [[control_type] * streams for _ in range(batch)]
        else:
            seq = list(control_type)
            if len(seq) == batch and all(isinstance(x, str) for x in seq):
                names = [[str(x)] * streams for x in seq]
            elif len(seq) == streams and all(isinstance(x, str) for x in seq):
                names = [list(map(str, seq)) for _ in range(batch)]
            elif len(seq) == batch and all(not isinstance(x, str) for x in seq):
                names = [list(map(str, x)) for x in seq]  # type: ignore[arg-type]
            else:
                raise ValueError("control_type must be a string, batch list, stream list, nested list, or tensor ids.")
        try:
            raw = [[self.control_type_to_id[name] for name in row] for row in names]
        except KeyError as exc:
            allowed = ", ".join(sorted(self.control_type_to_id))
            raise ValueError(f"Unsupported control_type {exc.args[0]!r}; allowed: {allowed}.") from exc
        return torch.tensor(raw, device=device, dtype=torch.long)

    def _task_names(self, task: str | Sequence[str], batch_size: int) -> list[str]:
        return [task] * batch_size if isinstance(task, str) else [str(v) for v in task]

    def _row_stream_mask(self, names: list[str], present_for: set[str], *, tokens: int, device: torch.device) -> torch.Tensor:
        rows = torch.tensor([name in present_for for name in names], device=device, dtype=torch.bool).view(-1, 1)
        return rows.expand(len(names), int(tokens))

    def _condition_tokens(
        self,
        x: torch.Tensor,
        source_latent: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        control_latents: Optional[torch.Tensor],
        control_type: torch.Tensor | str | Sequence[str] | None = None,
        control_strength: torch.Tensor | float | None = None,
        task: str | Sequence[str] = "txt2img",
    ) -> tuple[torch.Tensor, torch.Tensor, int, tuple[int, int], tuple[tuple[int, int, int, int, int, int], ...]]:
        if x.shape[-2] % self.cfg.patch_size != 0 or x.shape[-1] % self.cfg.patch_size != 0:
            raise ValueError(f"target latent shape {tuple(x.shape[-2:])} must be divisible by patch_size={self.cfg.patch_size}.")
        target = self.patch_embed(x)
        grid_hw = (x.shape[-2] // self.cfg.patch_size, x.shape[-1] // self.cfg.patch_size)
        target = add_2d_pos_embed(target, grid_hw, self.cfg.pos_embed) + self.type_image + self.type_target

        streams: list[tuple[torch.Tensor, torch.Tensor, tuple[int, int], int]] = []
        expected_hw = (x.shape[-2], x.shape[-1])
        task_names = self._task_names(task, x.shape[0])
        if len(task_names) != x.shape[0]:
            raise ValueError(f"task list length {len(task_names)} must match batch size {x.shape[0]}.")
        m = None if mask is None else (mask.unsqueeze(1) if mask.dim() == 3 else mask)
        source_mask_fused = False
        if source_latent is not None and m is not None and self.cfg.mask_as_source_channel:
            tokens, cond_grid = self._patch_image_like(
                self.source_mask_patch_embed,
                torch.cat([source_latent, m.to(source_latent)], dim=1),
                device=x.device,
                dtype=x.dtype,
                type_token=self.type_source,
                expected_hw=expected_hw,
            )
            streams.append((tokens, self._row_stream_mask(task_names, {"img2img", "inpaint"}, tokens=tokens.shape[1], device=x.device), cond_grid, self.cfg.source_patch_size))
            source_mask_fused = True
        elif source_latent is not None:
            tokens, cond_grid = self._patch_image_like(
                self.source_patch_embed,
                source_latent,
                device=x.device,
                dtype=x.dtype,
                type_token=self.type_source,
                expected_hw=expected_hw,
            )
            streams.append((tokens, self._row_stream_mask(task_names, {"img2img", "inpaint"}, tokens=tokens.shape[1], device=x.device), cond_grid, self.cfg.source_patch_size))
        if m is not None and not source_mask_fused:
            tokens, cond_grid = self._patch_image_like(
                self.mask_patch_embed,
                m,
                device=x.device,
                dtype=x.dtype,
                type_token=self.type_mask,
                expected_hw=expected_hw,
            )
            streams.append((tokens, self._row_stream_mask(task_names, {"inpaint"}, tokens=tokens.shape[1], device=x.device), cond_grid, self.cfg.mask_patch_size))
        if control_latents is not None:
            controls = control_latents.unsqueeze(1) if control_latents.dim() == 4 else control_latents
            if controls.dim() != 5:
                raise ValueError("control_latents must have shape [B,C,H,W] or [B,K,C,H,W].")
            if controls.shape[0] != x.shape[0]:
                raise ValueError("control_latents batch size must match x.")
            control_ids = self._control_type_ids(control_type, batch=x.shape[0], streams=controls.shape[1], device=x.device)
            control_gate = self._scalar_batch(
                control_strength,
                default=1.0,
                batch=x.shape[0],
                device=x.device,
                dtype=x.dtype,
            ).view(x.shape[0], 1, 1)
            for idx in range(controls.shape[1]):
                type_token = self.type_control
                if self.cfg.control_type_embed:
                    type_token = type_token + self.control_type_embed(control_ids[:, idx]).unsqueeze(1).to(dtype=x.dtype)
                control_base, cond_grid = self._patch_image_like_base(
                    self.control_patch_embed,
                    controls[:, idx],
                    device=x.device,
                    dtype=x.dtype,
                    expected_hw=expected_hw,
                )
                control_tokens = control_gate * (control_base + type_token)
                control_tokens = self.control_adapter(control_tokens)
                streams.append((control_tokens, self._row_stream_mask(task_names, {"control"}, tokens=control_tokens.shape[1], device=x.device), cond_grid, self.cfg.control_patch_size))
        if self.cfg.hierarchical_tokens_enabled:
            coarse_tokens, coarse_grid = self._patch_image_like_base(
                self.coarse_patch_embed,
                x,
                device=x.device,
                dtype=x.dtype,
                expected_hw=expected_hw,
            )
            streams.append((coarse_tokens + self.type_image + self.type_coarse, torch.ones((x.shape[0], coarse_tokens.shape[1]), device=x.device, dtype=torch.bool), coarse_grid, self.cfg.coarse_patch_size))

        streams.append((target, torch.ones((x.shape[0], target.shape[1]), device=x.device, dtype=torch.bool), grid_hw, self.cfg.patch_size))
        img = torch.cat([tokens for tokens, _, _, _ in streams], dim=1)
        img_mask = torch.cat([stream_mask for _, stream_mask, _, _ in streams], dim=1)

        rope_sections: list[tuple[int, int, int, int, int, int]] = []
        if self.cfg.conditioning_rope:
            offset = 0
            for tokens, _, section_grid, patch_size in streams:
                base_grid = self._base_grid_for_patch(int(patch_size))
                rope_sections.append((offset, tokens.shape[1], section_grid[0], section_grid[1], base_grid[0], base_grid[1]))
                offset += tokens.shape[1]
        target_tokens = target.shape[1]
        return img, img_mask, target_tokens, grid_hw, tuple(rope_sections)

    def _project_text_tokens_with_mask(self, text: TextConditioning, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor | None]:
        raw = text.tokens.to(device=device)
        token_types = text.token_types
        if token_types is None:
            projected = self.text_generic_in(raw.to(dtype=self.text_generic_in.weight.dtype)) + self.type_text
        else:
            token_types = token_types.to(device=device, dtype=torch.long)
            if token_types.shape != raw.shape[:2]:
                raise ValueError(f"text token_types shape {tuple(token_types.shape)} must match token shape {tuple(raw.shape[:2])}.")
            clip = self.text_clip_in(raw.to(dtype=self.text_clip_in.weight.dtype)) + self.type_clip
            t5 = self.text_t5_in(raw.to(dtype=self.text_t5_in.weight.dtype)) + self.type_t5
            generic = self.text_generic_in(raw.to(dtype=self.text_generic_in.weight.dtype)) + self.type_text
            projected = torch.where((token_types == 0).unsqueeze(-1), clip, torch.where((token_types == 1).unsqueeze(-1), t5, generic))
        projected = projected.to(dtype=dtype)
        txt_mask = text.mask.to(device=device, dtype=torch.bool) if text.mask is not None else None
        if self.text_resampler is not None:
            projected = self.text_resampler(projected, txt_mask).to(dtype=dtype)
            txt_mask = None
        return projected, txt_mask

    def _project_text_tokens(self, text: TextConditioning, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        projected, _ = self._project_text_tokens_with_mask(text, device=device, dtype=dtype)
        return projected

    def _task_ids(self, task: str | Sequence[str], batch_size: int, device: torch.device) -> torch.Tensor:
        tasks = [task] * batch_size if isinstance(task, str) else list(task)
        if len(tasks) != batch_size:
            raise ValueError(f"task list length {len(tasks)} must match batch size {batch_size}.")
        try:
            ids = [self.task_to_id[str(name)] for name in tasks]
        except KeyError as exc:
            allowed = ", ".join(sorted(self.task_to_id))
            raise ValueError(f"Unsupported MMDiT task {exc.args[0]!r}; allowed: {allowed}.") from exc
        return torch.tensor(ids, device=device, dtype=torch.long)

    def _scalar_batch(self, value: torch.Tensor | float | None, *, default: float, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), float(default), device=device, dtype=dtype)
        if not torch.is_tensor(value):
            return torch.full((batch,), float(value), device=device, dtype=dtype)
        out = value.to(device=device, dtype=dtype).reshape(-1)
        if out.numel() == 1:
            out = out.expand(batch)
        if out.shape[0] != batch:
            raise ValueError("conditioning scalar batch size mismatch")
        return out

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: TextConditioning, source_latent: torch.Tensor | None = None, mask: torch.Tensor | None = None, control_latents: torch.Tensor | None = None, control_type: torch.Tensor | str | Sequence[str] | None = None, task: str | Sequence[str] = "txt2img", strength: torch.Tensor | float | None = None, control_strength: torch.Tensor | float | None = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("x must have shape [B,C,H,W].")
        if t.dim() != 1:
            t = t.reshape(-1)
        if t.shape[0] != x.shape[0]:
            raise ValueError("t batch size must match x.")
        img, img_mask, target_tokens, grid_hw, rope_sections = self._condition_tokens(x, source_latent, mask, control_latents, control_type, control_strength, task=task)
        txt, txt_mask = self._project_text_tokens_with_mask(text, device=x.device, dtype=img.dtype)
        pooled = self.pooled_in(text.pooled.to(device=x.device, dtype=self.pooled_in.weight.dtype)).to(dtype=img.dtype)
        task_cond = self.task_embed(self._task_ids(task, x.shape[0], x.device)).to(dtype=img.dtype)
        cond_parts = self.t_embed(t).to(dtype=img.dtype) + pooled + task_cond
        if self.cfg.strength_embed:
            mask_area = torch.zeros(x.shape[0], device=x.device, dtype=img.dtype)
            if mask is not None:
                mm = mask.to(device=x.device, dtype=img.dtype)
                mask_area = mm.reshape(mm.shape[0], -1).mean(dim=1)
            strength_v = self._scalar_batch(strength, default=1.0, batch=x.shape[0], device=x.device, dtype=img.dtype)
            if control_latents is None:
                control_v = torch.zeros(x.shape[0], device=x.device, dtype=img.dtype)
            else:
                control_v = self._scalar_batch(control_strength, default=1.0, batch=x.shape[0], device=x.device, dtype=img.dtype)
            cond_parts = cond_parts + self.strength_in(torch.stack([strength_v, mask_area, control_v], dim=1))
        cond = self.cond_norm(cond_parts)

        all_blocks: list[nn.Module] = list(self.double_blocks) + list(self.single_blocks)
        use_rope_sections = self.cfg.pos_embed == "rope_2d" and bool(rope_sections)
        for idx, block in enumerate(all_blocks):
            if self._block_mode(idx) == "image_only":
                img, txt = self._run_block(block, img, txt, cond, None, img_mask, grid_hw, use_rope_sections, rope_sections)
            else:
                img, txt = self._run_block(block, img, txt, cond, txt_mask, img_mask, grid_hw, use_rope_sections, rope_sections)

        target = img[:, -target_tokens:]
        patches = self.final(target, cond)
        return unpatchify(patches, channels=self.cfg.latent_channels, height=x.shape[-2], width=x.shape[-1], patch_size=self.cfg.patch_size)
