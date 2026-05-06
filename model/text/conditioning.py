from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

TaskName = Literal["txt2img", "img2img", "inpaint", "control"]
TaskSpec = str | Sequence[str]


@dataclass(frozen=True)
class TextConditioning:
    tokens: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor
    is_uncond: torch.Tensor | None = None
    # Optional per-token encoder type IDs. 0=CLIP-like, 1=T5-like, 2=generic/other.
    # Existing caches and tests may omit this; the model then uses a generic text type embedding.
    token_types: torch.Tensor | None = None

    def to(self, *args: Any, **kwargs: Any) -> TextConditioning:
        tokens = self.tokens.to(*args, **kwargs)
        pooled = self.pooled.to(*args, **kwargs)
        return TextConditioning(
            tokens=tokens,
            mask=self.mask.to(device=tokens.device),
            pooled=pooled,
            is_uncond=self.is_uncond.to(device=tokens.device)
            if self.is_uncond is not None
            else None,
            token_types=self.token_types.to(device=tokens.device)
            if self.token_types is not None
            else None,
        )

    def replace_where(self, drop: torch.Tensor, empty: TextConditioning) -> TextConditioning:
        drop_tokens = drop.view(-1, 1, 1).to(device=self.tokens.device, dtype=torch.bool)
        drop_mask = drop.view(-1, 1).to(device=self.mask.device, dtype=torch.bool)
        drop_pooled = drop.view(-1, 1).to(device=self.pooled.device, dtype=torch.bool)
        is_uncond = drop.to(device=self.tokens.device, dtype=torch.bool)
        if self.is_uncond is not None:
            is_uncond = torch.logical_or(self.is_uncond.to(is_uncond.device), is_uncond)
        if self.token_types is None:
            token_types = None
        else:
            self_types = self.token_types.to(device=self.tokens.device, dtype=torch.long)
            if empty.token_types is None:
                empty_types = torch.zeros_like(self_types)
            else:
                empty_types = empty.token_types.to(device=self.tokens.device, dtype=torch.long)
                if empty_types.shape[:1] == (1,) and self_types.shape[:1] != (1,):
                    empty_types = empty_types.expand_as(self_types)
            drop_types = drop.view(-1, *([1] * (self_types.dim() - 1))).to(
                device=self.tokens.device, dtype=torch.bool
            )
            token_types = torch.where(drop_types, empty_types.to(self_types), self_types)
        return TextConditioning(
            tokens=torch.where(drop_tokens, empty.tokens.to(self.tokens), self.tokens),
            mask=torch.where(drop_mask, empty.mask.to(self.mask.device), self.mask),
            pooled=torch.where(drop_pooled, empty.pooled.to(self.pooled), self.pooled),
            is_uncond=is_uncond,
            token_types=token_types,
        )


@dataclass(frozen=True)
class ConditionBatch:
    text: TextConditioning
    source_latent: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    control_latents: torch.Tensor | None = None
    control_type: torch.Tensor | None = None
    task: TaskName | str = "txt2img"
    strength: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainBatch:
    x0: torch.Tensor
    text: TextConditioning
    source_latent: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    control_latents: torch.Tensor | None = None
    control_type: torch.Tensor | None = None
    task: str | list[str] = "txt2img"
    strength: torch.Tensor | None = None
    control_strength: torch.Tensor | None = None
    metadata: dict | None = None
