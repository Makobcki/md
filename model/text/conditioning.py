from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

import torch


@dataclass(frozen=True)
class TextConditioning:
    tokens: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor
    is_uncond: torch.Tensor | None = None

    def to(self, *args, **kwargs) -> "TextConditioning":
        return TextConditioning(
            tokens=self.tokens.to(*args, **kwargs),
            mask=self.mask.to(device=kwargs.get("device", None)) if "dtype" in kwargs else self.mask.to(*args, **kwargs),
            pooled=self.pooled.to(*args, **kwargs),
            is_uncond=self.is_uncond.to(device=kwargs.get("device", None)) if self.is_uncond is not None else None,
        )

    def replace_where(self, drop: torch.Tensor, empty: "TextConditioning") -> "TextConditioning":
        drop_tokens = drop.view(-1, 1, 1).to(device=self.tokens.device, dtype=torch.bool)
        drop_mask = drop.view(-1, 1).to(device=self.mask.device, dtype=torch.bool)
        drop_pooled = drop.view(-1, 1).to(device=self.pooled.device, dtype=torch.bool)
        is_uncond = drop.to(device=self.tokens.device, dtype=torch.bool)
        if self.is_uncond is not None:
            is_uncond = torch.logical_or(self.is_uncond.to(is_uncond.device), is_uncond)
        return TextConditioning(
            tokens=torch.where(drop_tokens, empty.tokens.to(self.tokens), self.tokens),
            mask=torch.where(drop_mask, empty.mask.to(self.mask.device), self.mask),
            pooled=torch.where(drop_pooled, empty.pooled.to(self.pooled), self.pooled),
            is_uncond=is_uncond,
        )


@dataclass(frozen=True)
class ConditionBatch:
    text: TextConditioning
    source_latent: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    control_latents: Optional[torch.Tensor] = None
    task: Literal["txt2img", "img2img", "inpaint", "control"] = "txt2img"
    strength: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class TrainBatch:
    x0: torch.Tensor
    text: TextConditioning
    source_latent: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    task: str = "txt2img"
    metadata: dict | None = None

