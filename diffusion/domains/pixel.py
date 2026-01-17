from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from diffusion.core.diffusion import Diffusion

from .domain import Batch, PreparedBatch


def _assert_in_range(name: str, x: torch.Tensor, lo: float, hi: float, eps: float = 1e-3) -> None:
    if x.min().item() < lo - eps or x.max().item() > hi + eps:
        raise RuntimeError(f"{name} is out of range [{lo}, {hi}]")


@dataclass
class PixelDomain:
    # Пиксельный домен: x в [-1,1], H=W=image_size.
    diffusion: Diffusion
    device: torch.device
    channels_last: bool

    name: str = "pixel"

    def prepare_batch(self, batch: Batch) -> PreparedBatch:
        x = batch.x
        if self.channels_last:
            x = x.to(self.device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(self.device, non_blocking=True)
        _assert_in_range("x0", x, -1.0, 1.0)
        txt_ids = batch.txt_ids
        txt_mask = batch.txt_mask
        if txt_ids is not None:
            txt_ids = txt_ids.to(self.device, non_blocking=True)
        if txt_mask is not None:
            txt_mask = txt_mask.to(self.device, non_blocking=True)
        return PreparedBatch(
            x=x,
            txt_ids=txt_ids,
            txt_mask=txt_mask,
        )

    def sample_noise_like(self, x0: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.diffusion.q_sample(x0, t, noise)

    def v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.diffusion.v_target(x0, t, noise)

    def decode_for_preview(self, x: torch.Tensor) -> torch.Tensor:
        # Приводим к [0,1] для превью.
        return (x.clamp(-1, 1) + 1.0) / 2.0

    def shape_info(self, x: torch.Tensor) -> Dict[str, int]:
        _, c, h, w = x.shape
        return {"c": c, "h": h, "w": w}
