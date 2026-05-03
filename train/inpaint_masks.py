from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn.functional as F


DEFAULT_INPAINT_MASK_MODES: dict[str, float] = {
    "rectangle": 0.5,
    "brush": 0.3,
    "random_blocks": 0.2,
}

_ALLOWED_MODES = {
    "rectangle",
    "brush",
    "center_rectangle",
    "full",
    "small",
    "large",
    "random_blocks",
}


@dataclass(frozen=True)
class InpaintMaskConfig:
    mask_min_area: float = 0.05
    mask_max_area: float = 0.60
    mask_modes: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_INPAINT_MASK_MODES))

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.mask_min_area) <= float(self.mask_max_area) <= 1.0):
            raise ValueError("inpaint mask area must satisfy 0 <= mask_min_area <= mask_max_area <= 1.")
        unknown = sorted(set(self.mask_modes) - _ALLOWED_MODES)
        if unknown:
            raise ValueError("inpaint.mask_modes contains unsupported mode(s): " + ", ".join(unknown))
        if any(float(v) < 0 for v in self.mask_modes.values()):
            raise ValueError("inpaint.mask_modes weights must be non-negative.")
        if sum(float(v) for v in self.mask_modes.values()) <= 0:
            raise ValueError("inpaint.mask_modes must include at least one positive weight.")
        if float(self.mask_modes.get("full", 0.0)) > 0 and float(self.mask_max_area) < 1.0:
            raise ValueError("inpaint.mask_modes.full requires mask_max_area=1.0.")

    @property
    def positive_modes(self) -> list[str]:
        return [name for name, weight in self.mask_modes.items() if float(weight) > 0]

    @property
    def probabilities(self) -> torch.Tensor:
        weights = torch.tensor([float(self.mask_modes[name]) for name in self.positive_modes], dtype=torch.float32)
        return weights / weights.sum()


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    if high <= low:
        return int(low)
    return int(torch.randint(low, high, (), generator=generator).item())


def _randfloat(generator: torch.Generator, low: float, high: float) -> float:
    if high <= low:
        return float(low)
    return float(torch.empty((), dtype=torch.float32).uniform_(float(low), float(high), generator=generator).item())


def _target_area_pixels(generator: torch.Generator, h: int, w: int, min_area: float, max_area: float) -> int:
    total = h * w
    lo = max(1, int(round(total * float(min_area))))
    hi = max(lo, int(round(total * float(max_area))))
    return _randint(generator, lo, hi + 1)


def _rectangle_mask(generator: torch.Generator, h: int, w: int, area_pixels: int, *, center: bool = False) -> torch.Tensor:
    aspect = _randfloat(generator, 0.4, 2.5)
    mh = int(round((area_pixels / max(aspect, 1e-6)) ** 0.5))
    mw = int(round(mh * aspect))
    mh = min(max(1, mh), h)
    mw = min(max(1, mw), w)
    if center:
        y0 = max((h - mh) // 2, 0)
        x0 = max((w - mw) // 2, 0)
    else:
        y0 = _randint(generator, 0, max(h - mh + 1, 1))
        x0 = _randint(generator, 0, max(w - mw + 1, 1))
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    mask[:, y0 : y0 + mh, x0 : x0 + mw] = 1.0
    return mask


def _brush_mask(generator: torch.Generator, h: int, w: int, target: int) -> torch.Tensor:
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    y = _randint(generator, 0, h)
    x = _randint(generator, 0, w)
    radius = max(1, min(h, w) // 18)
    steps = max(4, target // max(radius * radius, 1))
    for _ in range(steps):
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        mask[:, y0:y1, x0:x1] = 1.0
        y = min(max(0, y + _randint(generator, -radius * 3, radius * 3 + 1)), h - 1)
        x = min(max(0, x + _randint(generator, -radius * 3, radius * 3 + 1)), w - 1)
    return mask


def _random_blocks_mask(generator: torch.Generator, h: int, w: int, target: int) -> torch.Tensor:
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    remaining = target
    attempts = 0
    while remaining > 0 and attempts < 64:
        attempts += 1
        block_area = max(1, min(remaining, _randint(generator, max(1, target // 12), max(2, target // 3 + 1))))
        block = _rectangle_mask(generator, h, w, block_area, center=False)
        before = int(mask.sum().item())
        mask = torch.maximum(mask, block)
        remaining -= max(0, int(mask.sum().item()) - before)
    return mask


def _adjust_area(generator: torch.Generator, mask: torch.Tensor, *, min_area: float, max_area: float) -> torch.Tensor:
    total = int(mask.numel())
    lo = max(1, int(round(total * float(min_area))))
    hi = max(lo, int(round(total * float(max_area))))
    flat = mask.reshape(-1).clone()
    current = int(flat.sum().item())
    if current < lo:
        zeros = (flat <= 0).nonzero(as_tuple=False).reshape(-1)
        need = min(int(lo - current), int(zeros.numel()))
        if need > 0:
            perm = torch.randperm(int(zeros.numel()), generator=generator)[:need]
            flat[zeros[perm]] = 1.0
    elif current > hi:
        ones = (flat > 0).nonzero(as_tuple=False).reshape(-1)
        remove = min(int(current - hi), int(ones.numel()))
        if remove > 0:
            perm = torch.randperm(int(ones.numel()), generator=generator)[:remove]
            flat[ones[perm]] = 0.0
    return flat.reshape_as(mask)


def _select_mode(cfg: InpaintMaskConfig, generator: torch.Generator) -> str:
    modes = cfg.positive_modes
    if len(modes) == 1:
        return modes[0]
    idx = int(torch.multinomial(cfg.probabilities, 1, generator=generator).item())
    return modes[idx]


def generate_inpaint_mask(
    latent_shape: tuple[int, int] | tuple[int, int, int],
    *,
    config: InpaintMaskConfig | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    mode: str | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate a reproducible inpainting mask with shape [1,H,W]."""

    cfg = config or InpaintMaskConfig()
    if len(latent_shape) == 3:
        _, h, w = latent_shape
    elif len(latent_shape) == 2:
        h, w = latent_shape
    else:
        raise ValueError("latent_shape must be (H,W) or (C,H,W).")
    h = int(h)
    w = int(w)
    if h <= 0 or w <= 0:
        raise ValueError("mask spatial dimensions must be positive.")
    gen = generator if generator is not None else torch.Generator(device="cpu")
    if seed is not None and generator is None:
        gen.manual_seed(int(seed))
    selected = str(mode or _select_mode(cfg, gen))
    if selected not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported inpaint mask mode: {selected}")

    min_area = float(cfg.mask_min_area)
    max_area = float(cfg.mask_max_area)
    if selected == "full":
        mask = torch.ones((1, h, w), dtype=torch.float32)
    else:
        if selected == "small":
            local_min = min_area
            local_max = min(max_area, max(min_area, 0.15))
        elif selected == "large":
            local_min = max(min_area, min(max_area, 0.35))
            local_max = max_area
        else:
            local_min = min_area
            local_max = max_area
        target = _target_area_pixels(gen, h, w, local_min, local_max)
        if selected in {"rectangle", "small", "large"}:
            mask = _rectangle_mask(gen, h, w, target, center=False)
        elif selected == "center_rectangle":
            mask = _rectangle_mask(gen, h, w, target, center=True)
        elif selected == "brush":
            mask = _brush_mask(gen, h, w, target)
        elif selected == "random_blocks":
            mask = _random_blocks_mask(gen, h, w, target)
        else:
            raise ValueError(f"Unsupported inpaint mask mode: {selected}")
        mask = _adjust_area(gen, mask, min_area=local_min, max_area=local_max)

    if device is not None:
        mask = mask.to(device=device)
    return mask.to(dtype=dtype)
