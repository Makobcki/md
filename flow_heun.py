from __future__ import annotations

from typing import Callable, Optional

import torch

from diffusion.schedules import flow_timesteps
from model.text.conditioning import TextConditioning
from .cfg import cfg_predict


@torch.no_grad()
def sample_flow_heun(
    model,
    shape: tuple[int, int, int, int],
    text_cond: TextConditioning,
    *,
    uncond: TextConditioning | None = None,
    steps: int = 28,
    cfg_scale: float = 4.5,
    shift: float = 1.0,
    start_t: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    **model_kwargs,
) -> torch.Tensor:
    device = next(model.parameters()).device
    x = torch.randn(shape, device=device, generator=generator) if noise is None else noise.to(device)
    ts = flow_timesteps(int(steps), device=device, shift=float(shift)) * float(start_t)
    total = len(ts) - 1
    for i in range(total):
        t = ts[i].expand(shape[0])
        t_next = ts[i + 1].expand(shape[0])
        dt = ts[i + 1] - ts[i]
        v1 = cfg_predict(model, x, t, text_cond, uncond, cfg_scale, **model_kwargs)
        x_euler = x + dt.to(dtype=x.dtype) * v1
        v2 = cfg_predict(model, x_euler, t_next, text_cond, uncond, cfg_scale, **model_kwargs)
        x = x + dt.to(dtype=x.dtype) * 0.5 * (v1 + v2)
        if progress_cb is not None:
            progress_cb(i + 1, total)
    return x
