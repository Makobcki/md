from __future__ import annotations

from typing import Callable, Optional

import torch

from diffusion.schedules import flow_timesteps
from model.text.conditioning import TextConditioning
from .cfg import cfg_predict, preserve_inpaint_region


def _infer_sampling_device(model, text_cond: TextConditioning, noise: torch.Tensor | None) -> torch.device:
    if noise is not None:
        return noise.device
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return text_cond.tokens.device


@torch.no_grad()
def sample_flow_euler(
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
    device = _infer_sampling_device(model, text_cond, noise)
    x = torch.randn(shape, device=device, generator=generator) if noise is None else noise.to(device)
    inpaint_reference_noise = None
    if model_kwargs.get("task", "txt2img") == "inpaint" and model_kwargs.get("source_latent") is not None and model_kwargs.get("mask") is not None:
        src = model_kwargs["source_latent"].to(device=device, dtype=x.dtype)
        start = float(start_t)
        if start > 0.0:
            inpaint_reference_noise = (x - (1.0 - start) * src) / start
        else:
            inpaint_reference_noise = torch.zeros_like(src)
    ts = flow_timesteps(int(steps), device=device, shift=float(shift)) * float(start_t)
    total = len(ts) - 1
    for i in range(total):
        t = ts[i].expand(shape[0])
        dt = ts[i + 1] - ts[i]
        v = cfg_predict(model, x, t, text_cond, uncond, cfg_scale, **model_kwargs)
        x = x + dt.to(dtype=x.dtype) * v
        x = preserve_inpaint_region(
            x,
            source_latent=model_kwargs.get("source_latent"),
            mask=model_kwargs.get("mask"),
            task=model_kwargs.get("task", "txt2img"),
            reference_noise=inpaint_reference_noise,
            t=ts[i + 1],
        )
        if progress_cb is not None:
            progress_cb(i + 1, total)
    return x
