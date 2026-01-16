from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from diffusion.diffusion import Diffusion
from diffusion.model import UNet
from guided_v import _guided_v

@torch.no_grad()
def ddpm_ancestral_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 50,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = diffusion.device
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)
    b = x.shape[0]

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 0, steps + 1, device=device).long()
    total_steps = max(len(t_schedule) - 1, 0)

    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = t_schedule[i].repeat(b)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        x0 = diffusion.v_to_x0(x, t, v)
        eps = diffusion.v_to_eps(x, t, v)
        if self_conditioning:
            self_cond = x0.detach()

        if i == total_steps - 1:
            x = x0
            if progress_cb:
                progress_cb(i + 1, total_steps)
            break

        t_prev = t_schedule[i + 1].repeat(b)
        a_t = diffusion.alpha_bar[t].view(-1, 1, 1, 1)
        a_prev = diffusion.alpha_bar[t_prev].view(-1, 1, 1, 1)

        sigma = torch.sqrt(torch.clamp((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev), min=0.0))
        z = torch.randn_like(x, generator=generator) if sigma.max() > 0 else torch.zeros_like(x)
        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0)) * eps
        x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * z

        if progress_cb:
            progress_cb(i + 1, total_steps)

    # FIX: в оригинале не было return -> функция могла вернуть None
    return x
