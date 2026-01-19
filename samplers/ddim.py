from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from diffusion.diffusion import Diffusion
from diffusion.model import UNet
from .guided_v import _guided_v

@torch.no_grad()
def ddim_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 50,
    eta: float = 0.0,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
    clamp_x0: bool = True,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = diffusion.device
    b, _, _, _ = shape
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)

    T = diffusion.cfg.timesteps
    ts = torch.linspace(T - 1, 0, steps + 1, device=device).long()
    total_steps = max(len(ts) - 1, 0)

    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = ts[i].repeat(b)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        x0 = diffusion.v_to_x0(x, t, v)
        eps = diffusion.v_to_eps(x, t, v)
        if clamp_x0:
            x0 = x0.clamp(-1.0, 1.0)
        if self_conditioning:
            self_cond = x0.detach()

        if i == total_steps - 1:
            x = x0
            if progress_cb:
                progress_cb(i + 1, total_steps)
            break

        t_prev = ts[i + 1].repeat(b)
        a_t = diffusion.alpha_bar[t].view(-1, 1, 1, 1)
        a_prev = diffusion.alpha_bar[t_prev].view(-1, 1, 1, 1)

        sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev))

        if float(sigma.max().item()) > 0.0:
            z = torch.empty_like(x).normal_(generator=generator)
        else:
            z = torch.zeros_like(x)

        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0)) * eps
        x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * z

        if progress_cb:
            progress_cb(i + 1, total_steps)

    return x

