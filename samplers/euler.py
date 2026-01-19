from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from diffusion.diffusion import Diffusion
from diffusion.model import UNet
from .guided_v import _guided_v

@torch.no_grad()
def euler_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 30,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
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

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 0, steps + 1, device=device).long()
    total_steps = max(len(t_schedule) - 1, 0)

    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = t_schedule[i].repeat(b)
        t_next = t_schedule[i + 1].repeat(b)
        sqrt_a_t = diffusion.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_next = diffusion.sqrt_alpha_bar[t_next].view(-1, 1, 1, 1).to(x.dtype)
        sigma = diffusion.sigma_from_t(t).view(-1, 1, 1, 1).to(x.dtype)
        sigma_next = diffusion.sigma_from_t(t_next).view(-1, 1, 1, 1).to(x.dtype)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        eps = diffusion.v_to_eps(x, t, v)
        if self_conditioning:
            self_cond = diffusion.v_to_x0(x, t, v).detach()
        x_hat = x / sqrt_a_t
        x_hat_next = x_hat + (sigma_next - sigma) * eps
        x = x_hat_next * sqrt_a_next

        if progress_cb:
            progress_cb(i + 1, total_steps)

    t_final = t_schedule[-1].repeat(b)
    v_final = _guided_v(model, x, t_final, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
    return diffusion.v_to_x0(x, t_final, v_final)
