from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from diffusion.diffusion import Diffusion
from diffusion.model import UNet
from .guided_v import _guided_v


def _sigma_from_sqrt_a(sqrt_a: torch.Tensor) -> torch.Tensor:
    # sigma = sqrt(1 - alpha_bar)
    return torch.sqrt(torch.clamp(1.0 - sqrt_a * sqrt_a, min=0.0))

def _x0_eps_from_v(x: torch.Tensor, sqrt_a: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = _sigma_from_sqrt_a(sqrt_a)
    x0 = sqrt_a * x - sigma * v
    eps = sigma * x + sqrt_a * v
    return x0, eps

@torch.no_grad()
def dpm_solver_sample(
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

    x = torch.randn(shape, device=device, generator=generator) if noise is None else noise.to(device)

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 1, steps + 1, device=device).long()
    total_steps = len(t_schedule) - 1

    self_cond = None

    for i in range(total_steps):
        t = t_schedule[i].repeat(b)
        t_next = t_schedule[i + 1].repeat(b)
        t_mid = ((t + t_next) // 2).clamp(min=1)

        sqrt_a_t = diffusion.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_mid = diffusion.sqrt_alpha_bar[t_mid].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_next = diffusion.sqrt_alpha_bar[t_next].view(-1, 1, 1, 1).to(x.dtype)

        if txt_ids is None or txt_mask is None:
            cfg_scale = 1.0
            uncond_ids = None
            uncond_mask = None

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)

        sigma_t = torch.sqrt(torch.clamp(1.0 - sqrt_a_t * sqrt_a_t, min=0.0))
        x0 = sqrt_a_t * x - sigma_t * v
        eps = sigma_t * x + sqrt_a_t * v

        if self_conditioning:
            self_cond = x0.detach()

        sigma_mid = torch.sqrt(torch.clamp(1.0 - sqrt_a_mid * sqrt_a_mid, min=0.0))
        x_mid = sqrt_a_mid * x0 + sigma_mid * eps

        v_mid = _guided_v(model, x_mid, t_mid, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)

        sigma_mid2 = sigma_mid
        x0_mid = sqrt_a_mid * x_mid - sigma_mid2 * v_mid
        eps_mid = sigma_mid2 * x_mid + sqrt_a_mid * v_mid

        sigma_next = torch.sqrt(torch.clamp(1.0 - sqrt_a_next * sqrt_a_next, min=0.0))
        x = sqrt_a_next * x0_mid + sigma_next * eps_mid

        if progress_cb:
            progress_cb(i + 1, total_steps)

    t_last = t_schedule[-1].repeat(b)
    sqrt_a_last = diffusion.sqrt_alpha_bar[t_last].view(-1, 1, 1, 1).to(x.dtype)
    sigma_last = torch.sqrt(torch.clamp(1.0 - sqrt_a_last * sqrt_a_last, min=0.0))
    v_last = _guided_v(model, x, t_last, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
    x0_last = sqrt_a_last * x - sigma_last * v_last
    return x0_last


