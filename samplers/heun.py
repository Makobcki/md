from __future__ import annotations

from typing import Callable, Optional, Tuple
import torch

from diffusion.diffusion import Diffusion
from diffusion.model import UNet
from .guided_v import _guided_v


def _sigma_from_sqrt_a(sqrt_a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(1.0 - sqrt_a * sqrt_a, min=0.0))


@torch.no_grad()
def heun_sample(
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

    if txt_ids is None or txt_mask is None:
        cfg_scale = 1.0
        uncond_ids = None
        uncond_mask = None

    self_cond: Optional[torch.Tensor] = None

    for i in range(total_steps):
        t = t_schedule[i].repeat(b)
        t_next = t_schedule[i + 1].repeat(b)

        sqrt_a_t = diffusion.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_next = diffusion.sqrt_alpha_bar[t_next].view(-1, 1, 1, 1).to(x.dtype)

        sigma_t = _sigma_from_sqrt_a(sqrt_a_t).to(x.dtype)
        sigma_next = _sigma_from_sqrt_a(sqrt_a_next).to(x.dtype)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)

        x0 = sqrt_a_t * x - sigma_t * v
        eps = sigma_t * x + sqrt_a_t * v

        if self_conditioning:
            self_cond = x0.detach()

        x_euler = sqrt_a_next * x0 + sigma_next * eps

        v_next = _guided_v(
            model,
            x_euler,
            t_next,
            txt_ids,
            txt_mask,
            self_cond,
            cfg_scale,
            uncond_ids,
            uncond_mask,
        )

        x0_next = sqrt_a_next * x_euler - sigma_next * v_next
        eps_next = sigma_next * x_euler + sqrt_a_next * v_next

        x0_h = 0.5 * (x0 + x0_next)
        eps_h = 0.5 * (eps + eps_next)

        x = sqrt_a_next * x0_h + sigma_next * eps_h

        if progress_cb:
            progress_cb(i + 1, total_steps)

    t_last = t_schedule[-1].repeat(b)
    sqrt_a_last = diffusion.sqrt_alpha_bar[t_last].view(-1, 1, 1, 1).to(x.dtype)
    sigma_last = _sigma_from_sqrt_a(sqrt_a_last).to(x.dtype)

    v_last = _guided_v(model, x, t_last, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
    x0_last = sqrt_a_last * x - sigma_last * v_last
    return x0_last
