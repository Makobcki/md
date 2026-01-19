from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .diffusion import DiffusionConfig


def build_noise_schedule(
    cfg: "DiffusionConfig",
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cfg.noise_schedule == "linear":
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bar

    steps = cfg.timesteps
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float32, device=device) / float(steps)
    s = float(cfg.cosine_s)
    alpha_bar_fn = torch.cos(((t + s) / (1 + s)) * torch.pi / 2) ** 2
    alpha_bar_fn = alpha_bar_fn / alpha_bar_fn[0]
    betas = 1.0 - (alpha_bar_fn[1:] / alpha_bar_fn[:-1])
    betas = betas.clamp(min=0.0, max=0.999)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar
