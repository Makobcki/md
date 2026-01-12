from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "v"


class Diffusion:
    def __init__(self, cfg: DiffusionConfig, device: torch.device) -> None:
        if cfg.prediction_type != "v":
            raise ValueError(f"Only v-prediction is supported, got {cfg.prediction_type}")
        self.cfg = cfg
        self.device = device

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        s1 = self.sqrt_alpha_bar[t].view(b, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
        return s1 * x0 + s2 * noise

    def v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        s1 = self.sqrt_alpha_bar[t].view(b, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
        return s1 * noise - s2 * x0

    def v_to_x0(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b = xt.shape[0]
        s1 = self.sqrt_alpha_bar[t].view(b, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
        return s1 * xt - s2 * v

    def v_to_eps(self, xt: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b = xt.shape[0]
        s1 = self.sqrt_alpha_bar[t].view(b, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
        return s2 * xt + s1 * v

    def eps_to_x0(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        b = xt.shape[0]
        s1 = self.sqrt_alpha_bar[t].view(b, 1, 1, 1)
        s2 = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
        return (xt - s2 * eps) / s1

    def sigma_from_t(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha_bar[t]
        return torch.sqrt((1.0 - a) / a)
