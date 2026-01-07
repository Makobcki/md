from __future__ import annotations

import torch
from dataclasses import dataclass
import diffusion

@dataclass(frozen=True)
class DDIMConfig:
    timesteps: int = 1000
    eta: float = 0.0  # 0 = детерминированный DDIM, >0 добавляет стохастичность


class DDIM:
    def __init__(self, cfg: DDIMConfig, betas: torch.Tensor) -> None:
        """
        betas — те же betas, что использовались в DDPM (НЕ менять!)
        """
        self.cfg = cfg
        self.device = betas.device

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(
        self,
        model,
        shape,
        steps: int,
    ) -> torch.Tensor:
        """
        shape: (B, C, H, W)
        steps: <= cfg.timesteps
        """
        assert steps <= self.cfg.timesteps, (
            f"DDIM steps={steps} > trained timesteps={self.cfg.timesteps}"
        )

        B = shape[0]
        x = torch.randn(shape, device=self.device)

        # равномерно выбираем таймстепы
        t_seq = torch.linspace(self.cfg.timesteps - 1, 0, steps, device=self.device).round().long()

        for i in range(len(t_seq) - 1):
            t = t_seq[i]
            t_prev = t_seq[i + 1]

            tt = torch.full((B,), t, device=self.device, dtype=torch.long)

            v = model(x, tt)
            # Конвертируем v в шум, чтобы использовать старые формулы DDIM
            a_t = self.alpha_bar[tt].view(-1, 1, 1, 1)  # [B,1,1,1]
            sqrt_a = torch.sqrt(a_t)
            sqrt_1m = torch.sqrt(1.0 - a_t)

            # eps = sqrt(1-a)*x_t + sqrt(a)*v
            eps = sqrt_1m * x + sqrt_a * v

            a_t = self.alpha_bar[t]
            a_prev = self.alpha_bar[t_prev]

            a_t = a_t.view(1, 1, 1, 1)
            a_prev = a_prev.view(1, 1, 1, 1)

            # предсказание x0
            x0 = (x - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)
            x0 = x0.clamp(-1, 1)  # 🔥 важный стабилизатор

            # DDIM формула
            sigma = (
                self.cfg.eta
                * torch.sqrt((1 - a_prev) / (1 - a_t))
                * torch.sqrt(1 - a_t / a_prev)
            )

            noise = torch.randn_like(x) if self.cfg.eta > 0 else 0.0

            x = (
                torch.sqrt(a_prev) * x0
                + torch.sqrt(1 - a_prev - sigma**2) * eps
                + sigma * noise
            )

        return x
