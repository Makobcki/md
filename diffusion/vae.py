from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    from diffusers import AutoencoderKL
except Exception:  # pragma: no cover - optional dependency
    AutoencoderKL = None


@dataclass(frozen=True)
class VAEConfig:
    pretrained: Optional[str] = None
    freeze: bool = True
    scaling_factor: float = 0.18215


class VAEWrapper:
    """
    VAE wrapper for latent diffusion.

    Notes:
    - The VAE expects inputs in [-1, 1].
    - After decode we map back to [0, 1] and clamp.
    - scaling_factor matches Stable Diffusion (0.18215) so latents are normalized.
    """

    def __init__(
        self,
        *,
        pretrained: Optional[str],
        freeze: bool,
        scaling_factor: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not pretrained:
            raise RuntimeError("VAE pretrained path/name is required for latent diffusion.")
        if AutoencoderKL is None:
            raise RuntimeError("diffusers is required for VAEWrapper. Please install diffusers.")
        self.cfg = VAEConfig(pretrained=pretrained, freeze=bool(freeze), scaling_factor=float(scaling_factor))
        self.device = device
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(pretrained).to(device=device, dtype=dtype)
        if self.cfg.freeze:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image in [-1, 1] to latent z, scaled by scaling_factor.
        """
        ctx = torch.no_grad() if self.cfg.freeze else torch.enable_grad()
        with ctx:
            x = x.to(device=self.device, dtype=self.dtype)
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample()
            return z * self.cfg.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z to image in [0, 1].
        """
        ctx = torch.no_grad() if self.cfg.freeze else torch.enable_grad()
        with ctx:
            # 1. Нормализация латентов
            z_in = z / self.cfg.scaling_factor

            # 2. Приведение dtype/device к VAE (КРИТИЧНО)
            z_in = z_in.to(device=self.device, dtype=self.dtype)

            # 3. Decode
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    x = self.vae.decode(z_in).sample
            else:
                x = self.vae.decode(z_in).sample

            # 4. [-1, 1] -> [0, 1]
            return ((x + 1.0) / 2.0).clamp(0.0, 1.0)