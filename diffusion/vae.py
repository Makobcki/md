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
            posterior = self.vae.encode(x).latent_dist
            z = posterior.sample()
            return z * self.cfg.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent z to image in [0, 1].
        """
        ctx = torch.no_grad() if self.cfg.freeze else torch.enable_grad()
        with ctx:
            z_in = z / self.cfg.scaling_factor
            x = self.vae.decode(z_in).sample
            # VAE outputs in [-1, 1]; map to [0, 1].
            return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
