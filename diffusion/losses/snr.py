from __future__ import annotations

import torch


def get_min_snr_weights(alpha_bar_t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    eps = 1e-8
    a = alpha_bar_t.clamp(min=eps, max=1.0 - eps)
    snr = a / (1.0 - a + eps)
    g = torch.full_like(snr, float(gamma))
    return torch.minimum(snr, g) / (snr + 1.0)
