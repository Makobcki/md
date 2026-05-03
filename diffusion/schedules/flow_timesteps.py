from __future__ import annotations

import torch


def flow_timesteps(steps: int, *, device: torch.device, shift: float = 1.0) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    if shift != 1.0:
        s = float(shift)
        ts = (s * ts) / (1.0 + (s - 1.0) * ts).clamp_min(1e-6)
        ts[0] = 1.0
        ts[-1] = 0.0
    return ts

