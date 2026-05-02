from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int
    beta_start: float
    beta_end: float
    min_snr_gamma: float
    prediction_type: str
    noise_schedule: str
    cosine_s: float
