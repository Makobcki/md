from __future__ import annotations

import pytest
import torch

from config.train import TrainConfig
from diffusion.objectives.flow_matching import apply_timestep_shift, sample_timestep


def test_timestep_sampling_strategies_are_bounded_and_deterministic_with_seed() -> None:
    for mode in ("uniform", "logit_normal", "shifted_logit_normal", "cosmap", "cosmap_like"):
        torch.manual_seed(123)
        a = sample_timestep(64, mode=mode, device=torch.device("cpu"), shift=3.0)
        torch.manual_seed(123)
        b = sample_timestep(64, mode=mode, device=torch.device("cpu"), shift=3.0)
        assert torch.equal(a, b)
        assert torch.all((a >= 0.0) & (a <= 1.0))


def test_shifted_logit_normal_uses_positive_timestep_shift() -> None:
    t = torch.tensor([0.25, 0.5, 0.75])
    shifted = apply_timestep_shift(t, 3.0)
    assert torch.all(shifted > t)
    assert torch.all(shifted < 1.0)
    with pytest.raises(ValueError, match="positive"):
        apply_timestep_shift(t, 0.0)


def test_train_config_accepts_new_timestep_sampling_modes_and_rejects_bad_shift() -> None:
    for mode in ("uniform", "logit_normal", "shifted_logit_normal", "cosmap", "cosmap_like"):
        cfg = TrainConfig.from_dict({"flow": {"timestep_sampling": mode, "timestep_shift": 2.0}})
        assert cfg.flow_timestep_sampling == mode
        assert cfg.flow_timestep_shift == 2.0
    with pytest.raises(ValueError, match="flow_timestep_shift"):
        TrainConfig.from_dict({"flow": {"timestep_sampling": "shifted_logit_normal", "timestep_shift": 0.0}})
