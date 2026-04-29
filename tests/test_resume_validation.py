from __future__ import annotations

import pytest

pytest.importorskip("torch")

from config.train import TrainConfig
from train.runner import _validate_resume_compatibility


def test_resume_validation_rejects_mode_mismatch() -> None:
    cfg = TrainConfig(mode="latent", latent_cache=True)
    ckpt_cfg = TrainConfig(mode="pixel")

    with pytest.raises(RuntimeError, match="resume config mismatch"):
        _validate_resume_compatibility(cfg, ckpt_cfg)
