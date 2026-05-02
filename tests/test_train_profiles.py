from __future__ import annotations

from pathlib import Path

import pytest

from config.train import TrainConfig


ROOT = Path(__file__).resolve().parents[1]


def _load_profile(name: str) -> TrainConfig:
    return TrainConfig.from_yaml(str(ROOT / "config" / name))


def test_main_profile_is_mmdit_rectified_flow() -> None:
    cfg = _load_profile("train.yaml")

    assert cfg.architecture == "mmdit_rf"
    assert cfg.objective == "rectified_flow"
    assert cfg.prediction_type == "flow_velocity"
    assert cfg.mode == "latent"
    assert cfg.eval_sampler == "flow_heun"
    assert cfg.latent_cache is True
    assert cfg.text_cache is True
    assert cfg.cache_auto_prepare is True
    assert cfg.cache_rebuild_if_stale is False


def test_legacy_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Only architecture=mmdit_rf is supported"):
        TrainConfig.from_dict({"architecture": "unet" + "_v1"})


def test_supported_profiles_use_mmdit_rf() -> None:
    for name in ("train.yaml", "train_dev.yaml", "train_overfit.yaml", "train_smoke.yaml"):
        cfg = _load_profile(name)
        assert cfg.architecture == "mmdit_rf"
        assert cfg.objective == "rectified_flow"
        assert cfg.mode == "latent"
        assert cfg.sampling_sampler in {"flow_euler", "flow_heun"}
        assert cfg.eval_sampler in {"flow_euler", "flow_heun"}
