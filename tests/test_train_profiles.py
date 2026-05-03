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


def test_unsupported_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Only architecture=mmdit_rf is supported"):
        TrainConfig.from_dict({"architecture": "unsupported_architecture"})


def test_supported_profiles_use_mmdit_rf() -> None:
    for name in ("train.yaml", "train_dev.yaml", "train_overfit.yaml", "train_smoke.yaml"):
        cfg = _load_profile(name)
        assert cfg.architecture == "mmdit_rf"
        assert cfg.objective == "rectified_flow"
        assert cfg.mode == "latent"
        assert cfg.sampling_sampler in {"flow_euler", "flow_heun"}
        assert cfg.eval_sampler in {"flow_euler", "flow_heun"}


def test_base_profile_exists_and_train_yaml_is_alias_compatible() -> None:
    base = _load_profile("train_base.yaml")
    alias = _load_profile("train.yaml")

    assert base.architecture == "mmdit_rf"
    assert base.objective == "rectified_flow"
    assert base.prediction_type == "flow_velocity"
    assert alias.hidden_dim == base.hidden_dim
    assert alias.depth == base.depth
    assert alias.num_heads == base.num_heads


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_dim": 10, "num_heads": 4}, "hidden_dim must be divisible by num_heads"),
        ({"image_size": 500, "latent_downsample_factor": 8}, "image_size must be divisible"),
        ({"image_size": 520, "latent_downsample_factor": 8, "latent_patch_size": 4}, "latent side must be divisible"),
        ({"depth": 3, "double_stream_blocks": 1, "single_stream_blocks": 1}, "must equal depth"),
        ({"dataset_tasks": {"txt2img": 1.0, "bad_task": 0.1}}, "unsupported task"),
        ({"dataset_tasks": {"txt2img": -1.0}}, "weights must be non-negative"),
        ({"sampling_sampler": "bad_sampler"}, "sampling_sampler must be one of"),
        ({"amp_dtype": "fp8"}, "amp_dtype must be"),
        ({"latent_dtype": "fp32"}, "latent_dtype must be"),
        ({"text_cache": False, "allow_on_the_fly_text": False}, "text_cache=false"),
    ],
)
def test_invalid_configs_fail_early_with_clear_errors(override: dict, message: str) -> None:
    data = _load_profile("train_smoke.yaml").to_dict()
    data.update(override)
    with pytest.raises(ValueError, match=message):
        TrainConfig.from_dict(data)


def test_text_cache_false_can_be_explicitly_allowed_for_debug_only() -> None:
    data = _load_profile("train_smoke.yaml").to_dict()
    data.update({"text_cache": False, "allow_on_the_fly_text": True})
    cfg = TrainConfig.from_dict(data)
    assert cfg.text_cache is False
    assert cfg.allow_on_the_fly_text is True
