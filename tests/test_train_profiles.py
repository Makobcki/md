from __future__ import annotations

import pytest

from config.loader import load_train_config
from config.train import TrainConfig


def test_main_train_config_is_mmdit_rectified_flow() -> None:
    cfg = load_train_config()

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


def test_supported_kdl_presets_use_mmdit_rf() -> None:
    for overrides in (
        {},
        {"model": {"preset": "mmdit_1024"}},
        {"training": {"preset": "single_gpu_debug"}},
    ):
        cfg = load_train_config(overrides=overrides)
        assert cfg.architecture == "mmdit_rf"
        assert cfg.objective == "rectified_flow"
        assert cfg.mode == "latent"
        assert cfg.sampling_sampler in {"flow_euler", "flow_heun"}
        assert cfg.eval_sampler in {"flow_euler", "flow_heun"}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_dim": 10, "num_heads": 4}, "hidden_dim must be divisible by num_heads"),
        ({"image_size": 500, "latent_downsample_factor": 8}, "image_size must be divisible"),
        ({"image_size": 520, "latent_downsample_factor": 8, "latent_patch_size": 4}, "latent side must be divisible"),
        ({"depth": 3, "double_stream_blocks": 1, "single_stream_blocks": 1}, "must equal depth"),
        ({"dataset_tasks": {"txt2img": 1.0, "bad_task": 0.1}}, "unsupported task"),
        ({"dataset_tasks": {"txt2img": -1.0}}, "weights must be non-negative"),
        ({"source_patch_size": 5}, "source_patch_size"),
        ({"hidden_dim": 30, "num_heads": 5, "pos_embed": "sincos_2d"}, "sincos_2d requires hidden_dim divisible by 4"),
        ({"sampling_sampler": "bad_sampler"}, "sampling_sampler must be one of"),
        ({"amp_dtype": "fp8"}, "amp_dtype must be"),
        ({"latent_dtype": "fp32"}, "latent_dtype must be"),
        ({"text_cache": False, "allow_on_the_fly_text": False}, "text_cache=false"),
    ],
)
def test_invalid_configs_fail_early_with_clear_errors(override: dict, message: str) -> None:
    data = load_train_config(overrides={"training": {"preset": "single_gpu_debug"}}).to_dict()
    data.update(override)
    with pytest.raises(ValueError, match=message):
        TrainConfig.from_dict(data)


def test_text_cache_false_can_be_explicitly_allowed_for_debug_only() -> None:
    data = load_train_config(overrides={"training": {"preset": "single_gpu_debug"}}).to_dict()
    data.update({"text_cache": False, "allow_on_the_fly_text": True})
    cfg = TrainConfig.from_dict(data)
    assert cfg.text_cache is False
    assert cfg.allow_on_the_fly_text is True
