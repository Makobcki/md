from __future__ import annotations

import pytest

from config.train import TrainConfig, TEXT_ENCODER_PRESETS
from train.checkpoint_mmdit import build_mmdit_checkpoint_metadata


def test_text_preset_expands_to_full_encoder_config() -> None:
    cfg = TrainConfig.from_dict(
        {
            "text_preset": "clip_l_t5_base",
            "hidden_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "double_stream_blocks": 1,
            "single_stream_blocks": 0,
        }
    )

    text = cfg.extra["text"]
    assert cfg.text_preset == "clip_l_t5_base"
    assert text["preset"] == "clip_l_t5_base"
    assert text["encoders"] == TEXT_ENCODER_PRESETS["clip_l_t5_base"]["encoders"]
    assert cfg.text_dim == 1024
    assert cfg.pooled_dim == 1024


def test_explicit_text_config_overrides_preset_encoders() -> None:
    custom = [{"name": "fake", "model_name": "fake", "max_length": 5, "trainable": False, "cache": True}]
    cfg = TrainConfig.from_dict(
        {
            "text_preset": "clip_l_t5_large",
            "text": {"encoders": custom, "text_dim": 8, "pooled_dim": 8},
            "hidden_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "double_stream_blocks": 1,
            "single_stream_blocks": 0,
        }
    )

    assert cfg.text_dim == 8
    assert cfg.pooled_dim == 8
    assert cfg.extra["text"]["encoders"] == custom


def test_invalid_text_preset_fails_early() -> None:
    with pytest.raises(ValueError, match="Unsupported text_preset"):
        TrainConfig.from_dict({"text_preset": "t5_xxl_now", "hidden_dim": 32, "depth": 1, "num_heads": 4, "double_stream_blocks": 1, "single_stream_blocks": 0})


def test_checkpoint_metadata_contains_expanded_text_preset_encoders() -> None:
    cfg = TrainConfig.from_dict(
        {
            "text_preset": "clip_l_t5_base",
            "hidden_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "double_stream_blocks": 1,
            "single_stream_blocks": 0,
        }
    )
    meta = build_mmdit_checkpoint_metadata(
        cfg=cfg,
        cfg_dict=cfg.to_dict(),
        step=3,
        text_metadata={"encoders": cfg.extra["text"]["encoders"]},
        dataset_hash="abc",
    )

    assert meta["text_config"]["encoders"] == TEXT_ENCODER_PRESETS["clip_l_t5_base"]["encoders"]
    assert meta["text_config"]["max_length_total"] == 333
