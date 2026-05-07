from __future__ import annotations

import pytest

from train.checkpoint_metadata import (
    build_model_checkpoint_metadata,
    validate_checkpoint_compatibility,
)


def test_checkpoint_metadata_v2_rejects_cross_family() -> None:
    cfg = {
        "model": {
            "family": "pixart_sigma",
            "variant": "tiny",
            "architecture": {"latent_channels": 4, "hidden_size": 32},
            "diffusion": {"objective": "rectified_flow", "prediction_type": "velocity"},
        }
    }
    meta = build_model_checkpoint_metadata("pixart_sigma", cfg, training_state={"global_step": 2})

    assert meta["checkpoint"]["metadata_version"] == 2
    assert meta["checkpoint"]["model"]["family"] == "pixart_sigma"
    validate_checkpoint_compatibility("pixart_sigma", {"metadata": meta}, cfg)
    with pytest.raises(RuntimeError, match="family"):
        validate_checkpoint_compatibility("var", {"metadata": meta}, cfg)


def test_var_checkpoint_metadata_contains_tokenizer_shape() -> None:
    cfg = {
        "model": {
            "family": "var",
            "variant": "tiny",
            "architecture": {"scale_schedule": [1, 2], "max_token_length": 5},
            "tokenizer": {"kind": "vq", "codebook_size": 16, "codebook_dim": 8},
            "autoregressive": {
                "objective": "next_scale_prediction",
                "prediction_type": "token_logits",
                "loss": "cross_entropy",
            },
        }
    }

    meta = build_model_checkpoint_metadata("var", cfg)

    assert meta["checkpoint"]["tokenizer_config"]["codebook_size"] == 16
    assert meta["checkpoint"]["tokenizer_config"]["scale_schedule"] == [1, 2]
