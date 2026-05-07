from __future__ import annotations

import pytest

from config.train import TrainConfig
from model.registry import (
    build_model_capabilities,
    build_model_spec,
    build_runtime_contract,
    get_allowed_families,
    model_family,
)


def test_legacy_config_defaults_to_mmdit_family() -> None:
    cfg = TrainConfig.from_dict({"architecture": "mmdit_rf"})

    assert cfg.model_family == "mmdit"
    assert model_family(cfg.to_dict()) == "mmdit"
    assert build_model_spec(cfg).family == "mmdit"
    assert build_runtime_contract(cfg).sampler_kind == "latent_flow"


def test_unknown_family_lists_allowed_families() -> None:
    with pytest.raises(ValueError, match="Allowed: mmdit, pixart_sigma, var"):
        TrainConfig.from_dict({"model": {"family": "bad_family"}})


def test_pixart_contract_rejects_tokenizer_and_unsupported_dimensions() -> None:
    data = {
        "model": {
            "family": "pixart_sigma",
            "variant": "tiny",
            "architecture": {
                "image_size": 32,
                "latent_size": 5,
                "latent_channels": 4,
                "patch_size": 2,
                "hidden_size": 32,
                "depth": 1,
                "num_heads": 4,
            },
            "diffusion": {"objective": "rectified_flow", "prediction_type": "velocity"},
            "tokenizer": {"kind": "vq"},
        },
        "vae": {"downsample_factor": 8, "latent_channels": 4},
    }

    with pytest.raises(ValueError, match="tokenizer"):
        TrainConfig.from_dict(data)

    data["model"].pop("tokenizer")
    with pytest.raises(ValueError, match="latent_size"):
        TrainConfig.from_dict(data)


def test_var_contract_rejects_diffusion_and_flow_config() -> None:
    data = {
        "model": {
            "family": "var",
            "variant": "tiny",
            "architecture": {"scale_schedule": [1, 2], "max_token_length": 5},
            "diffusion": {"objective": "rectified_flow"},
            "tokenizer": {"kind": "vq", "codebook_size": 16, "codebook_dim": 8},
            "autoregressive": {
                "objective": "next_scale_prediction",
                "prediction_type": "token_logits",
                "loss": "cross_entropy",
            },
        },
    }

    with pytest.raises(ValueError, match="diffusion"):
        TrainConfig.from_dict(data)


def test_contracts_expose_capabilities_without_building_models() -> None:
    assert get_allowed_families() == ("mmdit", "pixart_sigma", "var")
    pixart = build_model_capabilities({"model": {"family": "pixart_sigma"}})
    var_contract = build_runtime_contract({"model": {"family": "var"}})

    assert pixart.text_to_image is True
    assert pixart.image_to_image is False
    assert var_contract.requires_tokenizer is True
    assert var_contract.objective == "next_scale_prediction"
