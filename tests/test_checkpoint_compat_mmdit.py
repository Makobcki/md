from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from diffusion.utils import EMA
from train.checkpoint_mmdit import validate_mmdit_checkpoint_compatibility
from train.loop_mmdit_full import _build_ckpt


def test_mmdit_checkpoint_compatibility_detects_mismatch() -> None:
    ckpt = {"architecture": "mmdit_rf", "cfg": {"hidden_dim": 64, "depth": 2}}
    validate_mmdit_checkpoint_compatibility(ckpt, {"architecture": "mmdit_rf", "hidden_dim": 64})
    with pytest.raises(RuntimeError):
        validate_mmdit_checkpoint_compatibility(ckpt, {"architecture": "mmdit_rf", "hidden_dim": 128})


def test_mmdit_checkpoint_uses_human_step(tmp_path) -> None:
    cfg = TrainConfig.from_dict(
        {
            "architecture": "mmdit_rf",
            "objective": "rectified_flow",
            "prediction_type": "flow_velocity",
            "hidden_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "double_stream_blocks": 1,
            "single_stream_blocks": 0,
            "text_dim": 16,
            "pooled_dim": 16,
        }
    )
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    ckpt = _build_ckpt(
        cfg=cfg,
        cfg_dict=cfg.to_dict(),
        model=model,
        optimizer=opt,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        ema=EMA(model),
        step=50,
        text_metadata={},
    )

    assert ckpt["step"] == 50


def test_mmdit_checkpoint_text_encoder_compat_ignores_cache_metadata() -> None:
    ckpt = {
        "architecture": "mmdit_rf",
        "objective": "rectified_flow",
        "text_encoders": [
            {
                "name": "t5",
                "model_name": "google/t5-v1_1-base",
                "max_length": 128,
                "trainable": False,
                "cache": True,
                "dtype": "bfloat16",
            }
        ],
    }
    cfg = {
        "architecture": "mmdit_rf",
        "objective": "rectified_flow",
        "text": {
            "encoders": [
                {
                    "name": "t5",
                    "model_name": "google/t5-v1_1-base",
                    "max_length": 128,
                    "trainable": False,
                    "cache": True,
                }
            ]
        },
    }

    validate_mmdit_checkpoint_compatibility(ckpt, cfg)

    cfg["text"]["encoders"][0]["max_length"] = 256
    with pytest.raises(RuntimeError, match="text_encoders differ"):
        validate_mmdit_checkpoint_compatibility(ckpt, cfg)


def test_mmdit_checkpoint_compatibility_detects_stage_a_mismatch() -> None:
    ckpt = {
        "metadata": {
            "architecture": "mmdit_rf",
            "objective": "rectified_flow",
            "prediction_type": "flow_velocity",
            "model_config": {
                "hidden_dim": 32,
                "depth": 4,
                "num_heads": 4,
                "double_stream_blocks": 2,
                "single_stream_blocks": 2,
                "text_resampler_enabled": True,
                "text_resampler_num_tokens": 8,
                "text_resampler_depth": 1,
                "text_resampler_mlp_ratio": 4.0,
                "attention_schedule": "hybrid",
                "early_joint_blocks": 1,
                "late_joint_blocks": 1,
                "x0_aux_weight": 0.05,
            },
            "text_config": {},
            "vae_config": {},
            "flow_config": {},
            "train_config_hash": "",
            "dataset_hash": "",
            "step": 0,
        }
    }
    cfg = {
        "architecture": "mmdit_rf",
        "objective": "rectified_flow",
        "hidden_dim": 32,
        "depth": 4,
        "num_heads": 4,
        "double_stream_blocks": 2,
        "single_stream_blocks": 2,
        "model": {
            "attention_schedule": "hybrid",
            "early_joint_blocks": 1,
            "late_joint_blocks": 1,
        },
        "text": {
            "resampler": {
                "enabled": True,
                "num_tokens": 8,
                "depth": 1,
                "mlp_ratio": 4.0,
            }
        },
        "loss": {"x0_aux_weight": 0.05},
    }
    validate_mmdit_checkpoint_compatibility(ckpt, cfg)

    changed = dict(cfg)
    changed["text"] = {"resampler": dict(cfg["text"]["resampler"], num_tokens=16)}
    with pytest.raises(RuntimeError, match="text_resampler_num_tokens mismatch"):
        validate_mmdit_checkpoint_compatibility(ckpt, changed)

    changed = dict(cfg)
    changed["model"] = dict(cfg["model"], attention_schedule="full")
    with pytest.raises(RuntimeError, match="attention_schedule mismatch"):
        validate_mmdit_checkpoint_compatibility(ckpt, changed)

    changed = dict(cfg)
    changed["loss"] = {"x0_aux_weight": 0.0}
    with pytest.raises(RuntimeError, match="x0_aux_weight mismatch"):
        validate_mmdit_checkpoint_compatibility(ckpt, changed)
