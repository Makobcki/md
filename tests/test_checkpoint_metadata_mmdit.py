from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from diffusion.io.ckpt import save_ckpt
from diffusion.utils import EMA
from train.checkpoint_mmdit import read_checkpoint_metadata, validate_checkpoint_metadata
from train.loop_mmdit_full import _build_ckpt


def _cfg() -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "hidden_dim": 32,
            "depth": 1,
            "num_heads": 4,
            "double_stream_blocks": 1,
            "single_stream_blocks": 0,
            "text_dim": 8,
            "pooled_dim": 8,
            "latent_channels": 4,
            "latent_patch_size": 2,
            "vae_scaling_factor": 0.18215,
            "flow_timestep_sampling": "uniform",
        }
    )


def test_checkpoint_contains_self_describing_metadata(tmp_path: Path) -> None:
    cfg = _cfg()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt = _build_ckpt(
        cfg=cfg,
        cfg_dict={**cfg.to_dict(), "dataset_hash": "dataset-sha256"},
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        ema=EMA(model),
        step=7,
        text_metadata={"encoders": [{"name": "fake", "model_name": "fake", "max_length": 3}]},
    )

    metadata = ckpt["metadata"]
    validate_checkpoint_metadata(metadata)
    assert metadata["architecture"] == "mmdit_rf"
    assert metadata["objective"] == "rectified_flow"
    assert metadata["prediction_type"] == "flow_velocity"
    assert metadata["model_config"]["hidden_dim"] == 32
    assert metadata["vae_config"]["scaling_factor"] == pytest.approx(0.18215)
    assert metadata["flow_config"]["timestep_sampling"] == "uniform"
    assert metadata["dataset_hash"] == "dataset-sha256"
    assert metadata["step"] == 7

    path = tmp_path / "ckpt.pt"
    save_ckpt(str(path), ckpt)
    loaded_metadata = read_checkpoint_metadata(path)
    assert loaded_metadata == metadata


def test_checkpoint_metadata_missing_required_fields_is_clear() -> None:
    with pytest.raises(RuntimeError, match="missing required field"):
        validate_checkpoint_metadata({"architecture": "mmdit_rf"})
