from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from diffusion.utils import EMA
from model.mmdit import MMDiTConfig, MMDiTFlowModel
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
    model = MMDiTFlowModel(
        MMDiTConfig(
            hidden_dim=32,
            depth=1,
            num_heads=4,
            double_stream_blocks=1,
            single_stream_blocks=0,
            text_dim=16,
            pooled_dim=16,
        )
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
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
