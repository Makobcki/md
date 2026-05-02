from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import EMA
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.loop_mmdit import training_step_mmdit
from train.loop_mmdit_full import run_mmdit_training_loop


def test_mmdit_training_step_backward_tiny_batch() -> None:
    cfg = MMDiTConfig(hidden_dim=32, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0, text_dim=16, pooled_dim=16, gradient_checkpointing=False)
    model = MMDiTFlowModel(cfg)
    batch = TrainBatch(
        x0=torch.randn(2, 4, 8, 8),
        text=TextConditioning(torch.randn(2, 2, 16), torch.ones(2, 2, dtype=torch.bool), torch.randn(2, 16)),
    )
    loss = training_step_mmdit(model=model, objective=RectifiedFlowObjective(), batch=batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_mmdit_full_loop_rejects_empty_dataloader(tmp_path) -> None:
    cfg = MMDiTConfig(
        hidden_dim=32,
        depth=1,
        num_heads=4,
        double_stream_blocks=1,
        single_stream_blocks=0,
        text_dim=16,
        pooled_dim=16,
        gradient_checkpointing=False,
    )
    model = MMDiTFlowModel(cfg)
    train_cfg = TrainConfig.from_dict(
        {
            "architecture": "mmdit_rf",
            "objective": "rectified_flow",
            "prediction_type": "flow_velocity",
            "max_steps": 1,
            "depth": 1,
            "double_stream_blocks": 1,
            "single_stream_blocks": 0,
            "hidden_dim": 32,
            "num_heads": 4,
            "text_dim": 16,
            "pooled_dim": 16,
        }
    )
    with pytest.raises(RuntimeError, match="dataloader is empty"):
        run_mmdit_training_loop(
            cfg=train_cfg,
            cfg_dict=train_cfg.to_dict(),
            model=model,
            dataloader=[],
            val_dataloader=None,
            objective=RectifiedFlowObjective(),
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            ema=EMA(model),
            device=torch.device("cpu"),
            out_dir=tmp_path,
            empty_text=TextConditioning(
                torch.zeros(1, 1, 16),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 16),
            ),
            start_step=0,
            text_metadata={},
        )
