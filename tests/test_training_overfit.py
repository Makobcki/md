from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.objectives import RectifiedFlowObjective
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.loop_mmdit import training_step_mmdit


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

