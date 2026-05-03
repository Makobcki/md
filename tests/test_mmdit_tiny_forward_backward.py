from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.objectives import RectifiedFlowObjective, rectified_flow_loss
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning


def test_mmdit_tiny_forward_backward_optimizer_step_without_external_dependencies() -> None:
    torch.manual_seed(7)
    cfg = MMDiTConfig(
        latent_channels=4,
        patch_size=2,
        hidden_dim=16,
        depth=1,
        num_heads=4,
        double_stream_blocks=1,
        single_stream_blocks=0,
        text_dim=8,
        pooled_dim=8,
        mlp_ratio=1.0,
        gradient_checkpointing=False,
        zero_init_final=False,
    )
    model = MMDiTFlowModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    objective = RectifiedFlowObjective(timestep_sampling="uniform")

    x0 = torch.randn(2, 4, 4, 4)
    text = TextConditioning(
        tokens=torch.randn(2, 3, 8),
        mask=torch.tensor([[True, True, False], [True, True, True]]),
        pooled=torch.randn(2, 8),
    )

    train_tuple = objective.sample_training_tuple(x0)
    pred = model(train_tuple.xt, train_tuple.t, text)
    loss = rectified_flow_loss(pred, train_tuple.target, train_tuple.weight)

    assert pred.shape == x0.shape
    assert torch.isfinite(loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
    optimizer.step()
