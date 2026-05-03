from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass


from diffusion.objectives import RectifiedFlowObjective
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.loop_mmdit import training_step_mmdit


def test_mmdit_tiny_model_can_overfit_fixed_latents_and_text() -> None:
    torch.manual_seed(11)
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
    objective = RectifiedFlowObjective(timestep_sampling="uniform")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2, weight_decay=0.0)

    batch = TrainBatch(
        x0=torch.randn(2, 4, 4, 4),
        text=TextConditioning(
            tokens=torch.randn(2, 1, 8),
            mask=torch.ones(2, 1, dtype=torch.bool),
            pooled=torch.randn(2, 8),
        ),
    )

    initial_loss: float | None = None
    losses: list[float] = []
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = training_step_mmdit(model=model, objective=objective, batch=batch, amp_enabled=False)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
        optimizer.step()
        losses.append(float(loss.detach()))

    assert initial_loss is not None
    final_loss = sum(losses[-10:]) / 10.0
    assert final_loss < initial_loss * 0.5
