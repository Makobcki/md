from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.objectives import RectifiedFlowObjective, rectified_flow_loss
from model.text.conditioning import TextConditioning


class _AffineTinyModel(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: TextConditioning) -> torch.Tensor:
        assert t.shape == (x.shape[0],)
        assert cond.tokens.shape[0] == x.shape[0]
        return x * self.scale + self.bias


def test_rectified_flow_objective_formula_shapes_dtype_and_backward() -> None:
    torch.manual_seed(123)
    x0 = torch.randn(3, 4, 5, 7, dtype=torch.float64)
    objective = RectifiedFlowObjective(timestep_sampling="uniform")

    torch.manual_seed(999)
    tup = objective.sample_training_tuple(x0)

    torch.manual_seed(999)
    eps = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    t_view = t.view(x0.shape[0], 1, 1, 1).to(dtype=x0.dtype)
    expected_xt = (1.0 - t_view) * x0 + t_view * eps
    expected_target = eps - x0

    assert tup.xt.shape == x0.shape
    assert tup.target.shape == x0.shape
    assert tup.t.shape == (x0.shape[0],)
    assert tup.weight.shape == (x0.shape[0],)
    assert tup.xt.dtype == x0.dtype
    assert tup.target.dtype == x0.dtype
    assert tup.weight.dtype == torch.float32
    assert torch.allclose(tup.t, t)
    assert torch.allclose(tup.xt, expected_xt)
    assert torch.allclose(tup.target, expected_target)

    cond = TextConditioning(
        tokens=torch.zeros(x0.shape[0], 2, 8, dtype=x0.dtype),
        mask=torch.ones(x0.shape[0], 2, dtype=torch.bool),
        pooled=torch.zeros(x0.shape[0], 8, dtype=x0.dtype),
    )
    model = _AffineTinyModel(channels=x0.shape[1]).to(dtype=x0.dtype)
    pred = model(tup.xt, tup.t, cond)
    loss = rectified_flow_loss(pred, tup.target, tup.weight)

    assert pred.shape == x0.shape
    assert pred.dtype == x0.dtype
    assert torch.isfinite(loss)

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
