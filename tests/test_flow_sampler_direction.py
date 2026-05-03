from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from samplers.flow_euler import sample_flow_euler


class PerfectFlowModel:
    def __init__(self, x0: torch.Tensor, eps: torch.Tensor) -> None:
        self.v = eps - x0

    def __call__(self, x: torch.Tensor, t: torch.Tensor, cond: TextConditioning) -> torch.Tensor:
        assert x.shape == self.v.shape
        assert t.shape == (x.shape[0],)
        assert cond.tokens.shape[0] == x.shape[0]
        return self.v.to(device=x.device, dtype=x.dtype)


def test_euler_sampler_runs_from_eps_at_t1_to_x0_at_t0_for_perfect_flow() -> None:
    torch.manual_seed(5)
    x0 = torch.randn(2, 4, 4, 4)
    eps = torch.randn_like(x0)
    text = TextConditioning(
        tokens=torch.zeros(2, 1, 8),
        mask=torch.ones(2, 1, dtype=torch.bool),
        pooled=torch.zeros(2, 8),
    )
    model = PerfectFlowModel(x0, eps)

    out = sample_flow_euler(model, x0.shape, text, steps=8, cfg_scale=1.0, shift=1.0, noise=eps)

    assert out.shape == x0.shape
    assert torch.allclose(out, x0, atol=1.0e-6, rtol=1.0e-6)
