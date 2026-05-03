from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from samplers.flow_euler import sample_flow_euler
from samplers.flow_heun import sample_flow_heun


class _ConstantVelocity(torch.nn.Module):
    def __init__(self, value: float = 0.25) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.value = float(value)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: TextConditioning) -> torch.Tensor:
        del t, text
        return torch.full_like(x, self.value) + self.anchor


class _NonConstantVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: TextConditioning) -> torch.Tensor:
        return x * (1.0 + t.view(-1, 1, 1, 1).to(dtype=x.dtype)) + self.anchor


def _text(batch: int = 2) -> TextConditioning:
    return TextConditioning(
        tokens=torch.zeros(batch, 1, 4),
        mask=torch.ones(batch, 1, dtype=torch.bool),
        pooled=torch.zeros(batch, 4),
    )


def test_heun_sampler_is_deterministic_and_finite_with_matching_shape() -> None:
    model = _NonConstantVelocity()
    shape = (2, 4, 4, 4)
    text = _text(shape[0])
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    out1 = sample_flow_heun(model, shape, text, steps=4, cfg_scale=1.0, generator=g1)
    out2 = sample_flow_heun(model, shape, text, steps=4, cfg_scale=1.0, generator=g2)

    assert out1.shape == shape
    assert torch.isfinite(out1).all()
    assert torch.equal(out1, out2)


def test_heun_differs_from_euler_for_non_constant_velocity() -> None:
    model = _NonConstantVelocity()
    shape = (2, 4, 4, 4)
    text = _text(shape[0])
    noise = torch.randn(shape)

    euler = sample_flow_euler(model, shape, text, steps=3, cfg_scale=1.0, shift=1.0, noise=noise)
    heun = sample_flow_heun(model, shape, text, steps=3, cfg_scale=1.0, shift=1.0, noise=noise)

    assert not torch.allclose(euler, heun)


def test_heun_matches_euler_for_constant_velocity() -> None:
    model = _ConstantVelocity(value=-0.75)
    shape = (2, 4, 4, 4)
    text = _text(shape[0])
    noise = torch.randn(shape)

    euler = sample_flow_euler(model, shape, text, steps=7, cfg_scale=1.0, shift=1.0, noise=noise)
    heun = sample_flow_heun(model, shape, text, steps=7, cfg_scale=1.0, shift=1.0, noise=noise)

    assert torch.allclose(euler, heun, atol=1.0e-6, rtol=1.0e-6)
