from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from samplers.cfg import cfg_predict


class _CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, x, t, text, **kwargs):
        self.calls += 1
        source = kwargs.get("source_latent")
        source_term = 0.0 if source is None else source.mean(dim=(1, 2, 3)).view(-1, 1, 1, 1)
        return x + t.view(-1, 1, 1, 1) + text.pooled.mean(dim=1).view(-1, 1, 1, 1) + source_term + self.anchor


def _text(pooled_value: float, batch: int = 2) -> TextConditioning:
    return TextConditioning(
        tokens=torch.full((batch, 3, 4), pooled_value),
        mask=torch.ones(batch, 3, dtype=torch.bool),
        pooled=torch.full((batch, 4), pooled_value),
    )


def test_batched_cfg_matches_two_pass_cfg_and_uses_one_forward() -> None:
    x = torch.randn(2, 1, 3, 3)
    t = torch.tensor([1.0, 0.5])
    source = torch.randn_like(x)
    cond = _text(2.0)
    uncond = _text(-1.0)
    scale = 1.75

    reference_model = _CountingModel()
    pred_uncond = reference_model(x, t, uncond, source_latent=source)
    pred_cond = reference_model(x, t, cond, source_latent=source)
    expected = pred_uncond + scale * (pred_cond - pred_uncond)
    assert reference_model.calls == 2

    batched_model = _CountingModel()
    actual = cfg_predict(batched_model, x, t, cond, uncond, scale=scale, source_latent=source)

    assert batched_model.calls == 1
    assert torch.allclose(actual, expected)


def test_cfg_scale_one_uses_cond_only() -> None:
    model = _CountingModel()
    x = torch.zeros(2, 1, 2, 2)
    t = torch.zeros(2)
    out = cfg_predict(model, x, t, _text(3.0), _text(-10.0), scale=1.0)
    assert model.calls == 1
    assert torch.allclose(out, torch.full_like(x, 3.0))
