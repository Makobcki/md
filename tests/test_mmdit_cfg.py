from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from samplers.cfg import cfg_predict


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t, text, **kwargs):
        del t, kwargs
        return x + text.pooled.mean(dim=1).view(-1, 1, 1, 1)


def test_cfg_predict_batched() -> None:
    model = _ToyModel()
    x = torch.zeros(1, 1, 2, 2)
    t = torch.ones(1)
    cond = TextConditioning(torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 2))
    uncond = TextConditioning(torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 2))
    out = cfg_predict(model, x, t, cond, uncond, scale=2.0)
    assert torch.allclose(out, torch.full_like(x, 2.0))

