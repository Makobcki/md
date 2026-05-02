from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from samplers.flow_euler import sample_flow_euler


class _ZeroFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x, t, text, **kwargs):
        del t, text, kwargs
        return torch.zeros_like(x) + self.p


def test_flow_sampler_same_seed_same_latent() -> None:
    model = _ZeroFlow()
    text = TextConditioning(torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 4))
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = sample_flow_euler(model, (1, 4, 4, 4), text, steps=3, generator=g1)
    b = sample_flow_euler(model, (1, 4, 4, 4), text, steps=3, generator=g2)
    assert torch.equal(a, b)

