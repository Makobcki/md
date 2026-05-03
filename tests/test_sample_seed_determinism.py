from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from samplers.flow_euler import sample_flow_euler


class _ZeroFlow(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, t: torch.Tensor, text: TextConditioning) -> torch.Tensor:
        del t, text
        return torch.zeros_like(x) + self.anchor


def test_latent_sampling_seed_determinism_without_vae() -> None:
    model = _ZeroFlow()
    shape = (1, 4, 4, 4)
    text = TextConditioning(
        tokens=torch.zeros(1, 1, 4),
        mask=torch.ones(1, 1, dtype=torch.bool),
        pooled=torch.zeros(1, 4),
    )

    out1 = sample_flow_euler(model, shape, text, steps=3, cfg_scale=1.0, generator=torch.Generator().manual_seed(42))
    out2 = sample_flow_euler(model, shape, text, steps=3, cfg_scale=1.0, generator=torch.Generator().manual_seed(42))
    out3 = sample_flow_euler(model, shape, text, steps=3, cfg_scale=1.0, generator=torch.Generator().manual_seed(43))

    assert torch.equal(out1, out2)
    assert not torch.equal(out1, out3)
