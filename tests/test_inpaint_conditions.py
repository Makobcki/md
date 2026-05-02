from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning


def test_mmdit_accepts_inpaint_mask() -> None:
    cfg = MMDiTConfig(hidden_dim=32, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0, text_dim=16, pooled_dim=16, gradient_checkpointing=False)
    model = MMDiTFlowModel(cfg)
    x = torch.randn(1, 4, 8, 8)
    text = TextConditioning(torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 16))
    out = model(x, torch.rand(1), text, source_latent=torch.randn_like(x), mask=torch.ones(1, 1, 8, 8), task="inpaint")
    assert out.shape == x.shape

