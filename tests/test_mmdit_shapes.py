from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning


def test_mmdit_forward_shape() -> None:
    cfg = MMDiTConfig(
        latent_channels=4,
        patch_size=2,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        double_stream_blocks=1,
        single_stream_blocks=1,
        text_dim=32,
        pooled_dim=32,
        gradient_checkpointing=False,
    )
    model = MMDiTFlowModel(cfg)
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    text = TextConditioning(
        tokens=torch.randn(2, 5, 32),
        mask=torch.ones(2, 5, dtype=torch.bool),
        pooled=torch.randn(2, 32),
    )
    out = model(x, t, text)
    assert out.shape == x.shape

