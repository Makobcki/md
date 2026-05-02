from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.mmdit.pos_embed import add_2d_pos_embed
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


def test_mmdit_pos_embed_default_is_sincos_and_rope_is_explicitly_unimplemented() -> None:
    assert MMDiTConfig(hidden_dim=64, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0).pos_embed == "sincos_2d"
    with pytest.raises(NotImplementedError, match="q/k inside attention"):
        add_2d_pos_embed(torch.zeros(1, 4, 64), (2, 2), "rope_2d")
