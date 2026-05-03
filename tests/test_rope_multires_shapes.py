from __future__ import annotations

import torch

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning


def test_rope_2d_supports_multiple_latent_grids() -> None:
    cfg = MMDiTConfig(
        hidden_dim=32,
        depth=1,
        num_heads=4,
        double_stream_blocks=1,
        single_stream_blocks=0,
        text_dim=16,
        pooled_dim=16,
        gradient_checkpointing=False,
        pos_embed="rope_2d",
    )
    model = MMDiTFlowModel(cfg)
    text = TextConditioning(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool), torch.randn(1, 16))
    for h, w in [(64, 64), (64, 96), (96, 64), (128, 128)]:
        x = torch.randn(1, 4, h, w)
        out = model(x, torch.rand(1), text)
        assert out.shape == x.shape
