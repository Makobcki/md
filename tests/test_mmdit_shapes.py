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


def test_mmdit_pos_embed_default_is_sincos_and_rope_is_attention_applied() -> None:
    assert MMDiTConfig(hidden_dim=64, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0).pos_embed == "rope_2d"
    tokens = torch.zeros(1, 4, 64)
    assert torch.equal(add_2d_pos_embed(tokens, (2, 2), "rope_2d"), tokens)


def test_mmdit_forward_shape_with_rope_2d_and_img_conditions() -> None:
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
        pos_embed="rope_2d",
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
    out = model(x, t, text, source_latent=torch.randn_like(x), mask=torch.ones(2, 1, 8, 8), task="inpaint")
    assert out.shape == x.shape


def test_mmdit_forward_shape_with_control_latents_and_task_list() -> None:
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
        pos_embed="rope_2d",
        gradient_checkpointing=False,
    )
    model = MMDiTFlowModel(cfg)
    x = torch.randn(2, 4, 8, 8)
    text = TextConditioning(
        tokens=torch.randn(2, 5, 32),
        mask=torch.ones(2, 5, dtype=torch.bool),
        pooled=torch.randn(2, 32),
    )
    controls = torch.randn(2, 2, 4, 8, 8)
    out = model(x, torch.rand(2), text, control_latents=controls, task=["txt2img", "control"])
    assert out.shape == x.shape


def test_mmdit_rejects_unknown_task() -> None:
    cfg = MMDiTConfig(hidden_dim=32, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0, text_dim=16, pooled_dim=16, gradient_checkpointing=False)
    model = MMDiTFlowModel(cfg)
    x = torch.randn(1, 4, 8, 8)
    text = TextConditioning(torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 16))
    with pytest.raises(ValueError, match="Unsupported MMDiT task"):
        model(x, torch.rand(1), text, task="bad_task")
