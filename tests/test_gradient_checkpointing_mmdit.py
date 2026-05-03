from __future__ import annotations

import torch

import model.mmdit.model as mmdit_model
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning


def _text(batch: int = 2) -> TextConditioning:
    return TextConditioning(
        tokens=torch.randn(batch, 3, 16),
        mask=torch.ones(batch, 3, dtype=torch.bool),
        pooled=torch.randn(batch, 16),
    )


def test_gradient_checkpointing_wraps_each_transformer_block_not_final(monkeypatch) -> None:
    calls: list[str] = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(function.__class__.__name__)
        return function(*args)

    monkeypatch.setattr(mmdit_model, "checkpoint", fake_checkpoint)
    cfg = MMDiTConfig(
        hidden_dim=32,
        depth=2,
        num_heads=4,
        double_stream_blocks=1,
        single_stream_blocks=1,
        text_dim=16,
        pooled_dim=16,
        gradient_checkpointing=True,
    )
    model = MMDiTFlowModel(cfg).train()
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    out = model(x, t, _text(2))
    out.square().mean().backward()
    assert calls == ["MMDiTDoubleBlock", "MMDiTSingleBlock"]


def test_gradient_checkpointing_disabled_in_eval(monkeypatch) -> None:
    calls: list[str] = []

    def fake_checkpoint(function, *args, **kwargs):
        calls.append(function.__class__.__name__)
        return function(*args)

    monkeypatch.setattr(mmdit_model, "checkpoint", fake_checkpoint)
    cfg = MMDiTConfig(
        hidden_dim=32,
        depth=1,
        num_heads=4,
        double_stream_blocks=1,
        single_stream_blocks=0,
        text_dim=16,
        pooled_dim=16,
        gradient_checkpointing=True,
    )
    model = MMDiTFlowModel(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 8, 8), torch.rand(1), _text(1))
    assert out.shape == (1, 4, 8, 8)
    assert calls == []
