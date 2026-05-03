from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
import train.loop_mmdit_full as loop


def _text(batch: int = 2) -> TextConditioning:
    return TextConditioning(
        tokens=torch.arange(batch * 2 * 3, dtype=torch.float32).view(batch, 2, 3),
        mask=torch.ones(batch, 2, dtype=torch.bool),
        pooled=torch.arange(batch * 3, dtype=torch.float32).view(batch, 3),
        is_uncond=torch.zeros(batch, dtype=torch.bool),
    )


def _empty(batch: int = 2) -> TextConditioning:
    return TextConditioning(
        tokens=torch.full((batch, 2, 3), -1.0),
        mask=torch.zeros(batch, 2, dtype=torch.bool),
        pooled=torch.full((batch, 3), -2.0),
        is_uncond=torch.ones(batch, dtype=torch.bool),
    )


def test_cfg_dropout_replaces_dropped_samples_with_empty_conditioning(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _text()
    empty = _empty()
    masks = iter([
        torch.tensor([0.2, 0.8]),  # drop first only for drop_prob=0.5
        torch.tensor([0.8, 0.2]),  # drop second only
    ])

    def fake_rand(n: int, *, device=None):
        return next(masks).to(device=device)

    monkeypatch.setattr(loop.torch, "rand", fake_rand)
    out1 = loop._replace_text_where(text, empty, 0.5)
    out2 = loop._replace_text_where(text, empty, 0.5)

    assert torch.equal(out1.tokens[0], empty.tokens[0])
    assert torch.equal(out1.tokens[1], text.tokens[1])
    assert out1.is_uncond.tolist() == [True, False]
    assert torch.equal(out2.tokens[0], text.tokens[0])
    assert torch.equal(out2.tokens[1], empty.tokens[1])
    assert out2.is_uncond.tolist() == [False, True]
    assert not torch.equal(out1.tokens, out2.tokens)


def test_cfg_dropout_zero_probability_leaves_conditioning_unchanged() -> None:
    text = _text()
    empty = _empty()
    out = loop._replace_text_where(text, empty, 0.0)
    assert out is text


def test_inpaint_weighted_mse_keeps_txt2img_loss_unchanged_and_weights_mask_regions() -> None:
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    full = loop._per_sample_flow_mse(pred, target, None)
    weighted = loop._per_sample_flow_mse(pred, target, mask, mask_weight=1.0, unmask_weight=0.1)
    all_equal_weight = loop._per_sample_flow_mse(pred, target, mask, mask_weight=1.0, unmask_weight=1.0)

    assert torch.allclose(full, torch.tensor([21.0]))
    expected = torch.tensor([(1.0 + 0.9 + 2.5 + 4.9) / 1.3])
    assert torch.allclose(weighted, expected)
    assert torch.allclose(all_equal_weight, full)


def test_grad_diagnostics_reports_nan_and_inf_without_crashing() -> None:
    model = torch.nn.Linear(2, 1)
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    model.weight.grad[0, 0] = float("nan")
    model.bias.grad[0] = float("inf")

    stats = loop._grad_diagnostics(model)

    assert stats["has_nan_grad"] is True
    assert stats["has_inf_grad"] is True
    assert stats["grad_norm_total"] >= 0.0
