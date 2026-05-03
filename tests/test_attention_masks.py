from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_joint_attention_sdpa_mask_uses_true_for_valid_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    import model.mmdit.attention as attention_module
    from model.mmdit.attention import JointAttention

    captured = {}

    def fake_sdpa(q, k, v, attn_mask=None, dropout_p=0.0):
        captured["attn_mask"] = attn_mask.detach().cpu() if attn_mask is not None else None
        captured["dropout_p"] = dropout_p
        return torch.zeros_like(q)

    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", fake_sdpa)
    attn = JointAttention(hidden_dim=16, num_heads=4, qk_norm=False)
    q = torch.randn(2, 5, 16)
    k = torch.randn(2, 5, 16)
    v = torch.randn(2, 5, 16)
    valid_mask = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])

    out = attn(q, k, v, valid_mask)

    assert out.shape == q.shape
    assert captured["attn_mask"].shape == (2, 1, 1, 5)
    assert captured["attn_mask"].dtype == torch.bool
    assert torch.equal(captured["attn_mask"][:, 0, 0, :], valid_mask)


def test_joint_attention_preserves_shape_without_text_mask() -> None:
    from model.mmdit.attention import JointAttention

    attn = JointAttention(hidden_dim=32, num_heads=4)
    x = torch.randn(2, 7, 32)

    out = attn(x, x, x, mask=None)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_joint_attention_rope_only_applies_to_image_token_slice() -> None:
    from model.mmdit.attention import JointAttention

    attn = JointAttention(hidden_dim=32, num_heads=4)
    q = torch.randn(1, 6, 32)
    k = torch.randn(1, 6, 32)
    v = torch.randn(1, 6, 32)

    out = attn(q, k, v, rope_grid_hw=(2, 2), rope_start=2, rope_length=4)

    assert out.shape == q.shape
    assert torch.isfinite(out).all()
