from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_text_encoder_sdpa_mask_uses_true_for_valid_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    import model.text.encoder as encoder_module
    from model.text.encoder import SDPATransformerBlock

    captured = {}
    expected = torch.tensor([[True, True, False], [True, False, False]])

    def fake_sdpa(q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False):
        captured["attn_mask"] = attn_mask.detach().clone()
        return torch.zeros_like(q)

    monkeypatch.setattr(encoder_module.F, "scaled_dot_product_attention", fake_sdpa)

    block = SDPATransformerBlock(dim=4, n_heads=2, dropout=0.0)
    block(torch.randn(2, 3, 4), expected)

    assert torch.equal(captured["attn_mask"], expected.unsqueeze(1).unsqueeze(2))


def test_cross_attention_sdpa_mask_uses_true_for_valid_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    import model.unet.attention as attention_module
    from model.unet.attention import CrossAttention2d

    captured = {}
    expected = torch.tensor([[True, True, False], [True, False, False]])

    def fake_sdpa(q, k, v, *, attn_mask=None, dropout_p=0.0, is_causal=False):
        captured["attn_mask"] = attn_mask.detach().clone()
        return torch.zeros_like(q)

    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", fake_sdpa)

    layer = CrossAttention2d(in_ch=4, ctx_dim=8, heads=2, head_dim=2)
    layer(torch.randn(2, 4, 2, 2), torch.randn(2, 3, 8), expected)

    assert torch.equal(captured["attn_mask"], expected.unsqueeze(1).unsqueeze(1))
