from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_different_text_ids_change_unet_output() -> None:
    from model.unet.unet import UNet, UNetConfig

    torch.manual_seed(0)
    model = UNet(
        UNetConfig(
            image_channels=3,
            base_channels=8,
            channel_mults=(1,),
            num_res_blocks=1,
            dropout=0.0,
            attn_resolutions=(4,),
            attn_heads=1,
            attn_head_dim=8,
            vocab_size=16,
            text_dim=8,
            text_layers=1,
            text_heads=1,
            text_max_len=4,
            use_text_conditioning=True,
            self_conditioning=False,
        )
    )
    model.eval()

    x = torch.randn(1, 3, 4, 4)
    t = torch.tensor([1], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])
    ids_a = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    ids_b = torch.tensor([[1, 4, 5, 0]], dtype=torch.long)

    with torch.no_grad():
        out_a = model(x, t, ids_a, mask)
        out_b = model(x, t, ids_b, mask)

    assert not torch.allclose(out_a, out_b)


def test_projected_cross_attention_and_spatial_text_conditioning_forward() -> None:
    from model.unet.unet import UNet, UNetConfig

    torch.manual_seed(0)
    model = UNet(
        UNetConfig(
            image_channels=3,
            base_channels=8,
            channel_mults=(1,),
            num_res_blocks=1,
            dropout=0.0,
            attn_resolutions=(4,),
            attn_heads=1,
            attn_head_dim=8,
            vocab_size=16,
            text_dim=8,
            cross_attn_dim=12,
            text_layers=1,
            text_heads=1,
            text_max_len=4,
            text_spatial_conditioning=True,
            use_text_conditioning=True,
            self_conditioning=False,
        )
    )

    x = torch.randn(1, 3, 4, 4)
    t = torch.tensor([1], dtype=torch.long)
    ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    mask = torch.tensor([[True, True, True, False]])

    out = model(x, t, ids, mask)

    assert out.shape == x.shape


def test_attention_placement_can_limit_self_attention_to_mid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import model.unet.attention as attention_module
    from model.unet.unet import UNet, UNetConfig

    calls = {"count": 0}

    def fake_self_attention(self, x):
        calls["count"] += 1
        return x

    monkeypatch.setattr(attention_module.SelfAttention2d, "forward", fake_self_attention)

    model = UNet(
        UNetConfig(
            image_channels=3,
            base_channels=8,
            channel_mults=(1, 2),
            num_res_blocks=1,
            dropout=0.0,
            attn_resolutions=(4,),
            attn_heads=1,
            attn_head_dim=8,
            use_text_conditioning=False,
            self_conditioning=False,
            attention_placement="mid_only",
        )
    )

    out = model(torch.randn(1, 3, 4, 4), torch.tensor([1], dtype=torch.long), None, None)

    assert out.shape == (1, 3, 4, 4)
    assert calls["count"] == 1
