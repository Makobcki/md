from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_model_cost_profile_counts_attention_sites() -> None:
    from diffusion.perf import build_model_cost_profile
    from model.unet.unet import UNet, UNetConfig

    cfg = UNetConfig(
        image_channels=4,
        base_channels=8,
        channel_mults=(1, 2),
        num_res_blocks=1,
        dropout=0.0,
        attn_resolutions=(4,),
        attn_heads=1,
        attn_head_dim=8,
        use_text_conditioning=False,
        self_conditioning=False,
        attention_placement="mid_down",
    )
    model = UNet(cfg)

    profile = build_model_cost_profile(
        model=model,
        cfg=cfg,
        image_size=32,
        mode="latent",
        latent_downsample_factor=8,
        batch_size=2,
        text_tokens=8,
        dtype=torch.bfloat16,
    )

    assert profile["total_params"] > 0
    assert profile["trainable_params"] == profile["total_params"]
    assert profile["latent_or_image_side"] == 4
    assert profile["attention"]["self_blocks"] == 1
    assert profile["attention"]["cross_blocks"] == 0
    assert profile["attention"]["self_estimated_flops"] > 0
