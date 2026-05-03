from __future__ import annotations


def test_model_cost_profile_counts_attention_sites() -> None:
    from diffusion.perf import build_model_cost_profile
    from model.mmdit import MMDiTConfig

    cfg = MMDiTConfig(
        hidden_dim=32,
        num_heads=4,
        depth=5,
        double_stream_blocks=3,
        single_stream_blocks=2,
        text_dim=16,
        pooled_dim=16,
    )
    profile = build_model_cost_profile(cfg, latent_hw=(8, 8), text_tokens=4)

    assert profile.attention_sites == 5
    assert profile.double_stream_attention_sites == 3
    assert profile.single_stream_attention_sites == 2
    assert profile.image_tokens == 16
    assert profile.total_tokens == 20
    assert profile.to_dict()["attention_sites"] == 5
