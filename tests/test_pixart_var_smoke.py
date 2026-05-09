from __future__ import annotations

import torch

from model.pixart_sigma import PixArtSigmaConfig, PixArtSigmaRFModel
from model.var import VARConfig, VARTransformer, deterministic_decode, next_scale_cross_entropy


def test_pixart_sigma_tiny_forward_backward_cpu() -> None:
    cfg = PixArtSigmaConfig(
        latent_channels=4,
        patch_size=2,
        hidden_size=32,
        depth=1,
        num_heads=4,
        caption_channels=16,
        cross_attention_dim=32,
        max_text_tokens=8,
    )
    model = PixArtSigmaRFModel(cfg)
    x = torch.randn(2, 4, 4, 4)
    t = torch.rand(2)
    text = torch.randn(2, 6, 16)
    out = model(x=x, t=t, text=text)

    assert out.shape == x.shape
    out.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_var_tiny_forward_loss_and_decode_cpu() -> None:
    cfg = VARConfig(
        codebook_size=32,
        hidden_size=32,
        depth=1,
        num_heads=4,
        scale_schedule=(1, 2),
        max_token_length=5,
    )
    model = VARTransformer(cfg)
    tokens = [torch.randint(0, 32, (2, 1)), torch.randint(0, 32, (2, 4))]
    logits = model(tokens)
    loss = next_scale_cross_entropy(logits, tokens[1])

    assert logits.shape == (2, 4, 32)
    assert loss.item() > 0
    loss.backward()
    decoded = deterministic_decode(model, batch_size=2, device=torch.device("cpu"))
    assert [item.shape[1] for item in decoded] == [1, 4]


def test_var_decode_uses_next_scale_for_multiscale_schedule_cpu() -> None:
    cfg = VARConfig(
        codebook_size=32,
        hidden_size=32,
        depth=1,
        num_heads=4,
        scale_schedule=(1, 2, 4),
        max_token_length=21,
    )
    model = VARTransformer(cfg)

    decoded = deterministic_decode(model, batch_size=2, device=torch.device("cpu"))

    assert [item.shape[1] for item in decoded] == [1, 4, 16]
