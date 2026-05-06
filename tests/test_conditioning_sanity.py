from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning


def _tiny_model() -> MMDiTFlowModel:
    return MMDiTFlowModel(
        MMDiTConfig(
            latent_channels=4,
            patch_size=2,
            hidden_dim=32,
            depth=2,
            num_heads=4,
            double_stream_blocks=1,
            single_stream_blocks=1,
            text_dim=16,
            pooled_dim=16,
            gradient_checkpointing=False,
            zero_init_final=False,
        )
    )


def _text(batch: int = 1, *, token_value: float = 0.0, pooled_value: float = 0.0) -> TextConditioning:
    tokens = torch.full((batch, 4, 16), float(token_value))
    pooled = torch.full((batch, 16), float(pooled_value))
    mask = torch.ones(batch, 4, dtype=torch.bool)
    return TextConditioning(tokens=tokens, mask=mask, pooled=pooled)


def test_different_text_conditioning_changes_mmdit_text_projection() -> None:
    model = _tiny_model()

    proj_a = model._project_text_tokens(_text(token_value=0.0, pooled_value=0.0), device=torch.device("cpu"), dtype=torch.float32)
    proj_b = model._project_text_tokens(_text(token_value=1.0, pooled_value=1.0), device=torch.device("cpu"), dtype=torch.float32)

    assert proj_a.shape == (1, 4, 32)
    assert proj_b.shape == (1, 4, 32)
    assert not torch.allclose(proj_a, proj_b)


def test_mmdit_accepts_img2img_inpaint_and_control_conditioning() -> None:
    model = _tiny_model()
    x = torch.randn(2, 4, 8, 8)
    t = torch.rand(2)
    text = _text(batch=2, token_value=0.25, pooled_value=0.5)
    source = torch.randn_like(x)
    mask = torch.ones(2, 1, 8, 8)
    controls = torch.randn(2, 2, 4, 8, 8)

    out = model(
        x,
        t,
        text,
        source_latent=source,
        mask=mask,
        control_latents=controls,
        task=["inpaint", "control"],
    )

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_mmdit_rejects_invalid_conditioning_shapes() -> None:
    model = _tiny_model()
    x = torch.randn(1, 4, 8, 8)
    text = _text()

    with pytest.raises(ValueError, match="spatial shape"):
        model(x, torch.rand(1), text, source_latent=torch.randn(1, 4, 4, 4), task="img2img")

    with pytest.raises(ValueError, match="Unsupported MMDiT task"):
        model(x, torch.rand(1), text, task="unsupported")


def test_control_strength_zero_masks_control_stream() -> None:
    model = _tiny_model()
    x = torch.randn(2, 4, 8, 8)
    controls = torch.randn(2, 1, 4, 8, 8)
    _, img_mask, _, _, _ = model._condition_tokens(
        x,
        source_latent=None,
        mask=None,
        control_latents=controls,
        control_strength=torch.tensor([0.0, 1.0]),
        task=["control", "control"],
    )
    control_tokens = (x.shape[-2] // model.cfg.control_patch_size) * (x.shape[-1] // model.cfg.control_patch_size)
    assert not img_mask[0, :control_tokens].any()
    assert img_mask[1, :control_tokens].all()


def test_mixed_task_activates_supplied_conditioning_streams() -> None:
    model = _tiny_model()
    x = torch.randn(1, 4, 8, 8)
    _, img_mask, _, _, _ = model._condition_tokens(
        x,
        source_latent=torch.randn_like(x),
        mask=torch.ones(1, 1, 8, 8),
        control_latents=torch.randn(1, 4, 8, 8),
        control_strength=1.0,
        task="mixed",
    )
    assert img_mask.all()


def test_control_type_string_list_is_rejected_when_batch_equals_streams() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="ambiguous"):
        model._control_type_ids(["image", "canny"], batch=2, streams=2, device=torch.device("cpu"))
