from __future__ import annotations

import pytest
import torch

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning
from model.text.pretrained import FrozenTextEncoderBundle


def test_fake_text_backend_marks_clip_and_t5_token_types() -> None:
    enc = FrozenTextEncoderBundle(
        {
            "text": {
                "backend": "fake",
                "text_dim": 16,
                "pooled_dim": 16,
                "encoders": [
                    {"name": "clip_l", "model_name": "fake", "max_length": 3},
                    {"name": "t5_base", "model_name": "fake", "max_length": 5},
                ],
            }
        },
        dtype=torch.float32,
        backend="fake",
    )
    cond = enc(["a cat"])
    assert cond.token_types is not None
    assert cond.token_types.shape == cond.mask.shape
    assert torch.equal(cond.token_types[:, :3], torch.zeros(1, 3, dtype=torch.long))
    assert torch.equal(cond.token_types[:, 3:], torch.ones(1, 5, dtype=torch.long))


def test_mmdit_uses_separate_text_projection_and_type_embeddings() -> None:
    cfg = MMDiTConfig(
        hidden_dim=32,
        depth=1,
        num_heads=4,
        double_stream_blocks=1,
        single_stream_blocks=0,
        text_dim=16,
        pooled_dim=16,
        gradient_checkpointing=False,
    )
    model = MMDiTFlowModel(cfg)
    assert hasattr(model, "text_clip_in")
    assert hasattr(model, "text_t5_in")
    assert hasattr(model, "type_clip")
    assert hasattr(model, "type_t5")
    assert hasattr(model, "type_image")
    text = TextConditioning(
        tokens=torch.randn(1, 4, 16),
        mask=torch.ones(1, 4, dtype=torch.bool),
        pooled=torch.randn(1, 16),
        token_types=torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
    )
    out = model(torch.randn(1, 4, 8, 8), torch.rand(1), text)
    assert out.shape == (1, 4, 8, 8)


def test_invalid_text_token_types_shape_is_rejected() -> None:
    cfg = MMDiTConfig(hidden_dim=32, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0, text_dim=16, pooled_dim=16)
    model = MMDiTFlowModel(cfg)
    text = TextConditioning(
        tokens=torch.randn(1, 4, 16),
        mask=torch.ones(1, 4, dtype=torch.bool),
        pooled=torch.randn(1, 16),
        token_types=torch.zeros(1, 3, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="token_types shape"):
        model(torch.randn(1, 4, 8, 8), torch.rand(1), text)
