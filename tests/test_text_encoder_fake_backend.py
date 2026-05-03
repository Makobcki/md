from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.text.pretrained import FrozenTextEncoderBundle


def _cfg() -> dict:
    return {
        "text": {
            "backend": "fake",
            "text_dim": 6,
            "pooled_dim": 4,
            "encoders": [
                {"name": "fake_clip", "model_name": "fake", "max_length": 3},
                {"name": "fake_t5", "model_name": "fake", "max_length": 5},
            ],
        }
    }


def test_fake_text_encoder_shapes_dtype_mask_and_empty_prompt() -> None:
    encoder = FrozenTextEncoderBundle.from_config(_cfg(), device="cpu", dtype=torch.float32)
    cond = encoder(["a cat", "", "a longer prompt with many words"])

    assert cond.tokens.shape == (3, 8, 6)
    assert cond.mask.shape == (3, 8)
    assert cond.pooled.shape == (3, 4)
    assert cond.tokens.dtype == torch.float32
    assert cond.pooled.dtype == torch.float32
    assert cond.mask.dtype == torch.bool
    assert cond.is_uncond is not None
    assert cond.is_uncond.tolist() == [False, True, False]
    assert cond.mask[1].sum().item() == 0
    assert cond.mask[0].sum().item() == 4  # two words in each fake encoder chunk
    assert cond.mask[2].sum().item() == 8  # clipped to 3 + 5


def test_fake_text_encoder_is_deterministic() -> None:
    encoder = FrozenTextEncoderBundle.from_config(_cfg(), device="cpu", dtype=torch.float32)
    a = encoder(["same prompt", "different"])
    b = encoder(["same prompt", "different"])
    c = encoder(["same prompt changed", "different"])

    assert torch.equal(a.tokens, b.tokens)
    assert torch.equal(a.mask, b.mask)
    assert torch.equal(a.pooled, b.pooled)
    assert not torch.equal(a.tokens[0], c.tokens[0])


def test_fake_text_encoder_metadata() -> None:
    encoder = FrozenTextEncoderBundle.from_config(_cfg(), device="cpu", dtype=torch.float32)
    meta = encoder.metadata()
    assert meta["backend"] == "fake"
    assert meta["text_dim"] == 6
    assert meta["pooled_dim"] == 4
    assert [item["max_length"] for item in meta["encoders"]] == [3, 5]
