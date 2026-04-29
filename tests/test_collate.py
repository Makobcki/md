from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data_loader.collate import collate_with_tokenizer


def test_collate_text_conditioned_three_tuple() -> None:
    batch = [
        (
            torch.full((3, 4, 4), 1.0),
            torch.tensor([1, 2, 0], dtype=torch.long),
            torch.tensor([True, True, False]),
        ),
        (
            torch.full((3, 4, 4), 2.0),
            torch.tensor([1, 3, 0], dtype=torch.long),
            torch.tensor([True, True, False]),
        ),
    ]

    imgs, ids, mask = collate_with_tokenizer(batch)

    assert imgs.shape == (2, 3, 4, 4)
    assert torch.equal(ids, torch.tensor([[1, 2, 0], [1, 3, 0]], dtype=torch.long))
    assert torch.equal(mask, torch.tensor([[True, True, False], [True, True, False]]))


def test_collate_text_conditioned_latent_flag_fallback() -> None:
    latent = torch.full((4, 2, 2), 1.0)
    image = torch.full((3, 16, 16), 2.0)
    encoded_image = torch.full((4, 2, 2), 3.0)

    def latent_encoder(x: torch.Tensor) -> torch.Tensor:
        assert x.shape == (1, 3, 16, 16)
        return encoded_image.unsqueeze(0)

    batch = [
        (
            latent,
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([True, True]),
            True,
        ),
        (
            image,
            torch.tensor([1, 0], dtype=torch.long),
            torch.tensor([True, False]),
            False,
        ),
    ]

    imgs, ids, mask = collate_with_tokenizer(batch, latent_encoder=latent_encoder)

    assert torch.equal(imgs, torch.stack([latent, encoded_image], dim=0))
    assert torch.equal(ids, torch.tensor([[1, 2], [1, 0]], dtype=torch.long))
    assert torch.equal(mask, torch.tensor([[True, True], [True, False]]))
