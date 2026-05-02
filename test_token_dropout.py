from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from data_loader.dataset import ImageTextDataset


class _Tokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2


def test_token_dropout_preserves_bos_and_eos() -> None:
    ds = ImageTextDataset(
        entries=[],
        tokenizer=_Tokenizer(),
        cond_drop_prob=0.0,
        token_drop_prob=1.0,
    )
    ids = torch.tensor([1, 10, 2, 0, 0], dtype=torch.long)
    mask = torch.tensor([True, True, True, False, False])

    dropped_ids, dropped_mask = ds._apply_token_dropout(ids, mask, random.Random(0))

    assert torch.equal(dropped_ids, torch.tensor([1, 0, 2, 0, 0], dtype=torch.long))
    assert torch.equal(dropped_mask, torch.tensor([True, False, True, False, False]))
