from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.mmdit.patch import patchify, unpatchify


def test_patchify_unpatchify_roundtrip() -> None:
    x = torch.randn(2, 4, 8, 8)
    tokens = patchify(x, patch_size=2)
    y = unpatchify(tokens, channels=4, height=8, width=8, patch_size=2)
    assert tokens.shape == (2, 16, 16)
    assert torch.equal(x, y)

