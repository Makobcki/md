from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from model.text.cache import TextCache


def test_text_cache_loads_shard(tmp_path) -> None:
    root = tmp_path / "text"
    (root / "shards").mkdir(parents=True)
    safetensors.save_file(
        {
            "tokens": torch.randn(1, 3, 4),
            "mask": torch.ones(1, 3, dtype=torch.uint8),
            "pooled": torch.randn(1, 4),
            "is_uncond": torch.zeros(1, dtype=torch.uint8),
        },
        str(root / "shards" / "text_00000.safetensors"),
    )
    (root / "index.jsonl").write_text(json.dumps({"key": "abc", "shard": "text_00000.safetensors", "idx": 0}) + "\n")
    cache = TextCache(root)
    cond = cache.load("abc")
    assert cond.tokens.shape == (3, 4)
    assert cond.mask.dtype == torch.bool
    assert cond.pooled.shape == (4,)

