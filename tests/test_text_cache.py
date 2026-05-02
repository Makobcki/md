from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from model.text.cache import TextCache
from config.train import TrainConfig
from train.runner import _validate_text_cache_for_mmdit


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


def test_text_cache_metadata_validation_catches_missing_key(tmp_path) -> None:
    root = tmp_path / "text"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "text_dim": 1024,
                "pooled_dim": 1024,
                "encoders": [
                    {
                        "name": "clip_l",
                        "model_name": "openai/clip-vit-large-patch14",
                        "max_length": 77,
                    }
                ],
            }
        )
    )
    (root / "index.jsonl").write_text(json.dumps({"key": "present", "shard": "text_00000.safetensors", "idx": 0}) + "\n")
    cache = TextCache(root)
    cfg = TrainConfig.from_dict(
        {
            "architecture": "mmdit_rf",
            "objective": "rectified_flow",
            "prediction_type": "flow_velocity",
            "text": {
                "text_dim": 1024,
                "pooled_dim": 1024,
                "encoders": [
                    {
                        "name": "clip_l",
                        "model_name": "openai/clip-vit-large-patch14",
                        "max_length": 77,
                    }
                ],
            },
        }
    )

    with pytest.raises(RuntimeError, match="text cache missing"):
        _validate_text_cache_for_mmdit(cache, cfg, [{"md5": "missing", "caption": "x"}])
