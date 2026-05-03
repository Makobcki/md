from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
from model.text.cache import TextCache
from config.train import TrainConfig
from train.runner import _validate_text_cache_for_mmdit


def _write_empty_prompt(root, *, tokens_shape=(1, 3, 4)) -> None:
    torch.save(
        {
            "tokens": torch.zeros(*tokens_shape),
            "mask": torch.zeros(tokens_shape[0], tokens_shape[1], dtype=torch.uint8),
            "pooled": torch.zeros(tokens_shape[0], tokens_shape[2]),
            "is_uncond": torch.ones(tokens_shape[0], dtype=torch.uint8),
        },
        str(root / "empty_prompt.safetensors"),
    )


def test_text_cache_loads_shard(tmp_path) -> None:
    root = tmp_path / "text"
    (root / "shards").mkdir(parents=True)
    torch.save(
        {
            "tokens": torch.randn(1, 3, 4),
            "mask": torch.ones(1, 3, dtype=torch.uint8),
            "pooled": torch.randn(1, 4),
            "is_uncond": torch.zeros(1, dtype=torch.uint8),
        },
        str(root / "shards" / "text_00000.safetensors"),
    )
    _write_empty_prompt(root)
    (root / "index.jsonl").write_text(json.dumps({"key": "abc", "shard": "text_00000.safetensors", "idx": 0}) + "\n")
    cache = TextCache(root)
    cond = cache.load("abc")
    assert cond.tokens.shape == (3, 4)
    assert cond.mask.dtype == torch.bool
    assert cond.pooled.shape == (4,)
    empty = cache.load_empty()
    assert empty.is_uncond is not None and empty.is_uncond.item() is True


def test_text_cache_missing_key_error_is_explicit(tmp_path) -> None:
    root = tmp_path / "text"
    root.mkdir(parents=True)
    (root / "index.jsonl").write_text("")
    cache = TextCache(root)
    with pytest.raises(KeyError, match="Text cache missing key: missing"):
        cache.load("missing")


def test_text_cache_metadata_validation_catches_missing_key(tmp_path) -> None:
    root = tmp_path / "text"
    (root / "shards").mkdir(parents=True)
    torch.save(
        {
            "tokens": torch.randn(1, 3, 1024),
            "mask": torch.ones(1, 3, dtype=torch.uint8),
            "pooled": torch.randn(1, 1024),
            "is_uncond": torch.zeros(1, dtype=torch.uint8),
        },
        str(root / "shards" / "text_00000.safetensors"),
    )
    _write_empty_prompt(root, tokens_shape=(1, 3, 1024))
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


def test_text_cache_validation_strict_false_warns_for_metadata_mismatch(tmp_path, capsys) -> None:
    root = tmp_path / "text"
    (root / "shards").mkdir(parents=True)
    torch.save(
        {
            "tokens": torch.randn(1, 3, 4),
            "mask": torch.ones(1, 3, dtype=torch.uint8),
            "pooled": torch.randn(1, 4),
            "is_uncond": torch.zeros(1, dtype=torch.uint8),
        },
        str(root / "shards" / "text_00000.safetensors"),
    )
    _write_empty_prompt(root)
    (root / "metadata.json").write_text(json.dumps({"text_dim": 999, "pooled_dim": 4, "encoders": []}))
    (root / "manifest.json").write_text(json.dumps({"num_samples": 1, "shards": [{"name": "text_00000.safetensors"}]}))
    (root / "index.jsonl").write_text(json.dumps({"key": "present", "shard": "text_00000.safetensors", "idx": 0}) + "\n")
    cfg = TrainConfig.from_dict({"text_dim": 4, "pooled_dim": 4, "cache": {"strict": False}})

    _validate_text_cache_for_mmdit(TextCache(root), cfg, [{"md5": "present", "caption": "x"}], strict=False)
    assert "text cache text_dim mismatch" in capsys.readouterr().out


def test_text_cache_reuses_loaded_shard(tmp_path, monkeypatch) -> None:
    root = tmp_path / "text"
    (root / "shards").mkdir(parents=True)
    (root / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"key": "a", "shard": "text_00000.safetensors", "idx": 0}),
                json.dumps({"key": "b", "shard": "text_00000.safetensors", "idx": 1}),
            ]
        )
        + "\n"
    )
    cache = TextCache(root)
    calls = {"count": 0}
    payload = {
        "tokens": torch.randn(2, 3, 4),
        "mask": torch.ones(2, 3, dtype=torch.uint8),
        "pooled": torch.randn(2, 4),
        "is_uncond": torch.zeros(2, dtype=torch.uint8),
    }

    def fake_load(path):
        calls["count"] += 1
        return payload

    monkeypatch.setattr(cache, "_load_safetensors", fake_load)
    cache.load("a")
    cache.load("b")
    assert calls["count"] == 1


def test_text_cache_dataset_hash_mismatch_allows_superset_cache(tmp_path, capsys) -> None:
    root = tmp_path / "text"
    (root / "shards").mkdir(parents=True)
    torch.save(
        {
            "tokens": torch.randn(2, 3, 4),
            "mask": torch.ones(2, 3, dtype=torch.uint8),
            "pooled": torch.randn(2, 4),
            "is_uncond": torch.zeros(2, dtype=torch.uint8),
        },
        str(root / "shards" / "text_00000.safetensors"),
    )
    _write_empty_prompt(root)
    (root / "metadata.json").write_text(json.dumps({"text_dim": 4, "pooled_dim": 4, "encoders": [], "dataset_hash": "old-full-split-hash"}))
    (root / "manifest.json").write_text(json.dumps({"num_samples": 2, "shards": [{"name": "text_00000.safetensors"}]}))
    (root / "index.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"key": "train-key", "shard": "text_00000.safetensors", "idx": 0}),
                json.dumps({"key": "val-key", "shard": "text_00000.safetensors", "idx": 1}),
            ]
        )
        + "\n"
    )
    cfg = TrainConfig.from_dict({"text_dim": 4, "pooled_dim": 4, "cache": {"strict": True}})

    _validate_text_cache_for_mmdit(TextCache(root), cfg, [{"md5": "train-key", "caption": "x"}], strict=True)

    out = capsys.readouterr().out
    assert "text cache dataset_hash mismatch" in out
    assert "compatible text cache superset" in out


def test_text_cache_text_hash_mismatch_is_still_strict(tmp_path) -> None:
    root = tmp_path / "text"
    (root / "shards").mkdir(parents=True)
    torch.save(
        {
            "tokens": torch.randn(1, 3, 4),
            "mask": torch.ones(1, 3, dtype=torch.uint8),
            "pooled": torch.randn(1, 4),
            "is_uncond": torch.zeros(1, dtype=torch.uint8),
        },
        str(root / "shards" / "text_00000.safetensors"),
    )
    _write_empty_prompt(root)
    (root / "metadata.json").write_text(json.dumps({"text_dim": 4, "pooled_dim": 4, "encoders": []}))
    (root / "index.jsonl").write_text(
        json.dumps({"key": "present", "shard": "text_00000.safetensors", "idx": 0, "text_hash": "definitely-stale"}) + "\n"
    )
    cfg = TrainConfig.from_dict({"text_dim": 4, "pooled_dim": 4})

    with pytest.raises(RuntimeError, match="prompt/text hash mismatch"):
        _validate_text_cache_for_mmdit(TextCache(root), cfg, [{"md5": "present", "caption": "new text"}], strict=True)
