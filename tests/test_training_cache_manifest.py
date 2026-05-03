from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from PIL import Image

from config.train import TrainConfig
from model.text.cache import TextCache
from scripts.prepare_text_cache import prepare_text_cache
from scripts.prepare_training_cache import main as prepare_training_cache_main


def _make_dataset(root: Path, *, n: int = 3) -> None:
    (root / "images").mkdir(parents=True)
    rows = []
    for idx in range(n):
        md5 = f"sample{idx}"
        Image.new("RGB", (512, 512), color=(idx, idx, idx)).save(root / "images" / f"{md5}.png")
        rows.append({"md5": md5, "file_name": f"images/{md5}.png", "caption": f"caption {idx}"})
    (root / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _cfg(root: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "data_root": str(root),
            "image_dir": "images",
            "caption_field": "caption",
            "require_512": True,
            "val_ratio": 0.0,
            "cache_dir": ".cache",
            "text_cache_dir": ".cache/text",
            "latent_cache_dir": ".cache/latents",
            "text": {
                "backend": "fake",
                "text_dim": 6,
                "pooled_dim": 4,
                "encoders": [{"name": "fake", "model_name": "fake", "max_length": 3}],
            },
            "model": {
                "hidden_dim": 16,
                "depth": 1,
                "num_heads": 4,
                "double_stream_blocks": 1,
                "single_stream_blocks": 0,
                "gradient_checkpointing": False,
            },
            "latent_dtype": "bf16",
            "amp_dtype": "bf16",
        }
    )


def test_prepare_text_cache_writes_sharded_manifest_and_empty_prompt(tmp_path: Path) -> None:
    _make_dataset(tmp_path, n=3)
    cfg = _cfg(tmp_path)
    out_dir = tmp_path / ".cache" / "text"

    manifest = prepare_text_cache(
        cfg=cfg,
        out_dir=out_dir,
        batch_size=2,
        shard_size=2,
        limit=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "empty_prompt.safetensors").exists()
    assert manifest["num_samples"] == 3
    assert [s["num_samples"] for s in manifest["shards"]] == [2, 1]
    assert manifest["shards"][0]["sample_ids"] == ["sample0", "sample1"]
    assert manifest["shards"][1]["sample_ids"] == ["sample2"]
    assert manifest["shards"][0]["tokens_shape"] == [2, 3, 6]

    cache = TextCache(out_dir)
    cache.validate_files_readable()
    assert cache.load("sample0").tokens.shape == (3, 6)
    assert cache.load_empty().tokens.shape == (1, 3, 6)


def test_prepare_training_cache_writes_unified_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_dataset(tmp_path, n=2)
    cfg_path = tmp_path / "train.yaml"
    cfg_data = _cfg(tmp_path).to_dict()
    cfg_path.write_text(yaml.safe_dump(cfg_data, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_training_cache",
            "--config",
            str(cfg_path),
            "--skip-latents",
            "--device",
            "cpu",
            "--text-dtype",
            "fp32",
            "--text-batch-size",
            "2",
            "--text-shard-size",
            "2",
        ],
    )
    prepare_training_cache_main()

    manifest_path = tmp_path / ".cache" / "training_cache_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["num_samples"] == 2
    assert manifest["latent_shape"] == [4, 64, 64]
    assert manifest["text"]["encoders"][0]["name"] == "fake"
    assert manifest["shards"]["text"][0]["sample_ids"] == ["sample0", "sample1"]
    assert manifest["shards"]["latents"]["mode"] == "files"
