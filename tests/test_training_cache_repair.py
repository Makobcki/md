from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

from PIL import Image

from config.train import TrainConfig
from model.text.cache import TextCache
from scripts.prepare_text_cache import prepare_text_cache
from scripts.prepare_training_cache import main as prepare_training_cache_main

import torch


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


def _write_config(root: Path) -> Path:
    cfg_path = root / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(root).to_dict(), sort_keys=False), encoding="utf-8")
    return cfg_path


def _prepare_text(root: Path, *, shard_size: int = 2) -> Path:
    cfg = _cfg(root)
    out = root / ".cache" / "text"
    prepare_text_cache(
        cfg=cfg,
        out_dir=out,
        batch_size=2,
        shard_size=shard_size,
        limit=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return out


def _run_prepare(monkeypatch: pytest.MonkeyPatch, cfg_path: Path, *extra: str) -> None:
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
            *extra,
        ],
    )
    prepare_training_cache_main()


def test_repair_regenerates_missing_text_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_dataset(tmp_path, n=3)
    cfg_path = _write_config(tmp_path)
    text_root = _prepare_text(tmp_path, shard_size=2)
    missing = text_root / "shards" / "text_00001.safetensors"
    missing.unlink()

    _run_prepare(monkeypatch, cfg_path, "--repair")

    assert missing.exists()
    TextCache(text_root).validate_files_readable()
    assert (tmp_path / ".cache" / "training_cache_manifest.json").exists()


def test_repair_rejects_broken_metadata_unless_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_dataset(tmp_path, n=2)
    cfg_path = _write_config(tmp_path)
    text_root = _prepare_text(tmp_path, shard_size=2)
    (text_root / "metadata.json").write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Broken text cache metadata"):
        _run_prepare(monkeypatch, cfg_path, "--repair")

    _run_prepare(monkeypatch, cfg_path, "--repair", "--force")
    metadata = json.loads((text_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["text_dim"] == 6


def test_repair_rejects_stale_dataset_hash_unless_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _make_dataset(tmp_path, n=2)
    cfg_path = _write_config(tmp_path)
    text_root = _prepare_text(tmp_path, shard_size=2)
    metadata_path = text_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_hash"] = "stale"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RuntimeError, match="dataset changed"):
        _run_prepare(monkeypatch, cfg_path, "--repair")

    _run_prepare(monkeypatch, cfg_path, "--repair", "--rebuild")
    repaired = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert repaired["dataset_hash"] != "stale"
