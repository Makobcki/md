from __future__ import annotations

import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
torch = pytest.importorskip("torch")

from config.train import TrainConfig
from data_loader import DataConfig, build_or_load_index
from data_loader.dataset import ImageTextDataset
from scripts.prepare_text_cache import prepare_text_cache


class _Tokenizer:
    def encode(self, text: str):
        # Encode the exact text into one token-like scalar so tests can inspect it.
        value = len(text)
        ids = torch.tensor([1, value, 2], dtype=torch.long)
        mask = torch.tensor([True, bool(text), True])
        return ids, mask


def _write_prompt_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (512, 512), color=(32, 32, 32)).save(root / "images" / "sample0.png")
    row = {
        "md5": "sample0",
        "file_name": "images/sample0.png",
        "caption": "caption text should not win",
        "prompt": "prompt text should win",
    }
    (root / "metadata.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def _tiny_prompt_cfg(root: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "data_root": str(root),
            "image_dir": "images",
            "caption_field": "caption",
            "text_field": "prompt",
            "require_512": True,
            "val_ratio": 0.0,
            "cache_dir": ".cache",
            "text_cache_dir": ".cache/text",
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


def test_index_can_prefer_prompt_over_caption(tmp_path: Path) -> None:
    _write_prompt_dataset(tmp_path)
    dcfg = DataConfig(
        root=str(tmp_path),
        image_dir="images",
        caption_field="caption",
        text_field="prompt",
        text_fields=[],
        require_512=True,
        val_ratio=0.0,
    )

    train_entries, val_entries = build_or_load_index(dcfg)

    assert val_entries == []
    assert len(train_entries) == 1
    entry = train_entries[0]
    assert entry["text"] == "prompt text should win"
    assert entry["text_source"] == "prompt"
    # Legacy field remains populated for existing training/cache code paths.
    assert entry["caption"] == "prompt text should win"


def test_train_config_accepts_nested_dataset_text_fields() -> None:
    cfg = TrainConfig.from_dict(
        {
            "dataset": {"text_field": "prompt", "text_fields": ["prompt", "caption", "text"]},
            "model": {"hidden_dim": 16, "depth": 1, "num_heads": 4, "double_stream_blocks": 1, "single_stream_blocks": 0},
            "text": {"text_dim": 8, "pooled_dim": 8},
        }
    )

    assert cfg.text_field == "prompt"
    assert cfg.text_fields == ["prompt", "caption", "text"]


def test_dataset_tokenizer_uses_exact_prompt_text(tmp_path: Path) -> None:
    _write_prompt_dataset(tmp_path)
    entries, _ = build_or_load_index(
        DataConfig(
            root=str(tmp_path),
            image_dir="images",
            caption_field="caption",
            text_field="prompt",
            require_512=True,
            val_ratio=0.0,
        )
    )
    ds = ImageTextDataset(entries, tokenizer=_Tokenizer(), cond_drop_prob=0.0)

    _img, ids, _mask = ds[0]

    assert int(ids[1].item()) == len("prompt text should win")


def test_prepare_text_cache_records_prompt_text_source(tmp_path: Path) -> None:
    _write_prompt_dataset(tmp_path)
    cfg = _tiny_prompt_cfg(tmp_path)
    out_dir = tmp_path / ".cache" / "text"

    manifest = prepare_text_cache(
        cfg=cfg,
        out_dir=out_dir,
        batch_size=1,
        shard_size=1,
        limit=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["text_field"] == "prompt"
    assert metadata["resolved_text_fields"][:2] == ["prompt", "caption"]
    assert manifest["text_field"] == "prompt"
    assert manifest["shards"][0]["sample_ids"] == ["sample0"]
