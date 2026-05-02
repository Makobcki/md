from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from PIL import Image

from data_loader import DataConfig, build_or_load_index
from data_loader import indexing


def test_index_cache_with_stale_image_paths_is_rebuilt(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_dir = root / "images"
    cache_dir = root / ".cache"
    image_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    Image.new("RGB", (512, 512)).save(image_dir / "abcdef.png")

    cfg = DataConfig(
        root=str(root),
        image_dir="images",
        meta_dir="",
        tags_dir="",
        images_only=True,
        require_512=True,
        val_ratio=0.0,
        cache_dir=".cache",
    )
    cache_path = cache_dir / "index_images_only_req5121_val0.0_imgdirimages.jsonl"
    stale_entry = {
        "md5": "stale",
        "img": str(tmp_path / "old-root" / "stale.png"),
        "caption": "",
        "tags_primary": [],
        "tags_gender": [],
    }
    cache_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "meta", "schema_version": 3, "config": indexing._cache_metadata(cfg)}),
                json.dumps({"split": "train", "entry": stale_entry}),
                json.dumps({"type": "done", "schema_version": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    train_entries, val_entries = build_or_load_index(cfg)

    assert val_entries == []
    assert len(train_entries) == 1
    assert train_entries[0]["md5"] == "abcdef"
    assert Path(train_entries[0]["img"]).exists()


def test_frozen_text_encoder_bundle_uses_t5_encoder_model(monkeypatch) -> None:
    import torch

    calls: list[str] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model_name: str):
            return cls()

    class FakeEncoder(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, model_name: str):
            calls.append(f"{cls.__name__}:{model_name}")
            return cls()

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

    class CLIPTextModel(FakeEncoder):
        pass

    class T5EncoderModel(FakeEncoder):
        pass

    class AutoModel(FakeEncoder):
        pass

    fake_transformers = types.SimpleNamespace(
        AutoModel=AutoModel,
        AutoTokenizer=FakeTokenizer,
        CLIPTextModel=CLIPTextModel,
        T5EncoderModel=T5EncoderModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    import model.text.pretrained as pretrained

    monkeypatch.setattr(pretrained, "_resolve_cached_model_path", lambda name: name)

    pretrained.FrozenTextEncoderBundle(
        {
            "text": {
                "text_dim": 8,
                "pooled_dim": 8,
                "encoders": [
                    {"name": "clip_l", "model_name": "openai/clip-vit-large-patch14"},
                    {"name": "t5", "model_name": "google/t5-v1_1-base"},
                ],
            }
        },
        device="cpu",
        dtype=torch.float32,
    )

    assert calls == [
        "CLIPTextModel:openai/clip-vit-large-patch14",
        "T5EncoderModel:google/t5-v1_1-base",
    ]


def test_makefile_exposes_mmdit_check_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "check-mmdit-smoke-resume:" in makefile
    assert "check-mmdit-overfit:" in makefile


def test_mmdit_smoke_config_indexes_metadata_jsonl(tmp_path: Path) -> None:
    from dataclasses import replace

    from config.train import TrainConfig

    cfg = TrainConfig.from_yaml("config/train_mmdit_rf_smoke.yaml")
    root = tmp_path / "pixso_512"
    image_dir = root / cfg.image_dir
    image_dir.mkdir(parents=True)
    Image.new("RGB", (512, 512)).save(image_dir / "sample-001.png")
    (root / "metadata.jsonl").write_text(
        json.dumps({"file_name": "sample-001.png", "text": "1girl, simple background"}) + "\n",
        encoding="utf-8",
    )

    cfg = replace(cfg, data_root=str(root), dataset_limit=0)
    dcfg = DataConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        images_only=bool(cfg.images_only),
        use_text_conditioning=bool(cfg.use_text_conditioning),
        min_tag_count=int(cfg.min_tag_count),
        require_512=bool(cfg.require_512),
        val_ratio=0.0,
        seed=int(cfg.seed),
        cache_dir=str(cfg.cache_dir),
        failed_list=str(cfg.failed_list),
    )

    train_entries, val_entries = build_or_load_index(dcfg)

    assert val_entries == []
    assert len(train_entries) == 1
    assert train_entries[0]["md5"] == "sample-001"
    assert train_entries[0]["caption"] == "1girl, simple background"
    assert Path(train_entries[0]["img"]).name == "sample-001.png"


def test_diffusion_package_exports_legacy_config() -> None:
    from diffusion import DiffusionConfig

    assert DiffusionConfig.__name__ == "DiffusionConfig"
