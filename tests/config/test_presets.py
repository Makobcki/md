from __future__ import annotations

import pytest

from config.loader import load_train_config, resolve_target_config
from config.resolver import resolve_config


def test_explicit_presets_and_main_config_merge() -> None:
    resolved = resolve_target_config("train")
    assert resolved.data["model"]["variant"] == "576"
    assert resolved.data["training"]["batch_size"] == 4
    assert resolved.data["output"]["dir"] == "runs/train"


def test_model_preset_alias_overrides_explicit_use() -> None:
    cfg = load_train_config(overrides={"model": {"preset": "mmdit_1024"}})
    assert cfg.image_size == 1024
    assert cfg.hidden_dim == 1536
    assert cfg.depth == 28


def test_cycle_detection(tmp_path) -> None:
    (tmp_path / "presets").mkdir()
    (tmp_path / "presets" / "a.kdl").write_text(
        'preset kind="test" name="a" version=1 { use "b.kdl" value 1 }',
        encoding="utf-8",
    )
    (tmp_path / "presets" / "b.kdl").write_text(
        'preset kind="test" name="b" version=1 { use "a.kdl" value 2 }',
        encoding="utf-8",
    )
    main = tmp_path / "train.kdl"
    main.write_text(
        'config target="train" version=2 { use "presets/a.kdl" }',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Cyclic config inheritance"):
        resolve_config(main, expected_target="train")


def test_preset_file_cannot_be_launched_directly() -> None:
    with pytest.raises(RuntimeError, match="Preset files cannot be used"):
        resolve_config("configs/presets/model/mmdit_576.kdl", expected_target="train")
