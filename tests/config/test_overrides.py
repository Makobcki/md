from __future__ import annotations

from config.loader import load_sample_config, load_train_config
from config.overrides import parse_set_overrides
from config.resolver import recursive_merge


def test_parse_set_overrides_types() -> None:
    overrides = parse_set_overrides(
        [
            "training.batch_size=8",
            "sampling.guidance_scale=4.5",
            "webui.auto_open=false",
            "model.checkpoint=null",
            "data.text_fields=[\"caption\", \"tags\"]",
            "prompt.text=a cinematic landscape",
        ]
    )

    assert overrides["training"]["batch_size"] == 8
    assert overrides["sampling"]["guidance_scale"] == 4.5
    assert overrides["webui"]["auto_open"] is False
    assert overrides["model"]["checkpoint"] is None
    assert overrides["data"]["text_fields"] == ["caption", "tags"]
    assert overrides["prompt"]["text"] == "a cinematic landscape"


def test_cli_overrides_win_after_presets() -> None:
    cfg = load_train_config(
        overrides=parse_set_overrides(
            [
                "model.preset=mmdit_1024",
                "training.batch_size=2",
                "output.dir=runs/mmdit_1024",
            ]
        )
    )

    assert cfg.image_size == 1024
    assert cfg.batch_size == 2
    assert cfg.out_dir == "runs/mmdit_1024"


def test_sample_override_mapping() -> None:
    cfg = load_sample_config(
        overrides=parse_set_overrides(
            [
                "prompt.text=a cinematic landscape",
                "model.checkpoint=checkpoints/latest.safetensors",
                "sampling.steps=40",
            ]
        )
    )

    assert cfg.options.prompt == "a cinematic landscape"
    assert cfg.options.ckpt == "checkpoints/latest.safetensors"
    assert cfg.options.steps == 40


def test_recursive_merge_replaces_lists() -> None:
    merged = recursive_merge(
        {"a": {"b": 1, "items": [1, 2]}},
        {"a": {"c": 2, "items": [3]}},
    )
    assert merged == {"a": {"b": 1, "c": 2, "items": [3]}}
