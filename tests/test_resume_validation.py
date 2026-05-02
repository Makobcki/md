from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from config.train import TrainConfig
from scripts.prepare_latents import _resolve_prepare_options
import train.runner as runner
from train.runner import (
    _ensure_latent_cache_ready,
    _load_resume_checkpoint,
    _resolve_latent_shard_index_path,
    _validate_resume_compatibility,
)


def test_resume_validation_rejects_mode_mismatch() -> None:
    cfg = TrainConfig(mode="latent", latent_cache=True)
    ckpt_cfg = TrainConfig(mode="pixel")

    with pytest.raises(RuntimeError, match="resume config mismatch"):
        _validate_resume_compatibility(cfg, ckpt_cfg)


def test_resume_checkpoint_loads_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_load_ckpt(path: str, device: object) -> dict:
        seen["path"] = path
        seen["device"] = device
        return {"format_version": 2, "model": {}}

    monkeypatch.setattr(runner, "load_ckpt", fake_load_ckpt)

    assert _load_resume_checkpoint("ckpt.pt") == {"format_version": 2, "model": {}}
    assert seen["path"] == "ckpt.pt"
    assert str(seen["device"]) == "cpu"


def test_missing_sharded_latent_index_reports_prepare_command(tmp_path) -> None:
    cfg = TrainConfig(
        data_root=str(tmp_path / "data"),
        mode="latent",
        latent_cache=True,
        latent_cache_sharded=True,
        latent_cache_index="index.jsonl",
        latent_precompute=False,
    )

    with pytest.raises(RuntimeError) as exc:
        _ensure_latent_cache_ready(cfg, tmp_path / "out")

    message = str(exc.value)
    assert "Missing sharded latent cache index" in message
    assert "scripts/prepare_latents.py" in message


def test_latent_shard_index_path_honors_configured_name(tmp_path) -> None:
    cfg = TrainConfig(
        data_root=str(tmp_path / "data"),
        mode="latent",
        latent_cache=True,
        latent_cache_sharded=True,
        latent_cache_dir=".cache/custom_latents",
        latent_cache_index="custom_index.jsonl",
    )

    assert _resolve_latent_shard_index_path(cfg) == (
        tmp_path / "data" / ".cache" / "custom_latents" / "custom_index.jsonl"
    )


def test_prepare_latents_options_honor_config_extra() -> None:
    cfg = TrainConfig.from_dict({
        "mode": "latent",
        "latent_dtype": "bf16",
        "latent_cache_sharded": False,
        "latent_prepare_batch_size": 7,
        "latent_prepare_num_workers": 0,
        "latent_prepare_decode_backend": "pil",
    })

    args = argparse.Namespace(batch_size=16, latent_dtype="fp16")
    options = _resolve_prepare_options(cfg, args, set())

    assert options.batch_size == 7
    assert options.num_workers == 0
    assert options.decode_backend == "pil"
    assert options.latent_dtype == "bf16"
    assert options.autocast_dtype == "bf16"
    assert options.shard_size == 0


def test_prepare_latents_cli_overrides_config_extra() -> None:
    cfg = TrainConfig.from_dict({
        "mode": "latent",
        "latent_dtype": "bf16",
        "latent_prepare_batch_size": 7,
    })

    args = argparse.Namespace(batch_size=16, latent_dtype="fp16")
    options = _resolve_prepare_options(cfg, args, {"batch_size", "latent_dtype"})

    assert options.batch_size == 16
    assert options.latent_dtype == "fp16"
