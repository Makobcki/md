from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from scripts.prepare_latents import _resolve_prepare_options, _sharded_cache_mismatch_reason
import train.runner as runner
from train.runner import (
    _ensure_latent_cache_ready_for_mmdit,
    _load_resume_checkpoint,
    _resolve_latent_shard_index_path,
)


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


def test_missing_sharded_latent_index_reports_auto_prepare_setting(tmp_path) -> None:
    cfg = TrainConfig(
        data_root=str(tmp_path / "data"),
        mode="latent",
        latent_cache=True,
        latent_cache_sharded=True,
        latent_cache_index="index.jsonl",
        vae_pretrained="./vae_sd_mse",
        cache_auto_prepare=False,
    )

    with pytest.raises(RuntimeError) as exc:
        _ensure_latent_cache_ready_for_mmdit(cfg, [{"md5": "a"}], torch.device("cpu"))

    message = str(exc.value)
    assert "Latent cache is missing" in message
    assert "cache.auto_prepare=true" in message


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


def test_sharded_cache_metadata_mismatch_reports_dtype(tmp_path) -> None:
    cache_dir = tmp_path / "latents"
    shard_dir = cache_dir / "shards"
    shard_dir.mkdir(parents=True)
    index_path = cache_dir / "index.jsonl"
    index_path.write_text('{"md5": "a", "shard": "shard_000000.pt", "idx": 0}\n', encoding="utf-8")
    torch.save(
        {
            "format_version": 3,
            "meta_common": {
                "format_version": 3,
                "vae_pretrained": "./vae_sd_mse",
                "scaling_factor": 0.18215,
                "latent_shape": [4, 64, 64],
                "dtype": "fp16",
            },
            "latents": torch.zeros(1, 4, 64, 64),
        },
        shard_dir / "shard_000000.pt",
    )

    reason = _sharded_cache_mismatch_reason(
        index_path=index_path,
        shard_dir=shard_dir,
        expected_meta={
            "format_version": 3,
            "vae_pretrained": "./vae_sd_mse",
            "scaling_factor": 0.18215,
            "latent_shape": [4, 64, 64],
            "dtype": "bf16",
        },
    )

    assert reason == "dtype: 'fp16' != 'bf16'"


def test_prepare_latents_uses_dataset_limit_as_default_limit() -> None:
    cfg = TrainConfig.from_dict({
        "mode": "latent",
        "latent_dtype": "bf16",
        "dataset_limit": 32,
    })

    args = argparse.Namespace(batch_size=16, latent_dtype="fp16")
    options = _resolve_prepare_options(cfg, args, set())

    assert options.limit == 32


def test_prepare_latents_cli_limit_overrides_dataset_limit() -> None:
    cfg = TrainConfig.from_dict({
        "mode": "latent",
        "latent_dtype": "bf16",
        "dataset_limit": 32,
    })

    args = argparse.Namespace(limit=8)
    options = _resolve_prepare_options(cfg, args, {"limit"})

    assert options.limit == 8
