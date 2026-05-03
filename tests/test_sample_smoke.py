from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from diffusion.utils import save_ckpt
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from sample.cli import _main_impl as sample_cli_main
from train.loop_mmdit import build_mmdit_checkpoint


def _tiny_cfg(data_root: str | None = None) -> dict:
    cfg = {
        "architecture": "mmdit_rf",
        "objective": "rectified_flow",
        "prediction_type": "flow_velocity",
        "mode": "latent",
        "image_size": 32,
        "latent_downsample_factor": 8,
        "latent_channels": 4,
        "latent_patch_size": 2,
        "hidden_dim": 16,
        "depth": 1,
        "num_heads": 4,
        "double_stream_blocks": 1,
        "single_stream_blocks": 0,
        "gradient_checkpointing": False,
        "zero_init_final": True,
        "text_dim": 8,
        "pooled_dim": 8,
        "amp_dtype": "bf16",
        "sampling_shift": 1.0,
        "vae_pretrained": "fake",
        "vae_scaling_factor": 0.18215,
        "vae": {"backend": "fake", "pretrained": "fake", "scaling_factor": 0.18215},
        "text": {
            "backend": "fake",
            "text_dim": 8,
            "pooled_dim": 8,
            "encoders": [{"name": "fake", "model_name": "fake", "max_length": 4}],
        },
        "dataset_tasks": {"txt2img": 1.0},
    }
    if data_root is not None:
        cfg["data_root"] = data_root
        cfg["text_cache_dir"] = ".cache/text"
    return cfg


def _write_tiny_checkpoint(path: Path, *, data_root: str | None = None) -> None:
    cfg = _tiny_cfg(data_root)
    model = MMDiTFlowModel(MMDiTConfig.from_dict(cfg))
    ckpt = build_mmdit_checkpoint(model=model, ema=None, optimizer=None, scheduler=None, step=7, cfg_dict=cfg)
    save_ckpt(str(path), ckpt)


def _run_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, sampler: str, latent_only: bool, neg_prompt: str = "") -> tuple[Path, dict]:
    ckpt = tmp_path / "tiny.pt"
    if not ckpt.exists():
        _write_tiny_checkpoint(ckpt)
    out = tmp_path / (f"sample_{sampler}_{'latent' if latent_only else 'image'}" + (".pt" if latent_only else ".png"))
    argv = [
        "sample.cli",
        "--ckpt",
        str(ckpt),
        "--out",
        str(out),
        "--n",
        "1",
        "--steps",
        "2",
        "--prompt",
        "a cat on a table",
        "--negative-prompt",
        neg_prompt,
        "--cfg",
        "1.5",
        "--sampler",
        sampler,
        "--seed",
        "123",
        "--device",
        "cpu",
    ]
    if latent_only:
        argv.append("--latent-only")
    else:
        argv.append("--fake-vae")
    monkeypatch.setattr(sys, "argv", argv)
    sample_cli_main()
    meta_path = out.with_suffix(".json")
    return out, json.loads(meta_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("sampler", ["flow_euler", "flow_heun"])
def test_sample_cli_txt2img_smoke_with_fake_vae_creates_image_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sampler: str,
) -> None:
    out, meta = _run_sample(monkeypatch, tmp_path, sampler=sampler, latent_only=False, neg_prompt="bad quality")

    assert out.exists()
    assert out.stat().st_size > 0
    assert out.with_suffix(".json").exists()
    assert meta["checkpoint_step"] == 7
    assert meta["architecture"] == "mmdit_rf"
    assert meta["objective"] == "rectified_flow"
    assert meta["prompt"] == "a cat on a table"
    assert meta["negative_prompt"] == "bad quality"
    assert meta["sampler"] == sampler
    assert meta["steps"] == 2
    assert meta["cfg"] == 1.5
    assert meta["seed"] == 123
    assert meta["sampling_shift"] == 1.0
    assert meta["image_size"] == [32, 32]
    assert meta["latent_shape"] == [4, 4, 4]
    assert meta["model_config"]["hidden_dim"] == 16
    assert meta["vae_config"]["runtime_backend"] == "fake"
    assert meta["text_encoder_config"]["backend"] == "fake"


def test_sample_cli_latent_only_smoke_without_vae_creates_latent_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out, meta = _run_sample(monkeypatch, tmp_path, sampler="flow_euler", latent_only=True)

    assert out.exists()
    z = torch.load(out, map_location="cpu")
    assert tuple(z.shape) == (1, 4, 4, 4)
    assert torch.isfinite(z).all()
    assert meta["latent_only"] is True
    assert meta["vae_config"]["runtime_backend"] == "latent_only"
    assert meta["negative_prompt"] == ""
    assert meta["negative_prompt_source"] == "encoder"



def test_sample_cli_uses_empty_prompt_cache_for_empty_negative_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "dataset"
    text_root = data_root / ".cache" / "text"
    text_root.mkdir(parents=True)
    torch.save(
        {
            "tokens": torch.zeros(1, 4, 8),
            "mask": torch.zeros(1, 4, dtype=torch.uint8),
            "pooled": torch.zeros(1, 8),
            "is_uncond": torch.ones(1, dtype=torch.uint8),
        },
        str(text_root / "empty_prompt.safetensors"),
    )
    ckpt = tmp_path / "tiny_cache.pt"
    _write_tiny_checkpoint(ckpt, data_root=str(data_root))
    out = tmp_path / "latent.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample.cli",
            "--ckpt",
            str(ckpt),
            "--out",
            str(out),
            "--n",
            "1",
            "--steps",
            "1",
            "--prompt",
            "cached empty negative prompt smoke",
            "--negative-prompt",
            "",
            "--cfg",
            "2.0",
            "--sampler",
            "flow_euler",
            "--seed",
            "11",
            "--device",
            "cpu",
            "--latent-only",
        ],
    )
    sample_cli_main()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["negative_prompt"] == ""
    assert meta["negative_prompt_source"] == "text_cache/empty_prompt"

def test_sample_cli_shift_override_is_used_in_sampler_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "tiny_shift.pt"
    _write_tiny_checkpoint(ckpt)
    out_a = tmp_path / "shift_1.pt"
    out_b = tmp_path / "shift_2.pt"

    base_argv = [
        "sample.cli",
        "--ckpt",
        str(ckpt),
        "--n",
        "1",
        "--steps",
        "3",
        "--prompt",
        "shift smoke",
        "--cfg",
        "1.0",
        "--sampler",
        "flow_euler",
        "--seed",
        "77",
        "--device",
        "cpu",
        "--latent-only",
    ]
    monkeypatch.setattr(sys, "argv", [*base_argv, "--out", str(out_a), "--shift", "1.0"])
    sample_cli_main()
    monkeypatch.setattr(sys, "argv", [*base_argv, "--out", str(out_b), "--shift", "2.0"])
    sample_cli_main()

    meta_a = json.loads(out_a.with_suffix(".json").read_text(encoding="utf-8"))
    meta_b = json.loads(out_b.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta_a["sampling_shift"] == 1.0
    assert meta_b["sampling_shift"] == 2.0
    assert torch.equal(torch.load(out_a, map_location="cpu"), torch.load(out_b, map_location="cpu"))

