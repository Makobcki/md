from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from diffusion.utils import EMA, save_ckpt
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from sample.build import build_all
from sample.cli import _main_impl as sample_cli_main
from train.loop_mmdit import build_mmdit_checkpoint


def _tiny_cfg() -> dict:
    return {
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
        "zero_init_final": False,
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


def _write_ckpt_with_zero_ema(path: Path) -> tuple[str, torch.Tensor, torch.Tensor]:
    cfg = _tiny_cfg()
    torch.manual_seed(3)
    model = MMDiTFlowModel(MMDiTConfig.from_dict(cfg))
    ema = EMA(model)
    name, raw = next(iter(model.named_parameters()))
    for key in ema.shadow:
        ema.shadow[key] = torch.zeros_like(ema.shadow[key])
    ckpt = build_mmdit_checkpoint(model=model, ema=ema, optimizer=None, scheduler=None, step=4, cfg_dict=cfg)
    save_ckpt(str(path), ckpt)
    return name, raw.detach().clone(), torch.zeros_like(raw.detach())


def test_build_all_can_load_raw_or_ema_weights(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    name, raw, ema_zero = _write_ckpt_with_zero_ema(ckpt)

    raw_built = build_all(str(ckpt), torch.device("cpu"), latent_only=True, use_ema=False)
    ema_built = build_all(str(ckpt), torch.device("cpu"), latent_only=True, use_ema=True)
    raw_param = dict(raw_built.model.named_parameters())[name].detach()
    ema_param = dict(ema_built.model.named_parameters())[name].detach()

    assert torch.allclose(raw_param, raw)
    assert torch.allclose(ema_param, ema_zero)


def test_sample_metadata_records_ema_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "tiny.pt"
    _write_ckpt_with_zero_ema(ckpt)
    out = tmp_path / "sample.pt"
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
            "ema flag",
            "--cfg",
            "1.0",
            "--sampler",
            "flow_euler",
            "--seed",
            "2",
            "--device",
            "cpu",
            "--latent-only",
            "--no-ema",
        ],
    )
    sample_cli_main()
    metadata = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["use_ema"] is False
