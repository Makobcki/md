from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
torch = pytest.importorskip("torch")
from PIL import Image

from diffusion.utils import save_ckpt
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning
from sample.cli import _main_impl as sample_cli_main
from samplers.flow_euler import sample_flow_euler
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
        "dataset_tasks": {"txt2img": 1.0, "img2img": 0.0, "inpaint": 0.0},
    }


def _write_tiny_checkpoint(path: Path) -> None:
    cfg = _tiny_cfg()
    model = MMDiTFlowModel(MMDiTConfig.from_dict(cfg))
    ckpt = build_mmdit_checkpoint(model=model, ema=None, optimizer=None, scheduler=None, step=9, cfg_dict=cfg)
    save_ckpt(str(path), ckpt)


def _write_image(path: Path, *, gray: int = 128) -> None:
    Image.new("RGB", (512, 512), (gray, gray, gray)).save(path)


def _write_mask(path: Path) -> None:
    im = Image.new("L", (512, 512), 0)
    for y in range(192, 320):
        for x in range(192, 320):
            im.putpixel((x, y), 255)
    im.save(path)


def test_sample_cli_img2img_with_fake_vae_writes_output_and_strength_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "tiny.pt"
    image = tmp_path / "init.png"
    out = tmp_path / "img2img.png"
    _write_tiny_checkpoint(ckpt)
    _write_image(image)

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
            "2",
            "--prompt",
            "same cat, winter outfit",
            "--init-image",
            str(image),
            "--strength",
            "0.55",
            "--task",
            "img2img",
            "--sampler",
            "flow_euler",
            "--seed",
            "44",
            "--device",
            "cpu",
            "--fake-vae",
        ],
    )
    sample_cli_main()

    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert out.exists() and out.stat().st_size > 0
    assert meta["task"] == "img2img"
    assert meta["init_image"] == str(image)
    assert meta["strength"] == 0.55
    assert meta["seed"] == 44


def test_sample_cli_inpaint_requires_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "tiny.pt"
    image = tmp_path / "init.png"
    out = tmp_path / "inpaint.png"
    _write_tiny_checkpoint(ckpt)
    _write_image(image)

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
            "replace background with neon city",
            "--init-image",
            str(image),
            "--task",
            "inpaint",
            "--device",
            "cpu",
            "--fake-vae",
        ],
    )
    with pytest.raises(RuntimeError, match="requires --mask"):
        sample_cli_main()


def test_sample_cli_inpaint_with_fake_vae_writes_output_and_mask_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "tiny.pt"
    image = tmp_path / "init.png"
    mask = tmp_path / "mask.png"
    out = tmp_path / "inpaint.png"
    _write_tiny_checkpoint(ckpt)
    _write_image(image)
    _write_mask(mask)

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
            "2",
            "--prompt",
            "replace background with neon city",
            "--init-image",
            str(image),
            "--mask",
            str(mask),
            "--task",
            "inpaint",
            "--sampler",
            "flow_heun",
            "--seed",
            "45",
            "--device",
            "cpu",
            "--fake-vae",
        ],
    )
    sample_cli_main()

    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert out.exists() and out.stat().st_size > 0
    assert meta["task"] == "inpaint"
    assert meta["init_image"] == str(image)
    assert meta["mask"] == str(mask)
    assert meta["sampler"] == "flow_heun"


class _ConstantFlow(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))
        self.value = float(value)

    def forward(self, x, t, text, **kwargs):
        del t, text, kwargs
        return torch.full_like(x, self.value)


def _cond() -> TextConditioning:
    return TextConditioning(torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 4))


def test_img2img_sampler_strength_zero_starts_and_ends_at_source_for_zero_velocity() -> None:
    source = torch.randn(1, 4, 4, 4)
    out = sample_flow_euler(
        _ConstantFlow(0.0),
        tuple(source.shape),
        _cond(),
        steps=3,
        noise=source.clone(),
        start_t=0.0,
        source_latent=source,
        task="img2img",
    )
    assert torch.equal(out, source)


def test_img2img_sampler_strength_one_matches_txt2img_when_model_ignores_source() -> None:
    model = _ConstantFlow(0.25)
    noise = torch.randn(1, 4, 4, 4)
    source = torch.randn(1, 4, 4, 4)
    txt = sample_flow_euler(model, tuple(noise.shape), _cond(), steps=4, noise=noise.clone(), start_t=1.0, task="txt2img")
    img = sample_flow_euler(
        model,
        tuple(noise.shape),
        _cond(),
        steps=4,
        noise=noise.clone(),
        start_t=1.0,
        source_latent=source,
        task="img2img",
    )
    assert torch.allclose(img, txt)


def test_inpaint_sampler_changes_masked_region_and_preserves_unmasked_region() -> None:
    source = torch.full((1, 4, 4, 4), 3.0)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 1.0
    out = sample_flow_euler(
        _ConstantFlow(1.0),
        tuple(source.shape),
        _cond(),
        steps=2,
        noise=torch.zeros_like(source),
        source_latent=source,
        mask=mask,
        task="inpaint",
    )
    assert torch.equal(out * (1.0 - mask), source * (1.0 - mask))
    assert not torch.equal(out * mask, source * mask)

def test_sample_cli_control_with_fake_vae_writes_output_and_control_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "tiny.pt"
    control = tmp_path / "control.png"
    out = tmp_path / "control.png.out.png"
    _write_tiny_checkpoint(ckpt)
    _write_image(control, gray=200)

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
            "2",
            "--prompt",
            "edge guided cat",
            "--control-image",
            str(control),
            "--control-strength",
            "0.75",
            "--task",
            "control",
            "--sampler",
            "flow_euler",
            "--seed",
            "46",
            "--device",
            "cpu",
            "--fake-vae",
        ],
    )
    sample_cli_main()

    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert out.exists() and out.stat().st_size > 0
    assert meta["task"] == "control"
    assert meta["control_image"] == str(control)
    assert meta["control_strength"] == 0.75
