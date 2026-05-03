from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from sample.cli import _metadata_sidecar_path, _sample_metadata, _write_sample_metadata


def test_sample_metadata_helpers_write_reproducible_json(tmp_path: Path) -> None:
    out = tmp_path / "sample.png"
    sidecar = _metadata_sidecar_path(out)
    assert sidecar == tmp_path / "sample.json"

    args = Namespace(
        ckpt="./runs/mmdit_smoke/ckpt_final.pt",
        prompt="1girl, simple background",
        neg_prompt="low quality",
        steps=2,
        cfg=1,
        n=1,
        task="txt2img",
        strength=1.0,
        init_image="",
        mask="",
        latent_only=False,
        fake_vae=False,
    )
    built = Namespace(
        cfg={
            "architecture": "mmdit_rf",
            "objective": "rectified_flow",
            "prediction_type": "flow_velocity",
            "image_size": 512,
            "latent_channels": 4,
            "latent_downsample_factor": 8,
            "hidden_dim": 64,
            "depth": 1,
            "num_heads": 4,
            "vae_pretrained": "./vae_sd_mse",
            "sampling_shift": 3.0,
            "text": {"backend": "fake", "text_dim": 32, "pooled_dim": 32, "encoders": []},
        },
        image_channels=4,
        h=512,
        w=512,
        latent_h=64,
        latent_w=64,
        checkpoint_step=123,
        checkpoint_metadata={},
        text_encoder=Namespace(metadata=lambda: {"backend": "fake", "encoders": [], "text_dim": 32, "pooled_dim": 32}),
    )
    metadata = _sample_metadata(args, built, sampler="flow_heun", seed=42)
    written = _write_sample_metadata(out, metadata)

    assert written == sidecar
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["checkpoint_path"] == "./runs/mmdit_smoke/ckpt_final.pt"
    assert payload["checkpoint_step"] == 123
    assert payload["architecture"] == "mmdit_rf"
    assert payload["objective"] == "rectified_flow"
    assert payload["prediction_type"] == "flow_velocity"
    assert payload["prompt"] == "1girl, simple background"
    assert payload["negative_prompt"] == "low quality"
    assert payload["sampler"] == "flow_heun"
    assert payload["steps"] == 2
    assert payload["cfg"] == 1.0
    assert payload["seed"] == 42
    assert payload["sampling_shift"] == 3.0
    assert payload["image_size"] == [512, 512]
    assert payload["latent_shape"] == [4, 64, 64]
    assert payload["model_config"]["hidden_dim"] == 64
    assert payload["model_config"]["depth"] == 1
    assert payload["model_config"]["num_heads"] == 4
    assert payload["vae_config"]["pretrained"] == "./vae_sd_mse"
    assert payload["text_encoder_config"] == {"backend": "fake", "encoders": [], "text_dim": 32, "pooled_dim": 32}
