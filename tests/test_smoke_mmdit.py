from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from scripts import smoke_mmdit_rf


def test_smoke_skips_eval_prompts_when_eval_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(
        f"""
architecture: mmdit_rf
mode: latent
image_size: 64
latent_channels: 4
latent_downsample_factor: 8
latent_patch_size: 2
objective: rectified_flow
prediction_type: flow_velocity
out_dir: {tmp_path / "run"}
eval_prompts_file: {tmp_path / "missing_prompts.txt"}
model:
  hidden_dim: 32
  depth: 1
  num_heads: 4
  mlp_ratio: 2.0
  qk_norm: true
  rms_norm: true
  swiglu: true
  adaln_zero: true
  pos_embed: sincos_2d
  double_stream_blocks: 1
  single_stream_blocks: 0
  gradient_checkpointing: false
text:
  enabled: true
  text_dim: 32
  pooled_dim: 32
training:
  eval_every: 0
  amp: false
flow:
  timestep_sampling: uniform
sampling:
  steps: 1
  cfg_scale: 1.0
  shift: 1.0
""",
        encoding="utf-8",
    )

    def fail_eval(*args, **kwargs):
        raise AssertionError("eval prompts should not be resolved when eval_every=0")

    def fake_batch(_cfg):
        x0 = torch.randn(1, 4, 8, 8)
        text = TextConditioning(
            tokens=torch.randn(1, 3, 32),
            mask=torch.ones(1, 3, dtype=torch.bool),
            pooled=torch.randn(1, 32),
            is_uncond=torch.zeros(1, dtype=torch.bool),
        )
        diagnostics = {
            "train_entries": 1,
            "val_entries": 0,
            "text_cache_entries": 1,
            "latent_cache_entries": 1,
            "first_md5": "synthetic",
        }
        return x0, text, diagnostics

    monkeypatch.setattr(smoke_mmdit_rf, "_resolve_eval_prompts", fail_eval)
    monkeypatch.setattr(smoke_mmdit_rf, "_load_first_batch", fake_batch)

    smoke_mmdit_rf.run(str(cfg_path))
