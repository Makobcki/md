from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from model.text.conditioning import TextConditioning
from scripts import smoke_mmdit_rf


def _install_fast_smoke_runtime(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict]]:
    saved: dict[str, dict] = {}
    sampler_calls: list[dict] = []

    class FastModel(torch.nn.Module):
        def __init__(self, _cfg) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x0, _t, _text):
            return x0 + self.weight * 0.0

    def fake_training_step_mmdit(*, model, objective, batch, amp_enabled):
        return model.weight.sum() * 0.0 + batch.x0.sum() * 0.0

    def fake_save_ckpt(path, payload):
        saved[str(path)] = payload
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fast-smoke-checkpoint")

    def fake_load_ckpt(path, device):
        return saved[str(path)]

    def fake_sampler(**kwargs):
        sampler_calls.append(kwargs)
        return torch.zeros(kwargs["shape"])

    monkeypatch.setattr(smoke_mmdit_rf, "MMDiTFlowModel", FastModel)
    monkeypatch.setattr(smoke_mmdit_rf, "training_step_mmdit", fake_training_step_mmdit)
    monkeypatch.setattr(smoke_mmdit_rf, "save_ckpt", fake_save_ckpt)
    monkeypatch.setattr(smoke_mmdit_rf, "load_ckpt", fake_load_ckpt)
    monkeypatch.setattr(smoke_mmdit_rf, "sample_flow_heun", fake_sampler)
    return {"sampler_calls": sampler_calls}


def test_smoke_skips_eval_prompts_when_eval_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "smoke.kdl"
    cfg_path.write_text(
        f"""
config target="train" version=2 {{
  model {{
    family "mmdit"
    variant "smoke"
    architecture {{
      image_size 64
      latent_channels 4
      patch_size 2
      hidden_size 32
      depth 1
      num_heads 4
      mlp_ratio 2.0
      qk_norm true
      rms_norm true
      swiglu true
      adaln_zero true
      double_stream_blocks 1
      single_stream_blocks 0
      gradient_checkpointing false
    }}
    diffusion {{
      objective "rectified_flow"
      prediction_type "velocity"
    }}
  }}
  text {{
    enabled true
    text_dim 32
    pooled_dim 32
  }}
  training {{
    eval_every 0
    amp false
  }}
  flow {{
    timestep_sampling "uniform"
  }}
  sampling {{
    steps 1
    cfg_scale 1.0
    shift 1.0
  }}
  output {{
    dir "{tmp_path / "run"}"
  }}
  eval_prompts_file "{tmp_path / "missing_prompts.txt"}"
}}
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

    runtime = _install_fast_smoke_runtime(monkeypatch)
    monkeypatch.setattr(smoke_mmdit_rf, "_resolve_eval_prompts", fail_eval)
    monkeypatch.setattr(smoke_mmdit_rf, "_load_first_batch", fake_batch)

    smoke_mmdit_rf.run(str(cfg_path))

    assert runtime["sampler_calls"]


def test_synthetic_smoke_does_not_touch_dataset_or_caches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "synthetic.kdl"
    cfg_path.write_text(
        f"""
config target="train" version=2 {{
  model {{
    family "mmdit"
    variant "smoke"
    architecture {{
      image_size 64
      latent_channels 4
      patch_size 2
      hidden_size 32
      depth 1
      num_heads 4
      mlp_ratio 2.0
      qk_norm true
      rms_norm true
      swiglu true
      adaln_zero true
      double_stream_blocks 1
      single_stream_blocks 0
      gradient_checkpointing false
    }}
    diffusion {{
      objective "rectified_flow"
      prediction_type "velocity"
    }}
  }}
  text {{
    enabled true
    text_dim 32
    pooled_dim 32
  }}
  training {{
    eval_every 500
    amp false
  }}
  flow {{
    timestep_sampling "uniform"
  }}
  sampling {{
    steps 99
    cfg_scale 1.0
    shift 1.0
  }}
  output {{
    dir "{tmp_path / "run"}"
  }}
  eval_prompts_file "{tmp_path / "missing_prompts.txt"}"
}}
""",
        encoding="utf-8",
    )

    def fail_eval(*args, **kwargs):
        raise AssertionError("synthetic smoke should not resolve eval prompts")

    def fail_batch(_cfg):
        raise AssertionError("synthetic smoke should not read dataset, text cache, or latent cache")

    runtime = _install_fast_smoke_runtime(monkeypatch)
    monkeypatch.setattr(smoke_mmdit_rf, "_resolve_eval_prompts", fail_eval)
    monkeypatch.setattr(smoke_mmdit_rf, "_load_first_batch", fail_batch)

    smoke_mmdit_rf.run(str(cfg_path), synthetic=True)

    assert (tmp_path / "run" / "synthetic_smoke_ckpt.pt").is_file()
    assert runtime["sampler_calls"]
    assert runtime["sampler_calls"][0]["steps"] == 2
