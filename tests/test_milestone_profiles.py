from __future__ import annotations

import contextlib
import io
from pathlib import Path

from config.train import TrainConfig
from train.runner import dry_run

ROOT = Path(__file__).resolve().parents[1]


def _cfg(name: str) -> TrainConfig:
    return TrainConfig.from_yaml(str(ROOT / "config" / name))


def test_milestone_a_profile_matches_target_shape_and_dry_runs() -> None:
    cfg = _cfg("train_milestone_a.yaml")
    assert cfg.hidden_dim == 384
    assert cfg.depth == 4
    assert cfg.num_heads == 6
    assert cfg.double_stream_blocks == 3
    assert cfg.single_stream_blocks == 1
    assert cfg.batch_size == 2
    assert cfg.grad_accum_steps == 8
    assert cfg.max_steps == 2000
    assert cfg.dataset_tasks == {"txt2img": 1.0, "img2img": 0.0, "inpaint": 0.0, "control": 0.0}
    with contextlib.redirect_stdout(io.StringIO()) as out:
        dry_run(cfg)
    assert "architecture=mmdit_rf" in out.getvalue()


def test_milestone_b_and_c_profiles_expand_real_text_presets() -> None:
    b = _cfg("train_milestone_b.yaml")
    c = _cfg("train_milestone_c.yaml")
    assert b.hidden_dim == 768
    assert b.depth == 12
    assert b.num_heads == 12
    assert b.double_stream_blocks == 8
    assert b.single_stream_blocks == 4
    assert b.max_steps == 50000
    assert [e["name"] for e in b.extra["text"]["encoders"]] == ["clip_l", "t5_base"]

    assert c.hidden_dim == 1024
    assert c.depth == 24
    assert c.num_heads == 16
    assert c.double_stream_blocks == 16
    assert c.single_stream_blocks == 8
    assert c.max_steps >= 300000
    assert [e["name"] for e in c.extra["text"]["encoders"]] == ["clip_l", "t5_base"]
