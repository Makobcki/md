from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from diffusion.utils import EMA, save_ckpt
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from train.eval_grid import run_fixed_seed_eval_grids
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
        "dataset_tasks": {"txt2img": 1.0},
    }


def _write_ckpt(path: Path) -> None:
    cfg = _tiny_cfg()
    torch.manual_seed(1)
    model = MMDiTFlowModel(MMDiTConfig.from_dict(cfg))
    ema = EMA(model)
    ckpt = build_mmdit_checkpoint(model=model, ema=ema, optimizer=None, scheduler=None, step=7, cfg_dict=cfg)
    save_ckpt(str(path), ckpt)


def _prompt_root(tmp_path: Path) -> Path:
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "core.txt").write_text("a cat\n\na dog\n", encoding="utf-8")
    (root / "style.txt").write_text("watercolor castle\n", encoding="utf-8")
    return root


def test_fixed_seed_eval_grid_writes_metadata_events_and_is_reproducible(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    _write_ckpt(ckpt)
    prompt_root = _prompt_root(tmp_path)

    out1 = tmp_path / "run1"
    result1 = run_fixed_seed_eval_grids(
        ckpt=ckpt,
        out_dir=out1,
        prompt_root=prompt_root,
        prompt_sets=["core", "style"],
        count_per_set=1,
        seed=123,
        sampler="flow_heun",
        steps=2,
        cfg=1.5,
        n_per_prompt=1,
        device="cpu",
        fake_vae=True,
    )
    grid1 = out1 / "eval" / "step_000007" / "core_grid.png"
    assert grid1.exists()
    assert (out1 / "eval" / "step_000007" / "style_grid.png").exists()
    metadata = json.loads((out1 / "eval" / "step_000007" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint_step"] == 7
    assert metadata["base"] == {"sampler": "flow_heun", "steps": 2, "cfg": 1.5, "seed": 123, "shift": 1.0}
    assert metadata["prompt_sets"] == ["core", "style"]
    assert result1.events
    assert result1.events[0]["type"] == "sample"
    assert result1.events[0]["prompt_set"] == "core"
    assert result1.events[0]["sampler"] == "flow_heun"
    assert result1.events[0]["steps"] == 2
    assert result1.events[0]["cfg"] == 1.5
    assert result1.events[0]["seed"] == 123
    assert result1.events[0]["path"] == "eval/step_000007/core_grid.png"

    out2 = tmp_path / "run2"
    run_fixed_seed_eval_grids(
        ckpt=ckpt,
        out_dir=out2,
        prompt_root=prompt_root,
        prompt_sets=["core"],
        count_per_set=1,
        seed=123,
        sampler="flow_heun",
        steps=2,
        cfg=1.5,
        n_per_prompt=1,
        device="cpu",
        fake_vae=True,
    )
    grid2 = out2 / "eval" / "step_000007" / "core_grid.png"
    assert grid1.read_bytes() == grid2.read_bytes()


def test_eval_cfg_step_and_sampler_sweeps_write_per_variant_metadata(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    _write_ckpt(ckpt)
    prompt_root = _prompt_root(tmp_path)
    out = tmp_path / "sweep"

    result = run_fixed_seed_eval_grids(
        ckpt=ckpt,
        out_dir=out,
        prompt_root=prompt_root,
        prompt_sets=["core"],
        count_per_set=1,
        seed=5,
        sampler="flow_euler",
        steps=1,
        cfg=1.0,
        n_per_prompt=1,
        device="cpu",
        fake_vae=True,
        cfg_values=[1.0, 2.5],
        step_values=[1, 2],
        sampler_values=["flow_euler", "flow_heun"],
    )

    step_dir = out / "eval" / "step_000007"
    assert (step_dir / "base" / "core_grid.png").exists()
    assert (step_dir / "cfg_2_50" / "core_grid.png").exists()
    assert (step_dir / "steps_002" / "metadata.json").exists()
    assert (step_dir / "flow_heun_steps_002_cfg_2_50" / "metadata.json").exists()
    metadata = json.loads((step_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["sweep"]["cfg"] == [1.0, 2.5]
    assert metadata["sweep"]["steps"] == [1, 2]
    assert metadata["sweep"]["samplers"] == ["flow_euler", "flow_heun"]
    assert len(result.events) == 8
    events = [json.loads(line) for line in (step_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(events) == 8
    assert all(event["type"] == "sample" for event in events)


def test_eval_shift_sweep_writes_shift_metadata_and_events(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    _write_ckpt(ckpt)
    prompt_root = tmp_path / "prompts"
    prompt_root.mkdir()
    (prompt_root / "core.txt").write_text("a cat\n", encoding="utf-8")
    result = run_fixed_seed_eval_grids(
        ckpt=ckpt,
        out_dir=tmp_path / "run",
        prompt_root=prompt_root,
        prompt_sets=["core"],
        count_per_set=1,
        seed=3,
        sampler="flow_euler",
        steps=1,
        cfg=1.0,
        n_per_prompt=1,
        shift=1.0,
        shift_values=[1.0, 2.0],
        device="cpu",
        fake_vae=True,
        use_ema=False,
    )
    assert result.metadata["sweep"]["shift"] == [1.0, 2.0]
    assert len(result.metadata["runs"]) == 2
    step_dir = tmp_path / "run" / "eval" / "step_000007"
    assert (step_dir / "base" / "metadata.json").exists()
    assert (step_dir / "shift_2_00" / "metadata.json").exists()
    events = [json.loads(line) for line in (step_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {event["shift"] for event in events} == {1.0, 2.0}

def test_eval_resolution_writes_separate_resolution_directory_and_metadata(tmp_path: Path) -> None:
    ckpt = tmp_path / "tiny.pt"
    _write_ckpt(ckpt)
    prompt_root = _prompt_root(tmp_path)
    out = tmp_path / "resolution_eval"

    result = run_fixed_seed_eval_grids(
        ckpt=ckpt,
        out_dir=out,
        prompt_root=prompt_root,
        prompt_sets=["core"],
        count_per_set=1,
        seed=321,
        sampler="flow_euler",
        steps=2,
        cfg=1.0,
        n_per_prompt=1,
        resolution=64,
        device="cpu",
        fake_vae=True,
    )

    step_dir = out / "eval" / "eval_64" / "step_000007"
    assert (step_dir / "core_grid.png").exists()
    metadata = json.loads((step_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["resolution"] == [64, 64]
    assert metadata["base"]["resolution"] == [64, 64]
    assert result.events[0]["resolution"] == [64, 64]
    assert result.events[0]["path"] == "eval/eval_64/step_000007/core_grid.png"
