from __future__ import annotations

import json
from pathlib import Path

from train.run_dirs import prepare_train_run_structure


def test_prepare_train_run_structure_creates_reproducible_layout(tmp_path: Path) -> None:
    manifest = tmp_path / "cache_manifest_src.json"
    manifest.write_text('{"version":1}\n', encoding="utf-8")

    paths = prepare_train_run_structure(
        base_out_dir=tmp_path / "runs",
        cfg_dict={"architecture": "mmdit_rf", "hidden_dim": 64},
        run_name="dev64",
        cache_manifest_source=manifest,
    )

    assert paths.run_dir.parent == tmp_path / "runs"
    assert paths.checkpoints_dir == paths.run_dir / "checkpoints"
    assert paths.samples_dir.exists()
    assert paths.eval_dir.exists()
    assert (paths.run_dir / "config.yaml").exists()
    assert (paths.run_dir / "config_resolved.yaml").exists()
    assert paths.train_log_path.exists()
    assert json.loads(paths.cache_manifest_path.read_text(encoding="utf-8"))["version"] == 1

import torch

from config.train import TrainConfig
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import EMA
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.loop_mmdit_full import run_mmdit_training_loop


def _tiny_cfg_for_run_dir(out_dir: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "out_dir": str(out_dir),
            "batch_size": 1,
            "grad_accum_steps": 1,
            "max_steps": 1,
            "log_every": 1,
            "save_every": 1,
            "val_every": 0,
            "eval_every": 0,
            "lr": 1e-3,
            "warmup_steps": 0,
            "amp": False,
            "hidden_dim": 16,
            "depth": 1,
            "num_heads": 4,
            "double_stream_blocks": 1,
            "single_stream_blocks": 0,
            "latent_channels": 4,
            "latent_patch_size": 2,
            "text_dim": 8,
            "pooled_dim": 8,
            "gradient_checkpointing": False,
            "zero_init_final": False,
            "flow_timestep_sampling": "uniform",
            "cond_drop_prob": 0.0,
        }
    )


def test_training_loop_writes_new_checkpoint_layout_and_latest(tmp_path: Path) -> None:
    cfg = _tiny_cfg_for_run_dir(tmp_path)
    model = MMDiTFlowModel(
        MMDiTConfig(
            latent_channels=4,
            patch_size=2,
            hidden_dim=16,
            depth=1,
            num_heads=4,
            double_stream_blocks=1,
            single_stream_blocks=0,
            text_dim=8,
            pooled_dim=8,
            gradient_checkpointing=False,
            zero_init_final=False,
        )
    )
    text = TextConditioning(
        tokens=torch.zeros(1, 3, 8),
        mask=torch.ones(1, 3, dtype=torch.bool),
        pooled=torch.zeros(1, 8),
    )
    batch = TrainBatch(x0=torch.randn(1, 4, 8, 8), text=text)
    checkpoint_dir = tmp_path / "checkpoints"

    run_mmdit_training_loop(
        cfg=cfg,
        cfg_dict=cfg.to_dict(),
        model=model,
        dataloader=[batch],
        val_dataloader=None,
        objective=RectifiedFlowObjective(timestep_sampling="uniform"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        ema=EMA(model),
        device=torch.device("cpu"),
        out_dir=tmp_path,
        checkpoint_dir=checkpoint_dir,
        empty_text=text,
        start_step=0,
        text_metadata={"encoders": []},
    )

    assert (checkpoint_dir / "step_000001.pt").exists()
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "final.pt").exists()
    # Compatibility filenames remain available for existing scripts.
    assert (tmp_path / "ckpt_0000001.pt").exists()
    assert (tmp_path / "ckpt_latest.pt").exists()
