from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest
import torch

from config.train import TrainConfig
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import EMA
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.dist import DistributedContext, create_distributed_context
from train.loop_mmdit_full import run_mmdit_training_loop
from train.runner import dry_run

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> TrainConfig:
    return TrainConfig.from_yaml(str(ROOT / "config" / name))


class _FakeNonMainAccelerator:
    is_main_process = False
    process_index = 1
    num_processes = 2
    device = torch.device("cpu")

    def prepare(self, *objects):
        return objects

    def unwrap_model(self, model):
        return getattr(model, "module", model)

    def wait_for_everyone(self):
        return None

    def reduce(self, tensor, reduction="mean"):
        assert reduction == "mean"
        return tensor


class _FakeMainAccelerator(_FakeNonMainAccelerator):
    is_main_process = True
    process_index = 0


def _tiny_cfg(out_dir: Path) -> TrainConfig:
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
            "distributed_backend": "none",
        }
    )


def _tiny_model() -> MMDiTFlowModel:
    return MMDiTFlowModel(
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


def _tiny_batch() -> TrainBatch:
    text = TextConditioning(
        tokens=torch.zeros(1, 3, 8),
        mask=torch.ones(1, 3, dtype=torch.bool),
        pooled=torch.zeros(1, 8),
    )
    return TrainBatch(x0=torch.randn(1, 4, 8, 8), text=text)


def test_distributed_smoke_profile_dry_runs_and_maps_nested_config() -> None:
    cfg = _load("train_distributed_smoke.yaml")
    assert cfg.distributed_backend == "accelerate"
    assert cfg.save_on_rank0_only is True
    assert cfg.distributed_metrics_aggregation is True
    assert cfg.fsdp_enabled is False
    with contextlib.redirect_stdout(io.StringIO()) as out:
        dry_run(cfg)
    assert "architecture=mmdit_rf" in out.getvalue()


def test_fsdp_template_is_present_but_disabled_by_default() -> None:
    cfg = _load("train_fsdp_template.yaml")
    assert cfg.distributed_backend == "accelerate"
    assert cfg.fsdp_enabled is False
    assert cfg.fsdp_min_hidden_dim >= 1024
    assert cfg.fsdp_auto_wrap_policy == "transformer_block"


def test_invalid_distributed_backend_and_enabled_fsdp_fail_early() -> None:
    with pytest.raises(ValueError, match="distributed_backend must be"):
        TrainConfig.from_dict({"distributed_backend": "mpi"})
    with pytest.raises(ValueError, match="fsdp.enabled=true is reserved"):
        TrainConfig.from_dict({"fsdp": {"enabled": True}})


def test_create_distributed_context_none_is_noop() -> None:
    cfg = TrainConfig.from_dict({})
    ctx = create_distributed_context(cfg, device=torch.device("cpu"))
    assert ctx.backend == "none"
    assert ctx.is_main_process is True
    assert ctx.world_size == 1
    x = object()
    assert ctx.prepare(x) == (x,)
    assert ctx.reduce_mean_float(3.0, device=torch.device("cpu")) == 3.0


def test_distributed_context_uses_fake_accelerator_rank_state() -> None:
    ctx = DistributedContext(backend="accelerate", accelerator=_FakeNonMainAccelerator(), device=torch.device("cpu"))
    assert ctx.is_main_process is False
    assert ctx.rank == 1
    assert ctx.world_size == 2
    assert ctx.should_write() is False
    assert ctx.reduce_mean_float(5.0, device=torch.device("cpu")) == 5.0


def test_training_loop_rank_gates_checkpoint_and_event_writes(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    model = _tiny_model()
    batch = _tiny_batch()
    checkpoint_dir = tmp_path / "checkpoints"
    dist = DistributedContext(backend="accelerate", accelerator=_FakeNonMainAccelerator(), device=torch.device("cpu"))

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
        empty_text=batch.text,
        start_step=0,
        text_metadata={"encoders": []},
        dist=dist,
    )

    assert not (tmp_path / "events.jsonl").exists()
    assert not (checkpoint_dir / "latest.pt").exists()
    assert not (tmp_path / "ckpt_latest.pt").exists()


def test_training_loop_main_rank_still_writes_outputs(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    model = _tiny_model()
    batch = _tiny_batch()
    checkpoint_dir = tmp_path / "checkpoints"
    dist = DistributedContext(backend="accelerate", accelerator=_FakeMainAccelerator(), device=torch.device("cpu"))

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
        empty_text=batch.text,
        start_step=0,
        text_metadata={"encoders": []},
        dist=dist,
    )

    assert (tmp_path / "events.jsonl").exists()
    assert (checkpoint_dir / "latest.pt").exists()
    assert (tmp_path / "ckpt_latest.pt").exists()
