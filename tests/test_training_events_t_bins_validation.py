from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import EMA
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.loop_mmdit_full import _run_validation, run_mmdit_training_loop


def _tiny_cfg(out_dir: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "out_dir": str(out_dir),
            "batch_size": 2,
            "grad_accum_steps": 1,
            "max_steps": 2,
            "log_every": 1,
            "save_every": 0,
            "val_every": 1,
            "val_batches": 1,
            "lr": 1e-3,
            "warmup_steps": 0,
            "amp": False,
            "hidden_dim": 32,
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


def _tiny_model() -> MMDiTFlowModel:
    return MMDiTFlowModel(
        MMDiTConfig(
            latent_channels=4,
            patch_size=2,
            hidden_dim=32,
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


def _batch(seed: int = 0) -> TrainBatch:
    g = torch.Generator().manual_seed(seed)
    b = 2
    return TrainBatch(
        x0=torch.randn(b, 4, 8, 8, generator=g),
        text=TextConditioning(
            tokens=torch.randn(b, 3, 8, generator=g),
            mask=torch.ones(b, 3, dtype=torch.bool),
            pooled=torch.randn(b, 8, generator=g),
        ),
    )


def test_training_loop_writes_train_and_eval_t_bin_events(tmp_path: Path) -> None:
    torch.manual_seed(123)
    cfg = _tiny_cfg(tmp_path)
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    objective = RectifiedFlowObjective(timestep_sampling="uniform")
    empty_text = TextConditioning(
        tokens=torch.zeros(2, 3, 8),
        mask=torch.ones(2, 3, dtype=torch.bool),
        pooled=torch.zeros(2, 8),
    )

    run_mmdit_training_loop(
        cfg=cfg,
        cfg_dict=cfg.to_dict(),
        model=model,
        dataloader=[_batch(1), _batch(2), _batch(3)],
        val_dataloader=[_batch(4)],
        objective=objective,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        ema=EMA(model),
        device=torch.device("cpu"),
        out_dir=tmp_path,
        empty_text=empty_text,
        start_step=0,
        text_metadata={"encoders": []},
    )

    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    train_events = [event for event in events if event.get("type") == "train"]
    eval_events = [event for event in events if event.get("type") == "eval"]
    assert train_events
    assert eval_events
    assert "train_loss" in train_events[-1]
    assert any(key.startswith("loss_t_bin_") for key in train_events[-1])
    assert "val_loss" in eval_events[-1]
    assert any(key.startswith("val_loss_t_bin_") for key in eval_events[-1])

    # The historical metrics/events.jsonl path is still populated for compatibility.
    assert (tmp_path / "metrics" / "events.jsonl").exists()


def test_validation_runs_no_grad_and_restores_train_mode() -> None:
    torch.manual_seed(321)
    model = _tiny_model()
    model.train()

    val_loss, val_bins = _run_validation(
        model=model,
        objective=RectifiedFlowObjective(timestep_sampling="uniform"),
        dataloader=[_batch(5)],
        device=torch.device("cpu"),
        max_batches=1,
        use_amp=False,
        amp_dtype=torch.bfloat16,
    )

    assert model.training is True
    assert val_loss is not None and torch.isfinite(torch.tensor(val_loss))
    assert any(key.startswith("val_loss_t_bin_") for key in val_bins)
    assert all(param.grad is None for param in model.parameters())
