from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from diffusion.io.ckpt import load_ckpt, normalize_state_dict_for_keys, normalize_state_dict_for_model, save_ckpt
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import EMA
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.checkpoint_mmdit import validate_mmdit_checkpoint_compatibility
from train.loop_mmdit_full import _build_ckpt, _loss_and_bins


def _cfg() -> TrainConfig:
    return TrainConfig.from_dict(
        {
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
            "amp": False,
        }
    )


def _model() -> MMDiTFlowModel:
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


def _batch() -> TrainBatch:
    return TrainBatch(
        x0=torch.randn(2, 4, 8, 8),
        text=TextConditioning(
            tokens=torch.randn(2, 3, 8),
            mask=torch.ones(2, 3, dtype=torch.bool),
            pooled=torch.randn(2, 8),
        ),
    )


def _train_one_step(model, optimizer, ema, objective, empty_text, step: int) -> int:
    optimizer.zero_grad(set_to_none=True)
    loss, _ = _loss_and_bins(
        model=model,
        objective=objective,
        batch=_batch(),
        empty_text=empty_text,
        cfg_drop_prob=0.0,
        use_amp=False,
        amp_dtype=torch.bfloat16,
    )
    assert torch.isfinite(loss)
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
    optimizer.step()
    ema.update(model)
    return step + 1


def test_checkpoint_resume_restores_train_state_and_continues(tmp_path: Path) -> None:
    torch.manual_seed(7)
    cfg = _cfg()
    cfg_dict = cfg.to_dict()
    objective = RectifiedFlowObjective(timestep_sampling="uniform")
    empty_text = TextConditioning(torch.zeros(2, 3, 8), torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 8))

    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema = EMA(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    step = 0
    for _ in range(2):
        step = _train_one_step(model, optimizer, ema, objective, empty_text, step)
    assert step == 2

    ckpt = _build_ckpt(
        cfg=cfg,
        cfg_dict=cfg_dict,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        step=step,
        text_metadata={"encoders": []},
    )
    path = tmp_path / "ckpt.pt"
    save_ckpt(str(path), ckpt)

    model2 = _model()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    ema2 = EMA(model2)
    scaler2 = torch.amp.GradScaler("cuda", enabled=False)
    loaded = load_ckpt(str(path), torch.device("cpu"))
    validate_mmdit_checkpoint_compatibility(loaded, cfg_dict)
    model2.load_state_dict(normalize_state_dict_for_model(loaded["model"], model2, name="model"), strict=True)
    optimizer2.load_state_dict(loaded["optimizer"])
    scaler2.load_state_dict(loaded["scaler"])
    ema_state = normalize_state_dict_for_keys(loaded["ema"], ema2.shadow.keys(), name="ema")
    ema2.shadow = {key: value.clone() for key, value in ema_state.items()}
    resumed_step = int(loaded["step"])
    assert resumed_step == 2

    for _ in range(2):
        resumed_step = _train_one_step(model2, optimizer2, ema2, objective, empty_text, resumed_step)
    assert resumed_step == 4
