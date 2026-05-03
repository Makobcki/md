from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch

from diffusion.objectives import RectifiedFlowObjective, rectified_flow_loss
from diffusion.utils import EMA
from model.mmdit import MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from train.checkpoint_mmdit import build_mmdit_checkpoint_metadata


def as_train_batch(batch) -> TrainBatch:
    if isinstance(batch, TrainBatch):
        return batch
    if isinstance(batch, dict):
        return TrainBatch(**batch)
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        x0 = batch[0]
        if len(batch) >= 2 and isinstance(batch[1], TextConditioning):
            text = batch[1]
        else:
            b = x0.shape[0]
            text = TextConditioning(
                tokens=torch.zeros(b, 1, 1024, dtype=x0.dtype),
                mask=torch.ones(b, 1, dtype=torch.bool),
                pooled=torch.zeros(b, 1024, dtype=x0.dtype),
                is_uncond=torch.ones(b, dtype=torch.bool),
            )
        return TrainBatch(x0=x0, text=text)
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def apply_cfg_dropout(
    text: TextConditioning,
    *,
    empty_text: Optional[TextConditioning],
    drop_prob: float,
) -> TextConditioning:
    if empty_text is None or drop_prob <= 0:
        return text
    b = text.tokens.shape[0]
    drop = torch.rand(b, device=text.tokens.device) < float(drop_prob)
    return text.replace_where(drop, empty_text)


def training_step_mmdit(
    *,
    model: MMDiTFlowModel,
    objective: RectifiedFlowObjective,
    batch: TrainBatch,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    empty_text: Optional[TextConditioning] = None,
    cfg_drop_prob: float = 0.0,
) -> torch.Tensor:
    x0 = batch.x0
    text = apply_cfg_dropout(batch.text, empty_text=empty_text, drop_prob=cfg_drop_prob)
    train_tuple = objective.sample_training_tuple(x0)
    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
        pred = model(
            x=train_tuple.xt,
            t=train_tuple.t,
            text=text,
            source_latent=batch.source_latent,
            mask=batch.mask,
            control_latents=batch.control_latents,
            task=batch.task,
        )
        return rectified_flow_loss(pred, train_tuple.target, train_tuple.weight, mask=batch.mask)


def build_mmdit_checkpoint(
    *,
    model: torch.nn.Module,
    ema: Optional[EMA],
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    step: int,
    cfg_dict: dict,
) -> dict:
    class _CfgProxy:
        def __init__(self, data: dict) -> None:
            self.latent_channels = int(data.get("latent_channels", 4))
            self.latent_patch_size = int(data.get("latent_patch_size", data.get("patch_size", 2)))
            self.hidden_dim = int(data.get("hidden_dim", data.get("model", {}).get("hidden_dim", 1024)))
            self.depth = int(data.get("depth", data.get("model", {}).get("depth", 24)))
            self.num_heads = int(data.get("num_heads", data.get("model", {}).get("num_heads", 16)))
            self.double_stream_blocks = int(data.get("double_stream_blocks", data.get("model", {}).get("double_stream_blocks", 16)))
            self.single_stream_blocks = int(data.get("single_stream_blocks", data.get("model", {}).get("single_stream_blocks", 8)))
            self.pos_embed = str(data.get("pos_embed", data.get("model", {}).get("pos_embed", "rope_2d")))
            self.text_dim = int(data.get("text_dim", data.get("text", {}).get("text_dim", 1024)))
            self.pooled_dim = int(data.get("pooled_dim", data.get("text", {}).get("pooled_dim", 1024)))
            self.vae_pretrained = str(data.get("vae_pretrained", data.get("vae", {}).get("pretrained", "")))
            self.vae_scaling_factor = float(data.get("vae_scaling_factor", data.get("vae", {}).get("scaling_factor", 0.18215)))
            flow = data.get("flow", {})
            self.flow_timestep_sampling = str(data.get("flow_timestep_sampling", flow.get("timestep_sampling", "logit_normal")))
            self.flow_logit_mean = float(data.get("flow_logit_mean", flow.get("logit_mean", 0.0)))
            self.flow_logit_std = float(data.get("flow_logit_std", flow.get("logit_std", 1.0)))
            self.flow_loss_weighting = str(data.get("flow_loss_weighting", flow.get("loss_weighting", "none")))
            self.flow_timestep_shift = float(data.get("flow_timestep_shift", flow.get("timestep_shift", flow.get("shift", 1.0))))
            self.flow_train_t_min = float(data.get("flow_train_t_min", flow.get("train_t_min", 0.0)))
            self.flow_train_t_max = float(data.get("flow_train_t_max", flow.get("train_t_max", 1.0)))

    metadata = build_mmdit_checkpoint_metadata(
        cfg=_CfgProxy(cfg_dict),
        cfg_dict=dict(cfg_dict),
        step=step,
        text_metadata={"encoders": cfg_dict.get("text", {}).get("encoders", [])},
        dataset_hash=str(cfg_dict.get("dataset_hash", "")),
    )
    return {
        "model": model.state_dict(),
        "ema": ema.shadow if ema is not None else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": int(step),
        "cfg": dict(cfg_dict),
        "metadata": metadata,
        "architecture": "mmdit_rf",
        "objective": "rectified_flow",
        "prediction_type": "flow_velocity",
        "vae": metadata["vae_config"],
        "text_encoders": metadata["text_config"].get("encoders", []),
    }


def run_minimal_mmdit_loop(
    *,
    model: MMDiTFlowModel,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    objective: RectifiedFlowObjective,
    device: torch.device,
    max_steps: int,
    grad_accum_steps: int = 1,
    ema: Optional[EMA] = None,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    empty_text: Optional[TextConditioning] = None,
    cfg_drop_prob: float = 0.0,
) -> list[float]:
    model.train()
    use_amp = bool(amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    losses: list[float] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        for raw in dataloader:
            batch = as_train_batch(raw)
            batch = TrainBatch(
                x0=batch.x0.to(device),
                text=TextConditioning(
                    tokens=batch.text.tokens.to(device),
                    mask=batch.text.mask.to(device),
                    pooled=batch.text.pooled.to(device),
                    is_uncond=batch.text.is_uncond.to(device) if batch.text.is_uncond is not None else None,
                    token_types=batch.text.token_types.to(device) if batch.text.token_types is not None else None,
                ),
                source_latent=batch.source_latent.to(device) if batch.source_latent is not None else None,
                mask=batch.mask.to(device) if batch.mask is not None else None,
                control_latents=batch.control_latents.to(device) if batch.control_latents is not None else None,
                task=batch.task,
                metadata=batch.metadata,
            )
            loss = training_step_mmdit(
                model=model,
                objective=objective,
                batch=batch,
                amp_enabled=use_amp,
                amp_dtype=amp_dtype,
                empty_text=empty_text,
                cfg_drop_prob=cfg_drop_prob,
            ) / max(int(grad_accum_steps), 1)
            scaler.scale(loss).backward()
            if (step + 1) % max(int(grad_accum_steps), 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            losses.append(float(loss.detach().cpu()) * max(int(grad_accum_steps), 1))
            step += 1
            if step >= max_steps:
                break
    return losses
