from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from config.train import TrainConfig
from diffusion.io.events import EventBus, JsonlFileSink, StdoutJsonSink
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import EMA
from model.mmdit import MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from model.text.pretrained import FrozenTextEncoderBundle
from diffusion.vae import VAEWrapper
from train.checkpoint import _prune_checkpoints, save_ckpt
from train.eval_mmdit import run_mmdit_eval_sampling
from train.schedulers import _apply_lr, _compute_lr
from train.webui import _is_webui_mode, _webui_metrics_path


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf values")


def _move_text(text: TextConditioning, device: torch.device) -> TextConditioning:
    return TextConditioning(
        tokens=text.tokens.to(device),
        mask=text.mask.to(device),
        pooled=text.pooled.to(device),
        is_uncond=text.is_uncond.to(device) if text.is_uncond is not None else None,
    )


def _move_batch(batch: TrainBatch, device: torch.device) -> TrainBatch:
    return TrainBatch(
        x0=batch.x0.to(device=device, dtype=torch.float32),
        text=_move_text(batch.text, device),
        source_latent=batch.source_latent.to(device=device, dtype=torch.float32)
        if batch.source_latent is not None
        else None,
        mask=batch.mask.to(device=device, dtype=torch.float32) if batch.mask is not None else None,
        task=batch.task,
        metadata=batch.metadata,
    )


def _replace_text_where(text: TextConditioning, empty_text: TextConditioning, drop_prob: float) -> TextConditioning:
    if drop_prob <= 0:
        return text
    drop = torch.rand(text.tokens.shape[0], device=text.tokens.device) < float(drop_prob)
    return text.replace_where(drop, empty_text)


def _loss_and_bins(
    *,
    model: MMDiTFlowModel,
    objective: RectifiedFlowObjective,
    batch: TrainBatch,
    empty_text: TextConditioning,
    cfg_drop_prob: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, float]]:
    train_tuple = objective.sample_training_tuple(batch.x0)
    text = _replace_text_where(batch.text, empty_text, cfg_drop_prob)
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        pred = model(
            x=train_tuple.xt,
            t=train_tuple.t,
            text=text,
            source_latent=batch.source_latent,
            mask=batch.mask,
            task=batch.task,
        )
        _assert_finite("mmdit_pred", pred)
        per = (pred - train_tuple.target.to(dtype=pred.dtype)).pow(2).mean(dim=[1, 2, 3])
        loss = (per * train_tuple.weight.to(device=per.device, dtype=per.dtype)).mean()

    stats: dict[str, float] = {}
    t = train_tuple.t.detach()
    per_f = per.detach().float()
    for idx in range(10):
        lo = idx / 10.0
        hi = (idx + 1) / 10.0
        if idx == 9:
            mask = (t >= lo) & (t <= hi)
        else:
            mask = (t >= lo) & (t < hi)
        if mask.any():
            stats[f"loss_t_{idx}_{idx + 1}"] = float(per_f[mask].mean().cpu())
    return loss, stats


@torch.no_grad()
def _run_validation(
    *,
    model: MMDiTFlowModel,
    objective: RectifiedFlowObjective,
    dataloader,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[Optional[float], dict[str, float]]:
    if dataloader is None or max_batches <= 0:
        return None, {}
    was_training = model.training
    model.eval()
    losses: list[float] = []
    bins: dict[str, list[float]] = {}
    try:
        for batch_idx, raw in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            batch = _move_batch(raw, device)
            train_tuple = objective.sample_training_tuple(batch.x0)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred = model(
                    x=train_tuple.xt,
                    t=train_tuple.t,
                    text=batch.text,
                    source_latent=batch.source_latent,
                    mask=batch.mask,
                    task=batch.task,
                )
                per = (pred - train_tuple.target.to(dtype=pred.dtype)).pow(2).mean(dim=[1, 2, 3])
            losses.append(float(per.mean().cpu()))
            t = train_tuple.t.detach()
            per_f = per.detach().float()
            for idx in range(10):
                lo = idx / 10.0
                hi = (idx + 1) / 10.0
                mask = ((t >= lo) & (t <= hi)) if idx == 9 else ((t >= lo) & (t < hi))
                if mask.any():
                    bins.setdefault(f"val_loss_t_{idx}_{idx + 1}", []).append(float(per_f[mask].mean().cpu()))
    finally:
        model.train(was_training)
    if not losses:
        return None, {}
    return sum(losses) / len(losses), {k: sum(v) / len(v) for k, v in bins.items() if v}


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().data.norm(2).item()) ** 2
    return total ** 0.5


def _build_ckpt(
    *,
    cfg: TrainConfig,
    cfg_dict: dict,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA,
    step: int,
    text_metadata: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "ema": ema.shadow,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": {"lr_scheduler": str(cfg.lr_scheduler)},
        "step": int(step),
        "cfg": cfg_dict,
        "architecture": "mmdit_rf",
        "objective": "rectified_flow",
        "vae": {
            "pretrained": str(cfg.vae_pretrained),
            "scaling_factor": float(cfg.vae_scaling_factor),
        },
        "text_encoders": text_metadata.get("encoders", []),
        "text_dim": int(cfg.text_dim),
        "pooled_dim": int(cfg.pooled_dim),
        "text_max_length_total": int(
            sum(int(item.get("max_length", 0)) for item in text_metadata.get("encoders", []))
        ),
    }


def run_mmdit_training_loop(
    *,
    cfg: TrainConfig,
    cfg_dict: dict,
    model: MMDiTFlowModel,
    dataloader,
    val_dataloader,
    objective: RectifiedFlowObjective,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA,
    device: torch.device,
    out_dir: Path,
    empty_text: TextConditioning,
    start_step: int,
    text_metadata: dict,
    eval_prompts: Optional[list[str]] = None,
    eval_text_encoder: Optional[FrozenTextEncoderBundle] = None,
    eval_vae: Optional[VAEWrapper] = None,
) -> None:
    try:
        dl_len = len(dataloader)
    except TypeError:
        dl_len = None
    if dl_len == 0:
        raise RuntimeError(
            "MMDiT training dataloader is empty. "
            "Check latent cache, text cache, batch_size, drop_last, and dataset filters."
        )

    max_steps = int(cfg.max_steps)
    grad_accum = max(int(cfg.grad_accum_steps), 1)
    log_every = int(cfg.log_every)
    save_every = int(cfg.save_every)
    val_every = int(cfg.val_every)
    val_batches = int(cfg.val_batches)
    base_lr = float(cfg.lr)
    warmup_steps = int(cfg.warmup_steps)
    decay_steps = int(cfg.decay_steps) if int(cfg.decay_steps) > 0 else max(max_steps - warmup_steps, 0)
    min_lr_ratio = float(cfg.min_lr_ratio)
    use_amp = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    grad_clip = float(cfg.grad_clip_norm)

    metrics_dir = out_dir / "metrics"
    sinks = [JsonlFileSink(metrics_dir / "events.jsonl")]
    if _is_webui_mode():
        sinks.append(StdoutJsonSink())
    metrics_path = _webui_metrics_path()
    if metrics_path is not None:
        sinks.append(JsonlFileSink(metrics_path, event_types=["metric"]))
    event_bus = EventBus(sinks)

    pbar = tqdm(
        total=max_steps,
        initial=min(start_step, max_steps),
        desc="mmdit_rf",
        unit="step",
        disable=_is_webui_mode(),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    empty_text = _move_text(empty_text, device)
    step = int(start_step)
    accum_idx = 0
    recent_losses: list[float] = []
    recent_bins: dict[str, list[float]] = {}

    while step < max_steps:
        saw_batch = False
        for raw in dataloader:
            saw_batch = True
            if step >= max_steps:
                break
            lr = _compute_lr(
                step=step,
                base_lr=base_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                min_lr_ratio=min_lr_ratio,
                scheduler=str(cfg.lr_scheduler),
            )
            _apply_lr(optimizer, lr)
            batch = _move_batch(raw, device)
            loss, bins = _loss_and_bins(
                model=model,
                objective=objective,
                batch=batch,
                empty_text=empty_text,
                cfg_drop_prob=float(cfg.cond_drop_prob),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            _assert_finite("mmdit_loss", loss.detach())
            scaler.scale(loss / grad_accum).backward()
            recent_losses.append(float(loss.detach().cpu()))
            for key, value in bins.items():
                recent_bins.setdefault(key, []).append(value)
            accum_idx += 1

            if accum_idx >= grad_accum:
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm = _grad_norm(model)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if bool(cfg.fail_on_nonfinite_grad):
                    for name, param in model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            raise RuntimeError(f"Non-finite gradient in {name}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                accum_idx = 0
                next_step = step + 1

                if log_every > 0 and next_step % log_every == 0:
                    event = {
                        "type": "metric",
                        "step": next_step,
                        "max_steps": max_steps,
                        "loss": sum(recent_losses) / max(len(recent_losses), 1),
                        "lr": lr,
                        "grad_norm": grad_norm,
                    }
                    event.update({k: sum(v) / len(v) for k, v in recent_bins.items() if v})
                    event_bus.emit(event)
                    recent_losses.clear()
                    recent_bins.clear()

                if val_every > 0 and next_step % val_every == 0:
                    val_loss, val_bins = _run_validation(
                        model=model,
                        objective=objective,
                        dataloader=val_dataloader,
                        device=device,
                        max_batches=val_batches,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                    if val_loss is not None:
                        event = {
                            "type": "metric",
                            "step": next_step,
                            "max_steps": max_steps,
                            "val_loss": val_loss,
                        }
                        event.update(val_bins)
                        event_bus.emit(event)

                if (
                    int(cfg.eval_every) > 0
                    and next_step % int(cfg.eval_every) == 0
                    and eval_prompts
                    and eval_text_encoder is not None
                    and eval_vae is not None
                ):
                    run_mmdit_eval_sampling(
                        step=next_step,
                        model=model,
                        ema=ema,
                        vae=eval_vae,
                        text_encoder=eval_text_encoder,
                        out_dir=out_dir,
                        prompts=eval_prompts,
                        eval_seed=int(cfg.eval_seed),
                        eval_sampler=str(cfg.eval_sampler),
                        eval_steps=int(cfg.eval_steps),
                        eval_cfg=float(cfg.eval_cfg),
                        eval_n=int(cfg.eval_n),
                        latent_channels=int(cfg.latent_channels),
                        image_size=int(cfg.image_size),
                        latent_downsample_factor=int(cfg.latent_downsample_factor),
                        shift=float(cfg.sampling_shift),
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                    )
                    event_bus.emit(
                        {
                            "type": "metric",
                            "step": next_step,
                            "max_steps": max_steps,
                            "eval_samples": len(eval_prompts) * int(cfg.eval_n),
                        }
                    )

                if save_every > 0 and next_step % save_every == 0:
                    ckpt = _build_ckpt(
                        cfg=cfg,
                        cfg_dict=cfg_dict,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        ema=ema,
                        step=next_step,
                        text_metadata=text_metadata,
                    )
                    save_ckpt(str(out_dir / f"ckpt_{next_step:07d}.pt"), ckpt)
                    save_ckpt(str(out_dir / "ckpt_latest.pt"), ckpt)
                    _prune_checkpoints(out_dir, int(cfg.ckpt_keep_last))

                step = next_step
                pbar.update(1)
        if not saw_batch:
            raise RuntimeError(
                "MMDiT training dataloader produced no batches. "
                "Check latent cache, text cache, batch_size, drop_last, and dataset filters."
            )

    ckpt = _build_ckpt(
        cfg=cfg,
        cfg_dict=cfg_dict,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        ema=ema,
        step=step,
        text_metadata=text_metadata,
    )
    save_ckpt(str(out_dir / "ckpt_final.pt"), ckpt)
    save_ckpt(str(out_dir / "ckpt_latest.pt"), ckpt)
    pbar.close()
