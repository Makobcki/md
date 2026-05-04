from __future__ import annotations

from pathlib import Path
import time
from typing import Optional

import torch
from tqdm import tqdm

from config.train import TrainConfig
from diffusion.io.events import AsyncEventBus, JsonlFileSink
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import EMA, unwrap_model
from model.mmdit import MMDiTFlowModel
from model.text.conditioning import TextConditioning, TrainBatch
from model.text.pretrained import FrozenTextEncoderBundle
from diffusion.vae import VAEWrapper
from train.checkpoint import _prune_checkpoints, save_ckpt
from train.eval_mmdit import run_mmdit_eval_sampling
from train.schedulers import _apply_lr, _compute_lr
from train.webui import _is_webui_mode, _webui_metrics_path
from train.metrics import loss_by_t_bins
from train.checkpoint_mmdit import build_mmdit_checkpoint_metadata
from train.dist import DistributedContext


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf values")


def _move_text(text: TextConditioning, device: torch.device) -> TextConditioning:
    return TextConditioning(
        tokens=text.tokens.to(device),
        mask=text.mask.to(device),
        pooled=text.pooled.to(device),
        is_uncond=text.is_uncond.to(device) if text.is_uncond is not None else None,
        token_types=text.token_types.to(device) if text.token_types is not None else None,
    )


def _move_batch(batch: TrainBatch, device: torch.device) -> TrainBatch:
    return TrainBatch(
        x0=batch.x0.to(device=device, dtype=torch.float32),
        text=_move_text(batch.text, device),
        source_latent=batch.source_latent.to(device=device, dtype=torch.float32)
        if batch.source_latent is not None
        else None,
        mask=batch.mask.to(device=device, dtype=torch.float32) if batch.mask is not None else None,
        control_latents=batch.control_latents.to(device=device, dtype=torch.float32)
        if batch.control_latents is not None
        else None,
        control_type=batch.control_type.to(device=device, dtype=torch.long) if batch.control_type is not None else None,
        task=batch.task,
        strength=batch.strength.to(device=device, dtype=torch.float32) if batch.strength is not None else None,
        control_strength=batch.control_strength.to(device=device, dtype=torch.float32) if batch.control_strength is not None else None,
        metadata=batch.metadata,
    )


def _replace_text_where(text: TextConditioning, empty_text: TextConditioning, drop_prob: float) -> TextConditioning:
    if drop_prob <= 0:
        return text
    drop = torch.rand(text.tokens.shape[0], device=text.tokens.device) < float(drop_prob)
    return text.replace_where(drop, empty_text)


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _per_sample_flow_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    mask_weight: float = 1.0,
    unmask_weight: float = 1.0,
) -> torch.Tensor:
    err = (pred.float() - target.to(device=pred.device, dtype=torch.float32)).pow(2)
    full = err.mean(dim=[1, 2, 3])
    if mask is None:
        return full
    m = mask.to(device=pred.device, dtype=torch.float32)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    if m.shape[-2:] != pred.shape[-2:]:
        raise RuntimeError(f"mask shape {tuple(m.shape)} is incompatible with prediction shape {tuple(pred.shape)}")
    if m.shape[0] != pred.shape[0]:
        raise RuntimeError(f"mask batch {m.shape[0]} is incompatible with prediction batch {pred.shape[0]}")
    if m.shape[1] not in {1, pred.shape[1]}:
        raise RuntimeError(f"mask channel count {m.shape[1]} is incompatible with prediction channels {pred.shape[1]}")
    weights = m * float(mask_weight) + (1.0 - m) * float(unmask_weight)
    weights = weights.expand_as(err) if weights.shape[1] == 1 else weights
    denom = weights.sum(dim=[1, 2, 3])
    weighted = (err * weights).sum(dim=[1, 2, 3]) / denom.clamp_min(1.0)
    return torch.where(denom > 0, weighted, full)


def _loss_and_bins(
    *,
    model: MMDiTFlowModel,
    objective: RectifiedFlowObjective,
    batch: TrainBatch,
    empty_text: TextConditioning,
    cfg_drop_prob: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    inpaint_loss_mask_weight: float = 1.0,
    inpaint_loss_unmask_weight: float = 1.0,
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
            control_latents=batch.control_latents,
            control_type=batch.control_type,
            task=batch.task,
            strength=batch.strength,
            control_strength=batch.control_strength,
        )
        _assert_finite("mmdit_pred", pred)
        per = _per_sample_flow_mse(
            pred,
            train_tuple.target,
            batch.mask,
            mask_weight=float(inpaint_loss_mask_weight),
            unmask_weight=float(inpaint_loss_unmask_weight),
        )
        loss = (per * train_tuple.weight.to(device=per.device, dtype=per.dtype)).mean()
        if float(getattr(model.cfg, "x0_aux_weight", 0.0)) > 0:
            t_view = train_tuple.t.to(device=pred.device, dtype=pred.dtype).view(-1, 1, 1, 1)
            x0_pred = train_tuple.xt.to(device=pred.device, dtype=pred.dtype) - t_view * pred
            x0_err = (x0_pred.float() - batch.x0.to(device=pred.device, dtype=torch.float32)).pow(2).mean(dim=[1, 2, 3])
            loss = loss + float(model.cfg.x0_aux_weight) * (x0_err * train_tuple.weight.to(device=x0_err.device, dtype=x0_err.dtype)).mean()

    stats = loss_by_t_bins(per.detach(), train_tuple.t.detach(), bins=10, prefix="loss_t_bin")
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
    inpaint_loss_mask_weight: float = 1.0,
    inpaint_loss_unmask_weight: float = 1.0,
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
                    control_latents=batch.control_latents,
                    control_type=batch.control_type,
                    task=batch.task,
                    strength=batch.strength,
                    control_strength=batch.control_strength,
                )
                per = _per_sample_flow_mse(
                    pred,
                    train_tuple.target,
                    batch.mask,
                    mask_weight=float(inpaint_loss_mask_weight),
                    unmask_weight=float(inpaint_loss_unmask_weight),
                )
            losses.append(float(per.mean().cpu()))
            for key, value in loss_by_t_bins(per.detach(), train_tuple.t.detach(), bins=10, prefix="val_loss_t_bin").items():
                bins.setdefault(key, []).append(value)
    finally:
        model.train(was_training)
    if not losses:
        return None, {}
    return sum(losses) / len(losses), {k: sum(v) / len(v) for k, v in bins.items() if v}


def _grad_diagnostics(model: torch.nn.Module) -> dict[str, float | bool]:
    total = 0.0
    has_nan = False
    has_inf = False
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        has_nan = has_nan or bool(torch.isnan(grad).any().item())
        has_inf = has_inf or bool(torch.isinf(grad).any().item())
        finite_grad = torch.nan_to_num(grad.float(), nan=0.0, posinf=0.0, neginf=0.0)
        total += float(finite_grad.norm(2).item()) ** 2
    norm = total ** 0.5
    return {
        "grad_norm_total": norm,
        "grad_norm": norm,
        "has_nan_grad": has_nan,
        "has_inf_grad": has_inf,
    }


def _grad_norm(model: torch.nn.Module) -> float:
    return float(_grad_diagnostics(model)["grad_norm_total"])


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
    metadata = build_mmdit_checkpoint_metadata(
        cfg=cfg,
        cfg_dict=cfg_dict,
        step=step,
        text_metadata=text_metadata,
        dataset_hash=str(cfg_dict.get("dataset_hash", "")),
    )
    return {
        "model": unwrap_model(model).state_dict(),
        "ema": ema.shadow,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": {"lr_scheduler": str(cfg.lr_scheduler)},
        "step": int(step),
        "cfg": cfg_dict,
        "metadata": metadata,
        "architecture": metadata["architecture"],
        "objective": metadata["objective"],
        "prediction_type": metadata["prediction_type"],
        "latent_channels": int(cfg.latent_channels),
        "latent_patch_size": int(cfg.latent_patch_size),
        "hidden_dim": int(cfg.hidden_dim),
        "depth": int(cfg.depth),
        "num_heads": int(cfg.num_heads),
        "double_stream_blocks": int(cfg.double_stream_blocks),
        "single_stream_blocks": int(cfg.single_stream_blocks),
        "pos_embed": str(cfg.pos_embed),
        "vae": metadata["vae_config"],
        "text_encoders": metadata["text_config"].get("encoders", []),
        "text_dim": int(cfg.text_dim),
        "pooled_dim": int(cfg.pooled_dim),
        "text_max_length_total": int(metadata["text_config"].get("max_length_total", 0)),
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
    checkpoint_dir: Path | None = None,
    start_step: int,
    text_metadata: dict,
    eval_prompts: Optional[list[str]] = None,
    eval_text_encoder: Optional[FrozenTextEncoderBundle] = None,
    eval_vae: Optional[VAEWrapper] = None,
    dist: DistributedContext | None = None,
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
    dist = dist or DistributedContext(device=device)
    write_outputs = dist.should_write()

    metrics_dir = out_dir / "metrics"
    checkpoint_dir = checkpoint_dir or out_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sinks = []
    if write_outputs:
        sinks.extend([JsonlFileSink(out_dir / "events.jsonl"), JsonlFileSink(metrics_dir / "events.jsonl")])
        metrics_path = _webui_metrics_path()
        if metrics_path is not None:
            sinks.append(JsonlFileSink(metrics_path, event_types=["metric", "progress", "train", "eval", "sample"]))
    event_bus = AsyncEventBus(sinks)

    pbar = tqdm(
        total=max_steps,
        initial=min(start_step, max_steps),
        desc="mmdit_rf",
        unit="step",
        disable=_is_webui_mode() or not write_outputs,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    empty_text = _move_text(empty_text, device)
    step = int(start_step)
    accum_idx = 0
    recent_losses: list[float] = []
    recent_bins: dict[str, list[float]] = {}
    recent_samples = 0
    run_start = time.perf_counter()
    log_window_start = run_start
    log_window_step = int(step)

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
                inpaint_loss_mask_weight=float(cfg.inpaint_loss_mask_weight),
                inpaint_loss_unmask_weight=float(cfg.inpaint_loss_unmask_weight),
            )
            _assert_finite("mmdit_loss", loss.detach())
            scaler.scale(loss / grad_accum).backward()
            recent_losses.append(float(loss.detach().cpu()))
            recent_samples += int(batch.x0.shape[0])
            for key, value in bins.items():
                recent_bins.setdefault(key, []).append(value)
            accum_idx += 1

            if accum_idx >= grad_accum:
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_stats = _grad_diagnostics(model)
                grad_norm = float(grad_stats["grad_norm_total"])
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if bool(cfg.fail_on_nonfinite_grad):
                    if bool(grad_stats["has_nan_grad"]) or bool(grad_stats["has_inf_grad"]):
                        for name, param in model.named_parameters():
                            if param.grad is not None and not torch.isfinite(param.grad).all():
                                raise RuntimeError(f"Non-finite gradient in {name}")
                        raise RuntimeError("Non-finite gradient detected")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                accum_idx = 0
                next_step = step + 1
                now = time.perf_counter()
                elapsed_sec = max(now - run_start, 0.0)
                completed_steps = max(int(next_step) - int(start_step), 1)
                avg_sec_per_step_progress = elapsed_sec / float(completed_steps)
                remaining_steps_progress = max(int(max_steps) - int(next_step), 0)
                eta_sec_progress = remaining_steps_progress * avg_sec_per_step_progress
                event_bus.emit(
                    {
                        "type": "progress",
                        "step": next_step,
                        "max_steps": max_steps,
                        "lr": lr,
                        "elapsed_sec": float(elapsed_sec),
                        "elapsed": _format_duration(elapsed_sec),
                        "eta_sec": float(eta_sec_progress),
                        "eta": _format_duration(eta_sec_progress),
                        "eta_h": float(eta_sec_progress) / 3600.0,
                        "sec_per_step": float(avg_sec_per_step_progress),
                        "s_per_step": float(avg_sec_per_step_progress),
                        "avg_sec_per_step": float(avg_sec_per_step_progress),
                        "steps_per_sec": float(1.0 / avg_sec_per_step_progress) if avg_sec_per_step_progress > 0 else 0.0,
                        "remaining_steps": int(remaining_steps_progress),
                    }
                )

                if log_every > 0 and next_step % log_every == 0:
                    window_elapsed_sec = max(now - log_window_start, 1.0e-9)
                    window_steps = max(int(next_step) - int(log_window_step), 1)
                    sec_per_step = window_elapsed_sec / float(window_steps)
                    avg_sec_per_step = elapsed_sec / float(max(int(next_step) - int(start_step), 1))
                    remaining_steps = max(int(max_steps) - int(next_step), 0)
                    eta_sec = remaining_steps * sec_per_step
                    mean_loss = sum(recent_losses) / max(len(recent_losses), 1)
                    mean_loss = dist.reduce_mean_float(mean_loss, device=device)
                    grad_norm = dist.reduce_mean_float(grad_norm, device=device)
                    event = {
                        "type": "train",
                        "step": next_step,
                        "max_steps": max_steps,
                        "loss": mean_loss,
                        "train_loss": mean_loss,
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "grad_norm_total": float(grad_stats["grad_norm_total"]),
                        "has_nan_grad": bool(grad_stats["has_nan_grad"]),
                        "has_inf_grad": bool(grad_stats["has_inf_grad"]),
                        "samples_per_sec": float(recent_samples) / window_elapsed_sec,
                        "steps_per_sec": float(window_steps) / window_elapsed_sec,
                        "sec_per_step": float(sec_per_step),
                        "s_per_step": float(sec_per_step),
                        "avg_sec_per_step": float(avg_sec_per_step),
                        "elapsed_sec": float(elapsed_sec),
                        "elapsed": _format_duration(elapsed_sec),
                        "eta_sec": float(eta_sec),
                        "eta": _format_duration(eta_sec),
                        "eta_h": float(eta_sec) / 3600.0,
                        "remaining_steps": int(remaining_steps),
                    }
                    local_bins = {k: sum(v) / len(v) for k, v in recent_bins.items() if v}
                    event.update({k: dist.reduce_mean_float(float(v), device=device) for k, v in local_bins.items()})
                    event_bus.emit(event)
                    recent_losses.clear()
                    recent_bins.clear()
                    recent_samples = 0
                    log_window_start = time.perf_counter()
                    log_window_step = int(next_step)

                if val_every > 0 and next_step % val_every == 0:
                    val_loss, val_bins = _run_validation(
                        model=model,
                        objective=objective,
                        dataloader=val_dataloader,
                        device=device,
                        max_batches=val_batches,
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        inpaint_loss_mask_weight=float(cfg.inpaint_loss_mask_weight),
                        inpaint_loss_unmask_weight=float(cfg.inpaint_loss_unmask_weight),
                    )
                    if val_loss is not None:
                        val_loss = dist.reduce_mean_float(float(val_loss), device=device)
                        event = {
                            "type": "eval",
                            "step": next_step,
                            "max_steps": max_steps,
                            "val_loss": val_loss,
                        }
                        event.update({k: dist.reduce_mean_float(float(v), device=device) for k, v in val_bins.items()})
                        event_bus.emit(event)

                if (
                    int(cfg.eval_every) > 0
                    and next_step % int(cfg.eval_every) == 0
                    and write_outputs
                    and eval_prompts
                    and eval_text_encoder is not None
                    and eval_vae is not None
                ):
                    sample_events = run_mmdit_eval_sampling(
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
                    for sample_event in sample_events:
                        event = dict(sample_event)
                        event["max_steps"] = max_steps
                        event_bus.emit(event)

                if save_every > 0 and next_step % save_every == 0 and write_outputs:
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
                    save_ckpt(str(checkpoint_dir / f"step_{next_step:06d}.pt"), ckpt)
                    save_ckpt(str(checkpoint_dir / "latest.pt"), ckpt)
                    # Backward-compatible flat names for older scripts/UI.
                    save_ckpt(str(out_dir / f"ckpt_{next_step:07d}.pt"), ckpt)
                    save_ckpt(str(out_dir / "ckpt_latest.pt"), ckpt)
                    _prune_checkpoints(out_dir, int(cfg.ckpt_keep_last))
                    _prune_checkpoints(checkpoint_dir, int(cfg.ckpt_keep_last))

                step = next_step
                pbar.update(1)
        if not saw_batch:
            raise RuntimeError(
                "MMDiT training dataloader produced no batches. "
                "Check latent cache, text cache, batch_size, drop_last, and dataset filters."
            )

    if write_outputs:
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
        save_ckpt(str(checkpoint_dir / "final.pt"), ckpt)
        save_ckpt(str(checkpoint_dir / "latest.pt"), ckpt)
        save_ckpt(str(out_dir / "ckpt_final.pt"), ckpt)
        save_ckpt(str(out_dir / "ckpt_latest.pt"), ckpt)
    dist.wait_for_everyone()
    event_bus.close()
    pbar.close()
