from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusion.core.diffusion import Diffusion
from diffusion.domains.domain import Batch
from diffusion.domains.latent import LatentDomain
from diffusion.domains.pixel import PixelDomain
from diffusion.io.events import EventBus, JsonlFileSink, StdoutJsonSink
from diffusion.losses.snr import get_min_snr_weights
from diffusion.utils import EMA
from diffusion.vae import VAEWrapper
from model.unet.unet import UNet
from text_enc.tokenizer import BPETokenizer

from data_loader import ImageTextDataset
from train.checkpoint import _prune_checkpoints, save_ckpt
from train.dist import _dist_all_reduce_sum, _dist_is_initialized, _dist_rank
from train.eval import _run_eval_sampling
from train.schedulers import _apply_lr, _compute_lr
from train.timing import _TimingStats
from train.webui import _is_webui_mode, _webui_metrics_path


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf values")


def _find_bad_grads(model: torch.nn.Module) -> list[str]:
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(name)
    return bad


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().data.norm(2)
        total += float(param_norm.item()) ** 2
    return total ** 0.5


def _configure_cudagraphs(enabled: bool) -> None:
    if not hasattr(torch, "_inductor"):
        return
    cfg = getattr(torch._inductor, "config", None)
    if cfg is None:
        return
    triton_cfg = getattr(cfg, "triton", None)
    if triton_cfg is not None and hasattr(triton_cfg, "cudagraphs"):
        triton_cfg.cudagraphs = bool(enabled)
    if hasattr(cfg, "cudagraphs"):
        cfg.cudagraphs = bool(enabled)


def _cudagraph_step_begin() -> None:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def run_training_loop(
    *,
    run_cfg,
    cfg_dict: dict,
    run_meta: dict,
    out_dir: Path,
    device: torch.device,
    perf_active: dict,
    use_text_conditioning: bool,
    self_conditioning: bool,
    self_cond_prob: float,
    effective_cond_drop_prob: float,
    tokenizer: Optional[BPETokenizer],
    ds: ImageTextDataset,
    dl_full: DataLoader,
    dl_curr: Optional[DataLoader],
    diff: Diffusion,
    domain: PixelDomain | LatentDomain,
    model: UNet,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA,
    start_step: int,
    eval_prompts: Optional[list[str]],
    eval_sampler: str,
    eval_steps: int,
    eval_cfg: float,
    eval_seed: int,
    eval_n: int,
    eval_vae: Optional[VAEWrapper],
    compile_cudagraphs: bool,
    amp_dtype: torch.dtype,
) -> None:
    max_steps = int(run_cfg.max_steps)
    grad_accum = int(run_cfg.grad_accum_steps)
    log_every = int(run_cfg.log_every)
    save_every = int(run_cfg.save_every)
    min_snr_gamma = float(run_cfg.min_snr_gamma)
    grad_clip = float(run_cfg.grad_clip_norm)
    base_lr = float(run_cfg.lr)
    warmup_steps = int(run_cfg.warmup_steps)
    decay_steps = int(run_cfg.decay_steps) if int(run_cfg.decay_steps) > 0 else max(max_steps - warmup_steps, 0)
    lr_scheduler = str(run_cfg.lr_scheduler)
    min_lr_ratio = float(run_cfg.min_lr_ratio)
    compile_warmup_steps = int(run_cfg.compile_warmup_steps) if bool(run_cfg.compile) else 0
    warmup_end_step = start_step + max(compile_warmup_steps, 0)
    if grad_accum > 1:
        compile_cudagraphs = False
    _configure_cudagraphs(compile_cudagraphs)

    webui_mode = _is_webui_mode()
    metrics_path = _webui_metrics_path()

    is_main = _dist_rank() == 0

    pbar = tqdm(
        total=int(run_cfg.max_steps),
        initial=start_step,
        desc="train",
        unit="step",
        disable=webui_mode or not is_main,  # важное: webui -> без tqdm, иначе мусор в stdout
    )

    metrics_dir = Path(run_cfg.out_dir) / "metrics"
    events_path = metrics_dir / "events.jsonl"
    sinks = [JsonlFileSink(events_path)]
    if webui_mode:
        sinks.append(StdoutJsonSink())
    if metrics_path is not None:
        sinks.append(JsonlFileSink(metrics_path, event_types=["metric"]))
    event_bus = EventBus(sinks)
    timing = _TimingStats(
        use_cuda_events=(device.type == "cuda"),
        gpu_sections={"fwd_bwd", "opt_step"},
    )

    def _log(message: str, step: Optional[int] = None) -> None:
        if webui_mode and is_main:
            event_bus.emit({
                "type": "log",
                "message": message,
                "step": int(step) if step is not None else start_step,
            })
        elif is_main:
            print(message)

    if is_main:
        mode_label = "ENABLED" if use_text_conditioning else "DISABLED (unconditional diffusion)"
        _log(f"[INFO] Text conditioning: {mode_label}", step=start_step)

    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_step = start_step

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stop_requested = {"value": False}

    def _request_stop(signum, _frame) -> None:
        stop_requested["value"] = True
        _log(f"[SIGNAL] stop requested via {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    dl = dl_curr if dl_curr is not None else dl_full
    if is_main and not use_text_conditioning and bool(run_cfg.curriculum_enabled):
        _log("[WARN] curriculum disabled because use_text_conditioning=false", step=start_step)
    it = iter(dl)
    start_time = time.perf_counter()

    initial_lr = _compute_lr(
        step=start_step,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr_ratio=min_lr_ratio,
        scheduler=lr_scheduler,
    )
    _apply_lr(opt, initial_lr)

    if is_main:
        if domain.name == "latent":
            cache_hit_rate = ds.latent_cache_hit_rate()
            if cache_hit_rate is None:
                _log("[WARN] latent cache hit rate unavailable")
            else:
                _log(f"[INFO] latent cache hit rate={cache_hit_rate:.2%} missing={ds.latent_cache_missing}")
                if cache_hit_rate == 0.0:
                    _log("[WARN] latent cache appears empty; did you run prepare_latents.py?")
        event_bus.emit({
            "type": "log",
            "message": (
                "runtime flags: "
                f"compile={bool(run_cfg.compile)}, "
                f"compile_cudagraphs={compile_cudagraphs}, "
                f"tf32={perf_active['tf32']}, "
                f"channels_last={perf_active['channels_last']}, "
                f"sdp_flash={perf_active['sdp_flash']}, "
                f"sdp_mem_efficient={perf_active['sdp_mem_efficient']}, "
                f"sdp_math={perf_active['sdp_math']}"
            ),
            "step": start_step,
        })
        event_bus.emit({
            "type": "status",
            "status": "start",
            "step": start_step,
            "resume": bool(run_cfg.resume_ckpt),
            "out_dir": str(out_dir),
        })

    log_loss_sum = 0.0
    log_loss_count = 0
    best_loss: Optional[float] = None
    best_step: Optional[int] = None

    for step in range(start_step, max_steps):
        if bool(run_cfg.curriculum_enabled) and dl_curr is not None and step == int(run_cfg.curriculum_steps):
            dl = dl_full
            it = iter(dl)
            if is_main:
                _log(f"[INFO] curriculum finished at step={step}", step=step)
        step_lr = _compute_lr(
            step=step,
            base_lr=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_ratio=min_lr_ratio,
            scheduler=lr_scheduler,
        )
        _apply_lr(opt, step_lr)
        if bool(run_cfg.compile) and compile_cudagraphs:
            _cudagraph_step_begin()
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        data_time = 0.0
        fwd_bwd_time = 0.0
        last_batch_stats = {"x_std": None, "v_std": None}

        capture_batch_stats = step % log_every == 0 and step >= warmup_end_step
        for accum_idx in range(grad_accum):
            with timing.section("data_fetch") as t_data:
                try:
                    x0, txt_ids, txt_mask = next(it)
                except StopIteration:
                    it = iter(dl)
                    x0, txt_ids, txt_mask = next(it)
            data_time += t_data.elapsed_sec

            batch = Batch(x=x0, txt_ids=txt_ids, txt_mask=txt_mask, domain=domain.name)
            prepared = domain.prepare_batch(batch)
            x0 = prepared.x
            txt_ids = prepared.txt_ids
            txt_mask = prepared.txt_mask

            b = x0.shape[0]
            t = torch.randint(0, diff.cfg.timesteps, (b,), device=x0.device, dtype=torch.long)
            noise = domain.sample_noise_like(x0)
            alpha_bar_t = diff.alpha_bar[t]
            _assert_finite("alpha_bar[t]", alpha_bar_t)
            xt = domain.q_sample(x0, t, noise)
            v_tgt = domain.v_target(x0, t, noise)
            if capture_batch_stats and accum_idx == grad_accum - 1:
                last_batch_stats["x_std"] = float(x0.detach().float().std().item())
                last_batch_stats["v_std"] = float(v_tgt.detach().float().std().item())

            with timing.section("fwd_bwd") as t_fwd:
                with torch.amp.autocast("cuda", enabled=bool(run_cfg.amp) and device.type == "cuda", dtype=amp_dtype):
                    self_cond = None
                    if self_conditioning and self_cond_prob > 0:
                        if torch.rand((), device=xt.device).item() < self_cond_prob:
                            with torch.no_grad():
                                v_sc = model(xt, t, txt_ids, txt_mask, None)
                                self_cond = diff.v_to_x0(xt, t, v_sc).detach()
                    v_pred = model(xt, t, txt_ids, txt_mask, self_cond)
                    _assert_finite("v_pred", v_pred)
                    if v_pred.shape != v_tgt.shape:
                        raise RuntimeError("v_pred/v_target shape mismatch")
                    per = F.mse_loss(
                        v_pred,
                        v_tgt.to(dtype=v_pred.dtype),
                        reduction="none",
                    ).mean(dim=[1, 2, 3])  # [B]
                    w = get_min_snr_weights(diff.alpha_bar[t], gamma=min_snr_gamma)        # [B]
                    loss = (per * w).mean() / grad_accum

            if not torch.isfinite(loss):
                dump_path = out_dir / f"nan_dump_{step:07d}.pt"
                save_ckpt(str(dump_path), {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": cfg_dict,
                    "meta": run_meta,
                    "batch_stats": {
                        "x0_min": float(x0.min().item()),
                        "x0_max": float(x0.max().item()),
                        "alpha_bar_min": float(alpha_bar_t.min().item()),
                        "alpha_bar_max": float(alpha_bar_t.max().item()),
                    },
                })
                raise RuntimeError(f"Non-finite loss at step={step}: {loss.item()}")

            total_loss += float(loss.detach().cpu())
            scaler.scale(loss).backward()
            fwd_bwd_time += t_fwd.elapsed_sec

        opt_step_start = time.perf_counter()
        if scaler.is_enabled():
            scaler.unscale_(opt)
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        else:
            grad_norm = _grad_norm(model)

        with timing.section("opt_step") as t_opt:
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        ema.update(model)

        bad_grads = _find_bad_grads(model)
        if bad_grads:
            dump_path = out_dir / f"nan_dump_{step:07d}.pt"
            save_ckpt(str(dump_path), {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.shadow,
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg_dict,
                "meta": run_meta,
                "batch_stats": {
                    "bad_grads": bad_grads[:10],
                },
            })
            raise RuntimeError(f"Non-finite grads at step={step}: {bad_grads[:5]}")

        opt_time = time.perf_counter() - opt_step_start
        step_time = time.perf_counter() - step_start
        timing.add_cpu("step_total", step_time)

        log_loss_sum += float(total_loss)
        log_loss_count += 1

        if step % log_every == 0 and step >= warmup_end_step:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.perf_counter()
            elapsed = now - last_log_time
            steps_done = max(step - last_log_step, 1)

            # сколько "картинок" реально прошло за этот лог-интервал
            images = steps_done * int(run_cfg.batch_size) * int(run_cfg.grad_accum_steps)
            img_per_sec = images / max(elapsed, 1e-9)

            peak_mem = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            )

            total_elapsed = now - start_time
            steps_left = int(run_cfg.max_steps) - step - 1
            sec_per_step = elapsed / steps_done
            eta_h = (steps_left * sec_per_step) / 3600.0

            loss_sum = log_loss_sum
            loss_count = log_loss_count
            if _dist_is_initialized():
                stats = torch.tensor([loss_sum, loss_count], device=device, dtype=torch.float64)
                stats = _dist_all_reduce_sum(stats)
                loss_sum = float(stats[0].item())
                loss_count = int(stats[1].item())
            loss_mean = loss_sum / max(loss_count, 1)

            # CLI-UI (только если НЕ webui)
            if not webui_mode and is_main:
                pbar.set_postfix({
                    "loss": f"{loss_mean:.6f}",
                    "img/s": f"{img_per_sec:.2f}",
                    "mem(MB)": f"{peak_mem:.0f}",
                    "eta(h)": f"{eta_h:.2f}",
                })

            if is_main:
                timing_stats = timing.report(reset=True)
                payload = {
                    "type": "metric",
                    "step": step,
                    "loss": float(loss_mean),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad_norm": float(grad_norm),
                    "x_std": last_batch_stats["x_std"],
                    "v_std": last_batch_stats["v_std"],
                    "ema_decay": float(ema.decay),
                    "cfg_drop_prob": float(effective_cond_drop_prob),
                    "img_per_sec": float(img_per_sec),
                    "peak_mem_mb": float(peak_mem),
                    "elapsed_sec": float(total_elapsed),
                    "eta_h": float(eta_h),
                    "sec_per_step": float(sec_per_step),
                    "data_time_sec": float(data_time),
                    "forward_backward_time_sec": float(fwd_bwd_time),
                    "optimizer_step_time_sec": float(opt_time),
                    "total_step_time_sec": float(step_time),
                    "timing": timing_stats,
                    "max_steps": int(run_cfg.max_steps),
                    "latent_cache_hit_rate": float(ds.latent_cache_hit_rate() or 0.0) if domain.name == "latent" else None,
                    "domain": domain.name,
                }
                event_bus.emit(payload)
                if best_loss is None or loss_mean < best_loss:
                    best_loss = float(loss_mean)
                    best_step = int(step)
                    best_path = out_dir / "ckpt_best.pt"
                    save_ckpt(str(best_path), {
                        "step": step,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "ema": ema.shadow,
                        "cfg": cfg_dict,
                        "meta": run_meta,
                    })
                    event_bus.emit({
                        "type": "log",
                        "message": f"saved best checkpoint {best_path}",
                        "step": step,
                        "best_loss": best_loss,
                    })

            last_log_time = now
            last_log_step = step
            log_loss_sum = 0.0
            log_loss_count = 0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        elif step + 1 == warmup_end_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            last_log_time = time.perf_counter()
            last_log_step = step
            log_loss_sum = 0.0
            log_loss_count = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if (
            eval_prompts is not None
            and int(run_cfg.eval_every) > 0
            and step % int(run_cfg.eval_every) == 0
            and is_main
        ):
            _log(f"[INFO] eval sampling at step={step}", step=step)
            _run_eval_sampling(
                step=step,
                model=model,
                ema=ema,
                diffusion=diff,
                tokenizer=tokenizer,
                out_dir=out_dir,
                prompts=eval_prompts,
                eval_seed=eval_seed,
                eval_sampler=eval_sampler,
                eval_steps=eval_steps,
                eval_cfg=eval_cfg,
                eval_n=eval_n,
                mode=domain.name,
                image_size=int(run_cfg.image_size),
                latent_channels=int(run_cfg.latent_channels),
                latent_downsample_factor=int(run_cfg.latent_downsample_factor),
                vae=eval_vae,
                use_text_conditioning=use_text_conditioning,
                self_conditioning=self_conditioning,
                use_amp=bool(run_cfg.amp) and device.type == "cuda",
                amp_dtype=amp_dtype,
            )

        if stop_requested["value"]:
            stop_path = out_dir / f"ckpt_stop_{step:07d}.pt"
            stop_payload = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg_dict,
                "meta": run_meta,
            }
            save_ckpt(str(stop_path), stop_payload)
            latest_path = out_dir / "ckpt_latest.pt"
            save_ckpt(str(latest_path), stop_payload)
            if is_main:
                event_bus.emit({
                    "type": "status",
                    "status": "stopped",
                    "step": step,
                    "ckpt": str(stop_path),
                })
            _log(f"[STOP] saved {stop_path}", step=step)
            return

        if step % save_every == 0 and step > 0:
            ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
            ckpt_payload = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg_dict,
                "meta": run_meta,
            }
            save_ckpt(str(ckpt_path), ckpt_payload)
            latest_path = out_dir / "ckpt_latest.pt"
            save_ckpt(str(latest_path), ckpt_payload)
            _prune_checkpoints(out_dir, int(run_cfg.ckpt_keep_last))
            if is_main:
                event_bus.emit({
                    "type": "log",
                    "message": f"saved {ckpt_path}",
                    "step": step,
                })
            _log(f"[OK] saved {ckpt_path}", step=step)

    final_path = out_dir / "ckpt_final.pt"
    final_payload = {
        "step": max_steps - 1,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "cfg": cfg_dict,
        "meta": run_meta,
    }
    save_ckpt(str(final_path), final_payload)
    latest_path = out_dir / "ckpt_latest.pt"
    save_ckpt(str(latest_path), final_payload)
    _prune_checkpoints(out_dir, int(run_cfg.ckpt_keep_last))
    if is_main:
        event_bus.emit({
            "type": "status",
            "status": "done",
            "step": max_steps - 1,
            "ckpt": str(final_path),
        })
    _log(f"[DONE] saved {final_path}", step=max_steps - 1)
