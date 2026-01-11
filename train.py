from __future__ import annotations

import argparse
import json
import os
import signal
import time
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from ddpm.config import TrainConfig
from ddpm.data import DanbooruConfig, DanbooruDataset, build_or_load_index, collate_with_tokenizer
from ddpm.diffusion import Diffusion, DiffusionConfig
from ddpm.model import UNet, UNetConfig
from ddpm.text import BPETokenizer, TextConfig
from ddpm.utils import EMA, build_run_metadata, load_ckpt, save_ckpt, seed_everything

def _is_webui_mode() -> bool:
    return os.environ.get("WEBUI") == "1"


def _webui_metrics_path() -> Path | None:
    run_dir = os.environ.get("WEBUI_RUN_DIR")
    if not run_dir:
        return None
    return Path(run_dir) / "metrics" / "train_metrics.jsonl"


def _emit_event_line(
    *,
    payload: dict,
    log_path: Path,
    metrics_path: Path | None,
    webui_mode: bool,
) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    # всегда пишем локальный jsonl
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # webui: печатаем в stdout + (опционально) отдельный metrics файл
    if webui_mode:
        print(line, flush=True)
        if metrics_path and payload.get("type") == "metric":
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")



def get_min_snr_weights(alpha_bar_t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    eps = 1e-8
    a = alpha_bar_t.clamp(min=eps, max=1.0 - eps)
    snr = a / (1.0 - a + eps)
    g = torch.full_like(snr, float(gamma))
    return torch.minimum(snr, g) / (snr + 1.0)


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf values")


def _assert_in_range(name: str, x: torch.Tensor, lo: float, hi: float, eps: float = 1e-3) -> None:
    if x.min().item() < lo - eps or x.max().item() > hi + eps:
        raise RuntimeError(f"{name} is out of range [{lo}, {hi}]")


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


def _sanity_overfit(
    *,
    model: UNet,
    tokenizer: BPETokenizer,
    entries: list[dict],
    diff: Diffusion,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    steps: int,
    max_images: int,
    max_loss: float,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA,
) -> None:
    if steps <= 0 or max_images <= 0:
        return

    if not entries:
        print("[SANITY] skip overfit: no training entries found")
        return

    max_images = min(max_images, len(entries))
    sanity_entries = entries[:max_images]
    sanity_ds = DanbooruDataset(
        entries=sanity_entries,
        tokenizer=tokenizer,
        cond_drop_prob=0.0,
        seed=0,
    )
    batch = [sanity_ds[i] for i in range(max_images)]
    x0, txt_ids, txt_mask = collate_with_tokenizer(batch)
    x0 = x0.to(device).to(memory_format=torch.channels_last)
    txt_ids = txt_ids.to(device)
    txt_mask = txt_mask.to(device)

    _assert_in_range("sanity x0", x0, -1.0, 1.0)

    backup = {
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": {k: v.detach().clone() for k, v in ema.shadow.items()},
    }

    model.train()
    last_loss = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        t = torch.randint(0, diff.cfg.timesteps, (x0.shape[0],), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)

        alpha_bar_t = diff.alpha_bar[t]
        _assert_finite("alpha_bar[t]", alpha_bar_t)

        xt = diff.q_sample(x0, t, noise)
        v_tgt = diff.v_target(x0, t, noise)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            v_pred = model(xt, t, txt_ids, txt_mask)
            if v_pred.shape != v_tgt.shape:
                raise RuntimeError("v_pred/v_target shape mismatch in sanity overfit")
            loss = F.mse_loss(v_pred, v_tgt.to(dtype=v_pred.dtype))

        scaler.scale(loss).backward()
        bad_grads = _find_bad_grads(model)
        if bad_grads:
            raise RuntimeError(f"Sanity overfit grads contain NaN/Inf: {bad_grads[:5]}")
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        last_loss = float(loss.detach().cpu())

        if step % max(steps // 5, 1) == 0:
            print(f"[SANITY] overfit step {step}/{steps} loss={last_loss:.6f}")
        if last_loss <= max_loss:
            break

    if last_loss is None or last_loss > max_loss:
        raise RuntimeError(f"Sanity overfit loss did not reach target: {last_loss} > {max_loss}")

    model.load_state_dict(backup["model"], strict=True)
    opt.load_state_dict(backup["opt"])
    scaler.load_state_dict(backup["scaler"])
    ema.shadow = backup["ema"]
    print(f"[SANITY] overfit OK (loss={last_loss:.6f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./train.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if args.resume:
        cfg = replace(cfg, resume_ckpt=str(args.resume))
    run_cfg = cfg

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    out_dir = Path(run_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    seed_everything(int(run_cfg.seed), deterministic=bool(run_cfg.deterministic))

    resume = str(run_cfg.resume_ckpt).strip()
    ck = None
    model_cfg = run_cfg
    if resume:
        ck = load_ckpt(resume, device)
        if "cfg" in ck and isinstance(ck["cfg"], dict):
            model_cfg = TrainConfig.from_dict(ck["cfg"])

    cfg_dict = model_cfg.to_dict()
    cfg_dict.update({
        "data_root": run_cfg.data_root,
        "image_dir": run_cfg.image_dir,
        "meta_dir": run_cfg.meta_dir,
        "tags_dir": run_cfg.tags_dir,
        "caption_field": run_cfg.caption_field,
        "min_tag_count": run_cfg.min_tag_count,
        "require_512": run_cfg.require_512,
        "val_ratio": run_cfg.val_ratio,
        "cache_dir": run_cfg.cache_dir,
        "failed_list": run_cfg.failed_list,
        "seed": run_cfg.seed,
        "out_dir": run_cfg.out_dir,
        "batch_size": run_cfg.batch_size,
        "grad_accum_steps": run_cfg.grad_accum_steps,
        "num_workers": run_cfg.num_workers,
        "prefetch_factor": run_cfg.prefetch_factor,
        "lr": run_cfg.lr,
        "weight_decay": run_cfg.weight_decay,
        "max_steps": run_cfg.max_steps,
        "log_every": run_cfg.log_every,
        "save_every": run_cfg.save_every,
        "cond_drop_prob": run_cfg.cond_drop_prob,
        "amp": run_cfg.amp,
        "amp_dtype": run_cfg.amp_dtype,
        "compile": run_cfg.compile,
        "grad_clip_norm": run_cfg.grad_clip_norm,
        "ema_decay": run_cfg.ema_decay,
        "resume_ckpt": run_cfg.resume_ckpt,
        "deterministic": run_cfg.deterministic,
        "sanity_overfit_steps": run_cfg.sanity_overfit_steps,
        "sanity_overfit_images": run_cfg.sanity_overfit_images,
        "sanity_overfit_max_loss": run_cfg.sanity_overfit_max_loss,
    })

    # ----------------------------
    # Dataset + vocab
    # ----------------------------
    dcfg = DanbooruConfig(
        root=str(run_cfg.data_root),
        image_dir=str(run_cfg.image_dir),
        meta_dir=str(run_cfg.meta_dir),
        tags_dir=str(run_cfg.tags_dir),
        caption_field=str(run_cfg.caption_field),
        min_tag_count=int(run_cfg.min_tag_count),
        require_512=bool(run_cfg.require_512),
        val_ratio=float(run_cfg.val_ratio),
        seed=int(run_cfg.seed),
        cache_dir=str(run_cfg.cache_dir),
        failed_list=str(run_cfg.failed_list),
    )

    text_cfg = TextConfig(
        vocab_path=str(model_cfg.text_vocab_path),
        merges_path=str(model_cfg.text_merges_path),
        max_len=int(model_cfg.text_max_len),
        lowercase=True,
        strip_punct=True,
    )

    train_entries, _val_entries = build_or_load_index(dcfg)

    tokenizer = BPETokenizer.from_files(
        vocab_path=model_cfg.text_vocab_path,
        merges_path=model_cfg.text_merges_path,
        cfg=text_cfg,
    )

    cfg_dict["text_max_len"] = int(text_cfg.max_len)
    with open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "run_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(build_run_metadata(), f, sort_keys=False, allow_unicode=True)

    ds = DanbooruDataset(
        entries=train_entries,
        tokenizer=tokenizer,
        cond_drop_prob=float(run_cfg.cond_drop_prob),
        seed=int(run_cfg.seed),
    )

    nw = int(run_cfg.num_workers)
    dl = DataLoader(
        ds,
        batch_size=int(run_cfg.batch_size),
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=int(run_cfg.prefetch_factor) if nw > 0 else None,
        collate_fn=collate_with_tokenizer,
    )

    # ----------------------------
    # Model
    # ----------------------------
    unet_cfg = UNetConfig(
        image_channels=3,
        base_channels=int(model_cfg.base_channels),
        channel_mults=tuple(model_cfg.channel_mults),
        num_res_blocks=int(model_cfg.num_res_blocks),
        dropout=float(model_cfg.dropout),
        attn_resolutions=tuple(model_cfg.attn_resolutions),
        attn_heads=int(model_cfg.attn_heads),
        attn_head_dim=int(model_cfg.attn_head_dim),
        vocab_size=len(tokenizer.vocab),
        text_dim=int(model_cfg.text_dim),
        text_layers=int(model_cfg.text_layers),
        text_heads=int(model_cfg.text_heads),
        text_max_len=int(model_cfg.text_max_len),
        use_scale_shift_norm=bool(model_cfg.use_scale_shift_norm),
        grad_checkpointing=bool(model_cfg.grad_checkpointing),
    )

    model = UNet(unet_cfg).to(device)
    model = model.to(memory_format=torch.channels_last)

    if bool(run_cfg.compile) and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_cfg.lr),
        weight_decay=float(run_cfg.weight_decay),
        fused=(device.type == "cuda"),
    )

    use_amp = bool(run_cfg.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if run_cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=float(run_cfg.ema_decay))

    prediction_type = str(model_cfg.prediction_type)
    diff = Diffusion(
        DiffusionConfig(
            timesteps=int(model_cfg.timesteps),
            beta_start=float(model_cfg.beta_start),
            beta_end=float(model_cfg.beta_end),
            prediction_type=prediction_type,
        ),
        device=device,
    )

    start_step = 0
    if resume:
        if ck is None:
            ck = load_ckpt(resume, device)
        model.load_state_dict(ck["model"], strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        elif "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        if "ema" in ck:
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        for group in opt.param_groups:
            group["lr"] = float(run_cfg.lr)
        start_step = int(ck.get("step", 0)) + 1

    # ----------------------------
    # Train loop
    # ----------------------------
    max_steps = int(run_cfg.max_steps)
    grad_accum = int(run_cfg.grad_accum_steps)
    log_every = int(run_cfg.log_every)
    save_every = int(run_cfg.save_every)
    min_snr_gamma = float(run_cfg.min_snr_gamma)
    grad_clip = float(run_cfg.grad_clip_norm)

    webui_mode = _is_webui_mode()
    metrics_path = _webui_metrics_path()

    pbar = tqdm(
        total=int(run_cfg.max_steps),
        initial=start_step,
        desc="train",
        unit="step",
        disable=webui_mode,  # важное: webui -> без tqdm, иначе мусор в stdout
    )

    log_every = int(run_cfg.log_every)
    log_path = Path(run_cfg.out_dir) / "train_log.jsonl"

    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_step = start_step

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


    stop_requested = {"value": False}

    def _request_stop(signum, _frame) -> None:
        stop_requested["value"] = True
        print(f"[SIGNAL] stop requested via {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    it = iter(dl)
    start_time = time.perf_counter()
    last_log = start_time

    sanity_steps = int(run_cfg.sanity_overfit_steps)
    sanity_images = int(run_cfg.sanity_overfit_images)
    sanity_max_loss = float(run_cfg.sanity_overfit_max_loss)
    _sanity_overfit(
        model=model,
        tokenizer=tokenizer,
        entries=train_entries,
        diff=diff,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        steps=sanity_steps,
        max_images=sanity_images,
        max_loss=sanity_max_loss,
        opt=opt,
        scaler=scaler,
        ema=ema,
    )

    _emit_event_line(
        payload={
            "type": "status",
            "status": "start",
            "step": start_step,
            "resume": bool(resume),
            "out_dir": str(out_dir),
        },
        log_path=log_path,
        metrics_path=metrics_path,
        webui_mode=webui_mode,
    )

    for step in range(start_step, max_steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        last_batch_stats = {"x_std": None, "v_std": None}

        for _ in range(grad_accum):
            try:
                x0, txt_ids, txt_mask = next(it)
            except StopIteration:
                it = iter(dl)
                x0, txt_ids, txt_mask = next(it)

            x0 = x0.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            txt_ids = txt_ids.to(device, non_blocking=True)
            txt_mask = txt_mask.to(device, non_blocking=True)

            _assert_in_range("x0", x0, -1.0, 1.0)

            b = x0.shape[0]
            t = torch.randint(0, diff.cfg.timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            alpha_bar_t = diff.alpha_bar[t]
            _assert_finite("alpha_bar[t]", alpha_bar_t)
            xt = diff.q_sample(x0, t, noise)
            v_tgt = diff.v_target(x0, t, noise)
            last_batch_stats["x_std"] = float(x0.detach().std().cpu().item())
            last_batch_stats["v_std"] = float(v_tgt.detach().std().cpu().item())

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                v_pred = model(xt, t, txt_ids, txt_mask)
                _assert_finite("v_pred", v_pred)
                if v_pred.shape != v_tgt.shape:
                    raise RuntimeError("v_pred/v_target shape mismatch")
                per = F.mse_loss(v_pred, v_tgt.to(dtype=v_pred.dtype), reduction="none").mean(dim=[1, 2, 3])  # [B]
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
                    "meta": build_run_metadata(),
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

        if scaler.is_enabled():
            scaler.unscale_(opt)
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        else:
            grad_norm = _grad_norm(model)

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
                "meta": build_run_metadata(),
                "batch_stats": {
                    "bad_grads": bad_grads[:10],
                },
            })
            raise RuntimeError(f"Non-finite grads at step={step}: {bad_grads[:5]}")

        if step % log_every == 0:
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

            # CLI-UI (только если НЕ webui)
            if not webui_mode:
                pbar.set_postfix({
                    "loss": f"{total_loss:.6f}",
                    "img/s": f"{img_per_sec:.2f}",
                    "mem(MB)": f"{peak_mem:.0f}",
                    "eta(h)": f"{eta_h:.2f}",
                })

            payload = {
                "type": "metric",
                "step": step,
                "loss": float(total_loss),
                "lr": float(opt.param_groups[0]["lr"]),
                "grad_norm": float(grad_norm),
                "x_std": last_batch_stats["x_std"],
                "v_std": last_batch_stats["v_std"],
                "ema_decay": float(run_cfg.ema_decay),
                "cfg_drop_prob": float(run_cfg.cond_drop_prob),
                "img_per_sec": float(img_per_sec),
                "peak_mem_mb": float(peak_mem),
                "elapsed_sec": float(total_elapsed),
                "eta_h": float(eta_h),
                "sec_per_step": float(sec_per_step),
                "max_steps": int(run_cfg.max_steps),
            }
            _emit_event_line(
                payload=payload,
                log_path=log_path,
                metrics_path=metrics_path,
                webui_mode=webui_mode,
            )

            last_log_time = now
            last_log_step = step

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if stop_requested["value"]:
            stop_path = out_dir / f"ckpt_stop_{step:07d}.pt"
            save_ckpt(str(stop_path), {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg_dict,
                "meta": build_run_metadata(),
            })
            _emit_event_line(
                payload={
                    "type": "status",
                    "status": "stopped",
                    "step": step,
                    "ckpt": str(stop_path),
                },
                log_path=log_path,
                metrics_path=metrics_path,
                webui_mode=webui_mode,
            )
            print(f"[STOP] saved {stop_path}")
            return

        if step % save_every == 0 and step > 0:
            ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
            save_ckpt(str(ckpt_path), {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg_dict,
                "meta": build_run_metadata(),
            })
            _emit_event_line(
                payload={
                    "type": "log",
                    "message": f"saved {ckpt_path}",
                    "step": step,
                },
                log_path=log_path,
                metrics_path=metrics_path,
                webui_mode=webui_mode,
            )
            print(f"[OK] saved {ckpt_path}")

    final_path = out_dir / "ckpt_final.pt"
    save_ckpt(str(final_path), {
        "step": max_steps - 1,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "cfg": cfg_dict,
        "meta": build_run_metadata(),
    })
    _emit_event_line(
        payload={
            "type": "status",
            "status": "done",
            "step": max_steps - 1,
            "ckpt": str(final_path),
        },
        log_path=log_path,
        metrics_path=metrics_path,
        webui_mode=webui_mode,
    )
    print(f"[DONE] saved {final_path}")


if __name__ == "__main__":
    main()
