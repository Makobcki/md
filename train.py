from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from ddpm.data import DanbooruConfig, TextConfig, SimpleTokenizer, DanbooruDataset, build_or_load_index, collate_with_tokenizer

from ddpm.model import UNet, UNetConfig
from ddpm.ddim import Diffusion, DiffusionConfig
from ddpm.utils import EMA, load_ckpt, save_ckpt

def _is_webui_mode() -> bool:
    return os.environ.get("WEBUI") == "1"


def _webui_metrics_path() -> Path | None:
    run_dir = os.environ.get("WEBUI_RUN_DIR")
    if not run_dir:
        return None
    return Path(run_dir) / "metrics" / "train_metrics.jsonl"


def _emit_metric_line(
    *,
    line: str,
    log_path: Path,
    metrics_path: Path | None,
    webui_mode: bool,
) -> None:
    # всегда пишем локальный jsonl
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    # webui: печатаем в stdout + (опционально) отдельный metrics файл
    if webui_mode:
        print(line, flush=True)
        if metrics_path:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")



def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def _sanity_overfit(
    *,
    model: UNet,
    tokenizer: SimpleTokenizer,
    text_cfg: TextConfig,
    entries: list[dict],
    diff: Diffusion,
    device: torch.device,
    use_amp: bool,
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
        text_cfg=text_cfg,
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

        with torch.amp.autocast("cuda", enabled=use_amp):
            v_pred = model(xt, t, txt_ids, txt_mask)
            if v_pred.shape != v_tgt.shape or v_pred.dtype != v_tgt.dtype:
                raise RuntimeError("v_pred/v_target shape or dtype mismatch in sanity overfit")
            loss = F.mse_loss(v_pred, v_tgt)

        scaler.scale(loss).backward()
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

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.resume:
        cfg["resume_ckpt"] = args.resume

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    out_dir = Path(cfg["out_dir"])
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

    seed_everything(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))

    # ----------------------------
    # Dataset + vocab
    # ----------------------------
    dcfg = DanbooruConfig(
        root=str(cfg["data_root"]),
        image_dir=str(cfg.get("image_dir", "image_512")),
        meta_dir=str(cfg.get("meta_dir", "meta")),
        caption_field=str(cfg.get("caption_field", "caption_llava_34b_no_tags_short")),
        min_tag_count=int(cfg.get("min_tag_count", 8)),
        require_512=bool(cfg.get("require_512", True)),
        val_ratio=float(cfg.get("val_ratio", 0.01)),
        seed=int(cfg["seed"]),
        cache_dir=str(cfg.get("cache_dir", ".cache")),
    )

    text_cfg = TextConfig(
        vocab_size=int(cfg.get("vocab_size", 50_000)),
        max_len=int(cfg.get("text_max_len", 64)),
        lowercase=True,
        strip_punct=True,
    )

    train_entries, _val_entries = build_or_load_index(dcfg)

    vocab_path = out_dir / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        tokenizer = SimpleTokenizer(vocab=vocab, text_cfg=text_cfg)
    else:
        captions = [e["caption"] for e in train_entries]
        tokenizer = SimpleTokenizer.build(captions, text_cfg)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer.vocab, f, ensure_ascii=False)

    cfg["vocab_path"] = str(vocab_path)
    cfg["text_max_len"] = int(text_cfg.max_len)
    with open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    ds = DanbooruDataset(
        entries=train_entries,
        text_cfg=text_cfg,
        tokenizer=tokenizer,
        cond_drop_prob=float(cfg.get("cond_drop_prob", 0.1)),
        seed=int(cfg["seed"]),
    )

    nw = int(cfg.get("num_workers", 8))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=int(cfg.get("prefetch_factor", 2)) if nw > 0 else None,
        collate_fn=collate_with_tokenizer,
    )

    # ----------------------------
    # Model
    # ----------------------------
    unet_cfg = UNetConfig(
        image_channels=3,
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 3, 4])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [32, 16])),
        attn_heads=int(cfg.get("attn_heads", 4)),
        attn_head_dim=int(cfg.get("attn_head_dim", 32)),
        vocab_size=len(tokenizer.vocab),
        text_dim=int(cfg.get("text_dim", 256)),
        text_layers=int(cfg.get("text_layers", 4)),
        text_heads=int(cfg.get("text_heads", 4)),
        text_max_len=int(cfg.get("text_max_len", 64)),
        use_scale_shift_norm=bool(cfg.get("use_scale_shift_norm", False)),
    )

    model = UNet(unet_cfg).to(device)
    model = model.to(memory_format=torch.channels_last)

    if bool(cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        fused=(device.type == "cuda"),
    )

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=float(cfg.get("ema_decay", 0.999)))

    diff = Diffusion(
        DiffusionConfig(
            timesteps=int(cfg.get("timesteps", 1000)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 2e-2)),
        ),
        device=device,
    )

    start_step = 0
    resume = str(cfg.get("resume_ckpt", "")).strip()
    if resume:
        ck = load_ckpt(resume, device)
        model.load_state_dict(ck["model"], strict=True)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        elif "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        if "ema" in ck:
            ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        start_step = int(ck.get("step", 0)) + 1

    # ----------------------------
    # Train loop
    # ----------------------------
    max_steps = int(cfg.get("max_steps", 120_000))
    grad_accum = int(cfg.get("grad_accum_steps", 8))
    log_every = int(cfg.get("log_every", 50))
    save_every = int(cfg.get("save_every", 2000))
    min_snr_gamma = float(cfg.get("min_snr_gamma", 5.0))
    grad_clip = float(cfg.get("grad_clip_norm", 1.0))

    webui_mode = _is_webui_mode()
    metrics_path = _webui_metrics_path()

    pbar = tqdm(
        total=int(cfg["max_steps"]),
        initial=start_step,
        desc="train",
        unit="step",
        disable=webui_mode,  # важное: webui -> без tqdm, иначе мусор в stdout
    )

    log_every = int(cfg["log_every"])
    log_path = Path(cfg["out_dir"]) / "train_log.jsonl"

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

    sanity_steps = int(cfg.get("sanity_overfit_steps", 0))
    sanity_images = int(cfg.get("sanity_overfit_images", 0))
    sanity_max_loss = float(cfg.get("sanity_overfit_max_loss", 0.1))
    _sanity_overfit(
        model=model,
        tokenizer=tokenizer,
        text_cfg=text_cfg,
        entries=train_entries,
        diff=diff,
        device=device,
        use_amp=use_amp,
        steps=sanity_steps,
        max_images=sanity_images,
        max_loss=sanity_max_loss,
        opt=opt,
        scaler=scaler,
        ema=ema,
    )

    for step in range(start_step, max_steps):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

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

            with torch.amp.autocast("cuda", enabled=use_amp):
                v_pred = model(xt, t, txt_ids, txt_mask)
                if v_pred.shape != v_tgt.shape or v_pred.dtype != v_tgt.dtype:
                    raise RuntimeError("v_pred/v_target shape or dtype mismatch")
                per = F.mse_loss(v_pred, v_tgt, reduction="none").mean(dim=[1, 2, 3])  # [B]
                w = get_min_snr_weights(diff.alpha_bar[t], gamma=min_snr_gamma)        # [B]
                loss = (per * w).mean() / grad_accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step={step}: {loss.item()}")

            total_loss += float(loss.detach().cpu())
            scaler.scale(loss).backward()

        if grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        ema.update(model)

        if step % log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.perf_counter()
            elapsed = now - last_log_time
            steps_done = max(step - last_log_step, 1)

            # сколько "картинок" реально прошло за этот лог-интервал
            images = steps_done * int(cfg["batch_size"]) * int(cfg["grad_accum_steps"])
            img_per_sec = images / max(elapsed, 1e-9)

            peak_mem = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            )

            total_elapsed = now - start_time
            steps_left = int(cfg["max_steps"]) - step - 1
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
                "img_per_sec": float(img_per_sec),
                "peak_mem_mb": float(peak_mem),
                "elapsed_sec": float(total_elapsed),
                "eta_h": float(eta_h),
                "sec_per_step": float(sec_per_step),
                "max_steps": int(cfg["max_steps"]),
            }
            line = json.dumps(payload, ensure_ascii=False)

            _emit_metric_line(
                line=line,
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
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "tokenizer_vocab": tokenizer.vocab,
                "cfg": cfg,
            })
            print(f"[STOP] saved {stop_path}")
            return

        if step % save_every == 0 and step > 0:
            ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
            save_ckpt(str(ckpt_path), {
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "tokenizer_vocab": tokenizer.vocab,
                "cfg": cfg,
            })
            print(f"[OK] saved {ckpt_path}")

    final_path = out_dir / "ckpt_final.pt"
    save_ckpt(str(final_path), {
        "step": max_steps - 1,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "tokenizer_vocab": tokenizer.vocab,
        "cfg": cfg,
    })
    print(f"[DONE] saved {final_path}")


if __name__ == "__main__":
    main()
