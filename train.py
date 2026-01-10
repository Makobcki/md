from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ddpm.data import DataConfig, ImageFolderRecursive
from ddpm.model import UNet
from ddpm.diffusion import DDPM, DiffusionConfig
from ddpm.utils import EMA, build_run_metadata, load_ckpt, save_ckpt, seed_everything

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_snr_weights(ddpm, t, snr_gamma=5.0):
    eps = 1e-8
    alpha_bar = ddpm.alpha_bar[t].clamp(min=eps, max=1.0 - eps)
    snr = alpha_bar / (1.0 - alpha_bar + eps)
    gamma = torch.full_like(snr, float(snr_gamma))
    return torch.minimum(snr, gamma) / (snr + 1.0)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    seed_everything(worker_seed, deterministic=False)

def _yaml_safe(obj):
    """Рекурсивно приводит объект к типам, которые точно сериализуются в YAML."""
    # базовые типы
    if obj is None or type(obj) in (bool, int, float, str):
        return obj

    # любые "похожие на строку" (TorchVersion, numpy.str_, etc.) -> строго builtin str
    if isinstance(obj, str):
        return str(obj)

    # bytes -> строка (чтобы не словить бинарные теги)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    # Path / device / dtype / любые странные объекты -> str
    from pathlib import Path
    if isinstance(obj, Path):
        return str(obj)

    # dict
    if isinstance(obj, dict):
        return {str(_yaml_safe(k)): _yaml_safe(v) for k, v in obj.items()}

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        return [_yaml_safe(x) for x in obj]

    # numpy scalar (если вдруг)
    try:
        import numpy as np  # optional
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    # torch scalar / tensor 0-dim
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                return obj.item()
            return f"<tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}>"
    except Exception:
        pass

    # fallback: строковое представление
    return str(obj)

def _save_yaml(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_yaml_safe(obj), f, sort_keys=False, allow_unicode=True)

def _normalize_state_dict_keys(sd: dict) -> dict:
    """
    Делает state_dict совместимым между:
    - torch.compile (ключи начинаются с _orig_mod.)
    - DDP (ключи начинаются с module.)
    """
    if not isinstance(sd, dict):
        return sd

    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out

def _get_autocast_device(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./train.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    webui_mode = os.environ.get("WEBUI") == "1"
    webui_run_dir = os.environ.get("WEBUI_RUN_DIR")

    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.resume is not None:
        cfg["resume_ckpt"] = str(args.resume)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    seed_everything(int(cfg["seed"]), deterministic=bool(cfg.get("deterministic", False)))

    run_meta = build_run_metadata()
    _save_yaml(out_dir / "config.yaml", cfg)
    _save_yaml(out_dir / "run_meta.yaml", run_meta)
    if webui_run_dir:
        _save_yaml(Path(webui_run_dir) / "config_snapshot.yaml", cfg)
        _save_yaml(Path(webui_run_dir) / "run_meta.yaml", run_meta)

    ds = ImageFolderRecursive(DataConfig(
        data_root=cfg["data_root"],
        split=cfg["split"],
        image_size=int(cfg["image_size"]),
    ))

    num_workers = int(cfg["num_workers"])
    use_workers = num_workers > 0
    generator = torch.Generator()
    generator.manual_seed(int(cfg["seed"]))

    dl_kwargs = {
        "batch_size": int(cfg["batch_size"]),
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": use_workers,
        "worker_init_fn": _seed_worker if use_workers else None,
        "generator": generator,
    }
    if use_workers:
        dl_kwargs["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

    dl = DataLoader(ds, **dl_kwargs)

    base_model = UNet(
        image_channels=3,
        base_channels=int(cfg["base_channels"]),
        channel_mults=tuple(cfg["channel_mults"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        dropout=float(cfg["dropout"]),
        grad_checkpoint=bool(cfg["grad_checkpoint"]),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [48])),
        attn_heads=int(cfg.get("attn_heads", 1)),
        attn_head_dim=int(cfg.get("attn_head_dim", 32)),
    ).to(device, memory_format=torch.channels_last)

    model = base_model
    if bool(cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(base_model)

    opt = torch.optim.AdamW(
        base_model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        fused=(device.type == "cuda"),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]) and device.type == "cuda")
    ema = EMA(base_model, decay=0.999)

    diffusion = DDPM(DiffusionConfig(timesteps=int(cfg["timesteps"])), device=device)

    start_step = 0
    resume = str(cfg.get("resume_ckpt") or "").strip()
    if resume:
        ck = load_ckpt(resume, device)

        model_sd = base_model.state_dict()
        ck_sd = _normalize_state_dict_keys(ck["model"])

        filtered = {}
        skipped = 0
        skipped_keys = []

        for k, v in ck_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped += 1
                if len(skipped_keys) < 20:
                    # покажем первые 20 для диагностики
                    exp = tuple(model_sd[k].shape) if k in model_sd else None
                    skipped_keys.append((k, tuple(v.shape), exp))

        missing, unexpected = base_model.load_state_dict(filtered, strict=False)

        matched = len(filtered)
        total = len(model_sd)
        if matched == 0:
            raise RuntimeError(
                "Resume failed: matched=0. Похоже, чекпоинт сохранён из torch.compile "
                "(ключи _orig_mod.*) или архитектура не совпадает. "
                "Добавь нормализацию ключей state_dict (strip _orig_mod./module.)."
            )

        print(f"[LOAD] matched={matched} skipped={skipped} total={total}")
        print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} total={total}")
        print("[LOAD] first skipped examples:")
        for k, got, exp in skipped_keys:
            print("  ", k, "got", got, "expected", exp)
        print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} total={total}")

        # ВАЖНО: НЕ грузим optimizer/scaler/ema после изменения архитектуры
        if "opt" in ck and "scaler" in ck and "ema" in ck:
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            ema.shadow = _normalize_state_dict_keys(ck["ema"])
        start_step = int(ck["step"]) + 1  # можно оставить 0, чтобы не путаться

        print("[RESUME] Loaded model.")

    model.train()

    pbar = tqdm(
        total=int(cfg["max_steps"]),
        initial=start_step,
        desc="train",
        unit="step",
        disable=webui_mode,
    )
    it = iter(dl)

    grad_accum = int(cfg["grad_accum_steps"])
    log_every = int(cfg["log_every"])
    save_every = int(cfg["save_every"])
    log_path = out_dir / "train_log.jsonl"
    metrics_path = Path(webui_run_dir) / "metrics" / "train_metrics.jsonl" if webui_run_dir else None
    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_step = start_step
    last_step_time = start_time
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    stop_requested = {"value": False}

    def _request_stop(signum, _frame) -> None:
        stop_requested["value"] = True
        print(f"[SIGNAL] Received {signal.Signals(signum).name}. Will stop after current step.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    for step in range(start_step, int(cfg["max_steps"])):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(grad_accum):
            try:
                x1 = next(it)
            except StopIteration:
                it = iter(dl)
                x1 = next(it)

            x1 = x1.to(device, non_blocking=True).to(memory_format=torch.channels_last)

            b = x1.shape[0]
            t = torch.randint(0, int(cfg["timesteps"]), (b,), device=device)
            noise = torch.randn_like(x1)
            xt = diffusion.q_sample(x1, t, noise)

            v_target = torch.sqrt(diffusion.alpha_bar[t])[:, None, None, None] * noise - \
                       torch.sqrt(1 - diffusion.alpha_bar[t])[:, None, None, None] * x1



            with torch.amp.autocast(_get_autocast_device(device), enabled=bool(cfg.get("amp", False))):
                pred = model(xt, t)

                # ВАЖНО: reduction="none", чтобы получить лосс для каждого примера в батче
                loss_mse = F.mse_loss(pred, v_target, reduction="none")
                loss_mse = loss_mse.mean(dim=[1, 2, 3])

                snr_gamma = float(cfg.get("min_snr_gamma", 5.0))
                weights = get_snr_weights(diffusion, t, snr_gamma=snr_gamma)

                loss = (loss_mse * weights).mean()

                if not torch.isfinite(loss).item():
                    raise RuntimeError(f"NaN/Inf loss at step {step}")

                loss = loss / grad_accum

            total_loss += loss.detach().item()

            scaler.scale(loss).backward()

        # gradient clipping
        grad_clip = float(cfg.get("grad_clip_norm", 0.0))
        if grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # optimizer step
        if scaler.is_enabled():
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        ema.update(base_model)

        if step % log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.perf_counter()
            elapsed = now - last_log_time
            steps_done = step - last_log_step if step > last_log_step else 1
            images = steps_done * int(cfg["batch_size"]) * grad_accum
            img_per_sec = images / max(elapsed, 1e-9)

            peak_mem = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            )

            total_elapsed = now - start_time
            steps_left = int(cfg["max_steps"]) - step - 1
            step_time = elapsed / max(steps_done, 1)
            eta_h = round(steps_left * step_time / 3600, 3)

            if not webui_mode:
                pbar.set_postfix({"loss": total_loss, "img/s": f"{img_per_sec:.1f}", "mem(MB)": f"{peak_mem:.0f}"})

            line = json.dumps({
                "type": "metric",
                "step": step,
                "loss": total_loss,
                "img_per_sec": img_per_sec,
                "peak_mem_mb": peak_mem,
                "elapsed_sec": total_elapsed,
                "eta_h": eta_h,
                "sec_per_step": step_time,
                "max_steps": int(cfg["max_steps"]),
            }, ensure_ascii=False)

            # 1 файл — 1 строка
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            if webui_mode:
                print(line)
                if metrics_path:
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(metrics_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")

            last_log_time = now
            last_log_step = step

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        pbar.update(1)

        if stop_requested["value"]:
            stop_path = out_dir / f"ckpt_stop_{step:07d}.pt"
            save_ckpt(str(stop_path), {
                "step": step,
                "model": base_model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg,
            })
            print(f"[STOP] saved {stop_path}")
            return

        if step % save_every == 0 and step > 0:
            save_ckpt(str(out_dir / f"ckpt_{step:07d}.pt"), {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg,
            })

    # final save
    final_path = out_dir / "ckpt_final.pt"
    save_ckpt(str(final_path), {
        "step": int(cfg["max_steps"]) - 2,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "cfg": cfg,
    })
    print(f"[DONE] saved {final_path}")


if __name__ == "__main__":
    main()
