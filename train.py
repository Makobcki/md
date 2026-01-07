from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from ddpm.data import DataConfig, ImageFolderRecursive
from ddpm.model import UNet
from ddpm.diffusion import DDPM, DiffusionConfig
from ddpm.utils import EMA, save_ckpt, load_ckpt

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_snr_weights(ddpm, t, snr_gamma=5.0):
    alpha_bar = ddpm.alpha_bar[t]
    snr = alpha_bar / (1.0 - alpha_bar)

    # Для v-prediction формула весов: min(SNR, gamma) / (SNR + 1)
    # Это делает обучение гораздо стабильнее
    weights = torch.stack([snr, torch.ones_like(t) * snr_gamma], dim=1).min(dim=1)[0] / (snr + 1)
    return weights.to(t.device)


def main() -> None:
    cfg = load_yaml("./train.yaml")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    torch.manual_seed(int(cfg["seed"]))
    torch.cuda.manual_seed_all(int(cfg["seed"]))

    ds = ImageFolderRecursive(DataConfig(
        data_root=cfg["data_root"],
        split=cfg["split"],
        image_size=int(cfg["image_size"]),
    ))

    dl = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = UNet(
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        fused=(device.type == "cuda"),
    )

    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]) and device.type == "cuda")
    ema = EMA(model, decay=0.999)

    diffusion = DDPM(DiffusionConfig(timesteps=int(cfg["timesteps"])), device=device)

    start_step = 0
    resume = str(cfg.get("resume_ckpt") or "").strip()
    if resume:
        ck = load_ckpt(resume, device)

        model_sd = model.state_dict()
        ck_sd = ck["model"]

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

        missing, unexpected = model.load_state_dict(filtered, strict=False)

        print(f"[LOAD] matched={len(filtered)} skipped={skipped}")
        print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")
        print("[LOAD] first skipped examples:")
        for k, got, exp in skipped_keys:
            print("  ", k, "got", got, "expected", exp)
        print(f"[LOAD] missing={len(missing)} unexpected={len(unexpected)}")

        # ВАЖНО: НЕ грузим optimizer/scaler/ema после изменения архитектуры
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        ema.shadow = ck["ema"]
        start_step = int(ck["step"]) + 1  # можно оставить 0, чтобы не путаться

        print("[RESUME] Loaded model.")

    model.train()

    pbar = tqdm(total=int(cfg["max_steps"]), initial=start_step, desc="train", unit="step")
    it = iter(dl)

    grad_accum = int(cfg["grad_accum_steps"])
    log_every = int(cfg["log_every"])
    save_every = int(cfg["save_every"])

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

            with torch.amp.autocast('cuda', enabled=bool(cfg.get("amp", False))):
                pred = model(xt, t)

                # ВАЖНО: reduction="none", чтобы получить лосс для каждого примера в батче
                loss_mse = F.mse_loss(pred, v_target, reduction="none")

                # Усредняем по пространству [C, H, W], оставляя размер [B]
                loss_mse = loss_mse.mean(dim=[1, 2, 3])

                # Теперь веса Min-SNR корректно умножаются на лосс каждого элемента батча
                weights = get_snr_weights(diffusion, t, snr_gamma=5.0)
                loss = (loss_mse * weights).mean()

                loss = loss / grad_accum

            total_loss += loss.detach().item()

            scaler.scale(loss).backward()

            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN loss at step {step}")

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

        ema.update(model)

        if step % log_every == 0:
            pbar.set_postfix({"loss": total_loss})

        pbar.update(1)

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
