from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from ddpm.diffusion import DDPM, DiffusionConfig
from ddpm.model import UNet
from ddpm.utils import EMA, load_ckpt, save_ckpt, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default="./runs/sanity/ckpt_sanity.pt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=True)

    model = UNet(
        image_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 3),
        num_res_blocks=1,
        dropout=0.0,
        grad_checkpoint=False,
        attn_resolutions=(16,),
        attn_heads=2,
        attn_head_dim=16,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, decay=0.9)

    diffusion = DDPM(DiffusionConfig(timesteps=100), device=device)

    model.train()
    for step in range(args.steps):
        x0 = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        t = torch.randint(0, diffusion.cfg.timesteps, (args.batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = diffusion.q_sample(x0, t, noise)
        v_target = diffusion.get_v(x0, noise, t)

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
            pred = model(xt, t)
            loss = F.mse_loss(pred, v_target)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)

    out_path = Path(args.out)
    save_ckpt(str(out_path), {
        "step": args.steps,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "cfg": {"image_size": args.image_size, "timesteps": diffusion.cfg.timesteps},
    })

    ck = load_ckpt(str(out_path), device)
    assert "model" in ck and "opt" in ck and "scaler" in ck and "ema" in ck
    print(f"[OK] sanity check saved and loaded {out_path}")


if __name__ == "__main__":
    main()
