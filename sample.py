from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from ddpm.ddim import DDIM, DDIMConfig
from ddpm.diffusion import DDPM, DiffusionConfig
from ddpm.model import UNet
from ddpm.utils import EMA, load_ckpt, seed_everything, strip_state_dict_prefixes


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,255]
    x = (x.clamp(-1, 1) + 1) * 0.5
    x = (x * 255.0).round().clamp(0, 255).to(torch.uint8)
    return x


def save_grid(images: torch.Tensor, out_path: Path, nrow: int = 4) -> None:
    """
    images: [B,C,H,W] uint8
    """
    b, c, h, w = images.shape
    ncol = (b + nrow - 1) // nrow
    canvas = Image.new("RGB", (nrow * w, ncol * h))

    idx = 0
    for r in range(ncol):
        for cidx in range(nrow):
            if idx >= b:
                break
            img = images[idx].permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(img, mode="RGB")
            canvas.paste(im, (cidx * w, r * h))
            idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ckpt_XXXX.pt")
    ap.add_argument("--out", default="./samples/grid.png")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=False)

    ck = load_ckpt(args.ckpt, device)
    cfg = ck["cfg"]

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

    ck["model"] = strip_state_dict_prefixes(ck["model"])
    if "ema" in ck and isinstance(ck["ema"], dict):
        ck["ema"] = strip_state_dict_prefixes(ck["ema"])

    model = model.float()

    # apply EMA
    ema = EMA(model)
    ema.shadow = ck["ema"]
    ema.copy_to(model)

    model.eval()

    # используем те же betas, что при обучении
    ddpm = DDPM(DiffusionConfig(timesteps=int(cfg["timesteps"])), device=device)

    ddim = DDIM(
        DDIMConfig(
            timesteps=int(cfg["timesteps"]),
            eta=0.0,  # 0 = максимально чётко, >0 = больше разнообразия
        ),
        betas=ddpm.betas,
    )

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad(), torch.autocast(autocast_device, enabled=device.type == "cuda"):
        x = ddim.sample(
            model,
            shape=(args.n, 3, int(cfg["image_size"]), int(cfg["image_size"])),
            steps=args.steps,  # теперь 50–200
        )

    imgs = to_uint8(x)
    save_grid(imgs, Path(args.out), nrow=4)
    print(f"[OK] saved {args.out}")


if __name__ == "__main__":
    main()
