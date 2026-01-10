#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from ddpm.diffusion import DDPM, DiffusionConfig
from ddpm.ddim import DDIMSampler
from ddpm.model import UNet
from ddpm.text import Vocab, SimpleTokenizer, TextEncoder
from ddpm.utils import EMA, load_ckpt


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--neg", default="")
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = load_ckpt(args.ckpt, device)

    cfg = ck["cfg"]
    diffusion = DDPM(DiffusionConfig(timesteps=int(cfg["timesteps"])), device=device)

    # --- text ---
    vocab_path = Path(cfg["vocab_path"])
    vocab = Vocab.load(vocab_path)
    tokenizer = SimpleTokenizer(vocab, max_len=int(cfg["max_text_len"]))

    text_encoder = TextEncoder(
        vocab_size=vocab.size,
        dim=int(cfg["cond_dim"]),
        max_len=int(cfg["max_text_len"]),
        depth=int(cfg.get("text_depth", 4)),
        heads=int(cfg.get("text_heads", 8)),
        dropout=float(cfg.get("text_dropout", 0.0)),
    ).to(device)
    text_encoder.load_state_dict(ck["text_encoder"], strict=True)
    text_encoder.eval()

    # --- unet ---
    model = UNet(
        image_channels=3,
        base_channels=int(cfg["base_channels"]),
        channel_mults=tuple(cfg["channel_mults"]),
        num_res_blocks=int(cfg["num_res_blocks"]),
        dropout=float(cfg["dropout"]),
        grad_checkpoint=False,
        attn_resolutions=tuple(cfg.get("attn_resolutions", [32])),
        attn_heads=int(cfg.get("attn_heads", 2)),
        attn_head_dim=int(cfg.get("attn_head_dim", 32)),
        cond_dim=int(cfg["cond_dim"]),
        xattn_resolutions=tuple(cfg.get("xattn_resolutions", [32, 16])),
    ).to(device, memory_format=torch.channels_last)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()

    # EMA for UNet only
    if "ema" in ck:
        ema = EMA(model)
        ema.shadow = ck["ema"]
        ema.copy_to(model)

    # contexts
    if args.prompt.strip():
        cond_tokens = tokenizer.batch_encode([args.prompt]).to(device)
        cond_ctx = text_encoder(cond_tokens)
    else:
        cond_ctx = None

    if args.neg.strip():
        uncond_tokens = tokenizer.batch_encode([args.neg]).to(device)
        uncond_ctx = text_encoder(uncond_tokens)
    else:
        # empty prompt
        uncond_tokens = tokenizer.empty_tokens(1).to(device)
        uncond_ctx = text_encoder(uncond_tokens)

    sampler = DDIMSampler(diffusion)
    x = sampler.sample(
        model,
        shape=(args.n, 3, int(cfg["image_size"]), int(cfg["image_size"])),
        steps=args.steps,
        eta=0.0,
        device=device,
        progress=True,
        cond_ctx=cond_ctx,
        uncond_ctx=uncond_ctx,
        cfg_scale=float(args.cfg) if cond_ctx is not None else 0.0,
    )

    x = (x.clamp(-1, 1) + 1) / 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(x, out, nrow=int((args.n) ** 0.5))
    print(f"[OK] saved {out}")


if __name__ == "__main__":
    main()
