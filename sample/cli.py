#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion.events import EventBus, StdoutJsonSink
from diffusion.perf import PerfConfig, configure_performance

from .build import build_all
from samplers.guided_v import (
    ddpm_ancestral_sample,
    ddim_sample,
    dpm_solver_sample,
    euler_sample,
    heun_sample,
)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--neg", default="", help="Deprecated; use --neg_prompt")
    ap.add_argument("--neg_prompt", default="")
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument(
        "--sampler",
        default="ddim",
        choices=("ddim", "diffusion", "euler", "heun", "dpm_solver"),
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    event_bus = EventBus([StdoutJsonSink()])

    if args.seed is None:
        base_seed = random.SystemRandom().randint(0, 2**31 - 1)
    else:
        base_seed = int(args.seed)
    seeds = [base_seed + i for i in range(args.n)]
    event_bus.emit({"type": "status", "status": "start", "seed": base_seed, "n": args.n})

    built = build_all(args.ckpt, device)

    mode_label = "ENABLED" if built.use_text_conditioning else "DISABLED (unconditional diffusion)"
    print(f"[INFO] Text conditioning: {mode_label}")

    configure_performance(
        PerfConfig(
            tf32=bool(built.cfg.get("tf32", True)),
            cudnn_benchmark=bool(built.cfg.get("cudnn_benchmark", True)),
            channels_last=bool(built.cfg.get("channels_last", True)),
            enable_flash_sdp=bool(built.cfg.get("enable_flash_sdp", True)),
            enable_mem_efficient_sdp=bool(built.cfg.get("enable_mem_efficient_sdp", True)),
            enable_math_sdp=bool(built.cfg.get("enable_math_sdp", False)),
        ),
        device,
    )

    tokenizer = built.tokenizer
    if built.use_text_conditioning:
        if tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")
        prompt = args.prompt.strip()
        neg_prompt = args.neg_prompt if args.neg_prompt else args.neg

        if prompt:
            cond_ids, cond_mask = tokenizer.encode(prompt)
            cond_ids = cond_ids.unsqueeze(0).to(device)
            cond_mask = cond_mask.unsqueeze(0).to(device)

            uncond_text = neg_prompt.strip() if neg_prompt.strip() else ""
            uncond_ids, uncond_mask = tokenizer.encode(uncond_text)
            uncond_ids = uncond_ids.unsqueeze(0).to(device)
            uncond_mask = uncond_mask.unsqueeze(0).to(device)

            cfg_scale = float(args.cfg)
        else:
            cond_ids, cond_mask = tokenizer.encode("")
            cond_ids = cond_ids.unsqueeze(0).to(device)
            cond_mask = cond_mask.unsqueeze(0).to(device)

            uncond_ids = None
            uncond_mask = None
            cfg_scale = 0.0
    else:
        cond_ids = None
        cond_mask = None
        uncond_ids = None
        uncond_mask = None
        cfg_scale = 1.0
        if args.prompt.strip() or args.neg_prompt.strip() or args.neg.strip():
            print("[WARN] use_text_conditioning=false, prompt ignored")

    sampler = getattr(args, "sampler", "heun")

    h = built.h
    w = built.w
    latent_h = built.latent_h
    latent_w = built.latent_w

    per_image_steps = max(int(args.steps), 0)
    total_steps = per_image_steps * max(args.n, 1)

    samples = []
    for i, seed in enumerate(seeds):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        if built.mode == "latent":
            if latent_h is None or latent_w is None:
                raise RuntimeError("latent_h/latent_w are not resolved for latent mode.")
            noise = torch.randn((1, built.image_channels, latent_h, latent_w), device=device, generator=gen)
            shape = (1, built.image_channels, latent_h, latent_w)
        else:
            noise = torch.randn((1, 3, h, w), device=device, generator=gen)
            shape = (1, 3, h, w)

        def _progress_cb(step: int, _total: int, image_index: int = i) -> None:
            event_bus.emit(
                {
                    "type": "metric",
                    "step": image_index * per_image_steps + step,
                    "max_steps": total_steps,
                    "sampler": sampler,
                }
            )

        sample_kwargs = dict(
            model=built.model,
            diffusion=built.diffusion,
            shape=shape,
            txt_ids=cond_ids,
            txt_mask=cond_mask,
            steps=args.steps,
            cfg_scale=cfg_scale,
            uncond_ids=uncond_ids,
            uncond_mask=uncond_mask,
            self_conditioning=built.self_conditioning,
            noise=noise,
            generator=gen,
            progress_cb=_progress_cb,
        )

        if sampler == "ddim":
            x = ddim_sample(eta=0.0, clamp_x0=True, **sample_kwargs)
        elif sampler == "diffusion":
            x = ddpm_ancestral_sample(**sample_kwargs)
        elif sampler == "euler":
            x = euler_sample(**sample_kwargs)
        elif sampler == "dpm_solver":
            x = dpm_solver_sample(**sample_kwargs)
        elif sampler == "heun":
            x = heun_sample(**sample_kwargs)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        if built.mode == "latent":
            if built.vae is None:
                raise RuntimeError("VAE is not initialized for latent mode.")
            x = built.vae.decode(x)

        samples.append(x)

    x = torch.cat(samples, dim=0)

    if built.mode == "pixel":
        x = (x.clamp(-1, 1) + 1) / 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(x, out, nrow=int((args.n) ** 0.5))
    event_bus.emit({"type": "status", "status": "done", "path": str(out)})
    print(f"[OK] saved {out}")


if __name__ == "__main__":
    main()
