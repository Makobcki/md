#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion.config import TrainConfig
from diffusion.ddim import (
    ddpm_ancestral_sample,
    ddim_sample,
    dpm_solver_sample,
    euler_sample,
    heun_sample,
)
from diffusion.diffusion import Diffusion, DiffusionConfig
from diffusion.events import EventBus, StdoutJsonSink
from diffusion.model import UNet, UNetConfig
from diffusion.perf import PerfConfig, configure_performance
from diffusion.text import BPETokenizer, TextConfig
from diffusion.utils import EMA, load_ckpt
from diffusion.vae import VAEWrapper


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

    ck = load_ckpt(args.ckpt, device)

    cfg = TrainConfig.from_dict(ck["cfg"]).to_dict()
    diffusion = Diffusion(
        DiffusionConfig(
            timesteps=int(cfg.get("timesteps", 1000)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 2e-2)),
            prediction_type=str(cfg.get("prediction_type", "v")),
        ),
        device=device,
    )

    configure_performance(
        PerfConfig(
            tf32=bool(cfg.get("tf32", True)),
            cudnn_benchmark=bool(cfg.get("cudnn_benchmark", True)),
            channels_last=bool(cfg.get("channels_last", True)),
            enable_flash_sdp=bool(cfg.get("enable_flash_sdp", True)),
            enable_mem_efficient_sdp=bool(cfg.get("enable_mem_efficient_sdp", True)),
            enable_math_sdp=bool(cfg.get("enable_math_sdp", False)),
        ),
        device,
    )

    vocab_path = cfg.get("text_vocab_path")
    merges_path = cfg.get("text_merges_path")
    if not vocab_path or not merges_path:
        raise RuntimeError("Checkpoint missing text_vocab_path/text_merges_path.")

    text_cfg = TextConfig(
        vocab_path=str(vocab_path),
        merges_path=str(merges_path),
        max_len=int(cfg.get("text_max_len", 64)),
        lowercase=True,
        strip_punct=True,
    )
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, text_cfg)

    mode = str(cfg.get("mode", "pixel"))
    image_channels = 3 if mode == "pixel" else int(cfg.get("latent_channels", 4))
    unet_cfg = UNetConfig(
        image_channels=image_channels,
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

    channels_last = bool(cfg.get("channels_last", True))
    if channels_last:
        model = UNet(unet_cfg).to(device, memory_format=torch.channels_last)
    else:
        model = UNet(unet_cfg).to(device)
    model.load_state_dict(ck["model"], strict=True)

    # EMA for UNet only
    if "ema" in ck:
        ema = EMA(model)
        ema.shadow = ck["ema"]
        ema.copy_to(model)

    model.eval()

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

    sampler = getattr(args, "sampler", "heun")
    h = int(cfg.get("image_size", 512))
    w = int(cfg.get("image_size", 512))
    if mode == "latent":
        downsample = int(cfg.get("latent_downsample_factor", 8))
        if h % downsample != 0 or w % downsample != 0:
            raise RuntimeError("image_size must be divisible by latent_downsample_factor.")
        latent_h = h // downsample
        latent_w = w // downsample
        vae_pretrained = str(cfg.get("vae_pretrained", ""))
        if not vae_pretrained:
            raise RuntimeError("latent mode requires vae_pretrained in checkpoint config.")
        vae = VAEWrapper(
            pretrained=vae_pretrained,
            freeze=True,
            scaling_factor=float(cfg.get("vae_scaling_factor", 0.18215)),
            device=device,
            dtype=torch.bfloat16 if cfg.get("amp_dtype") == "bf16" else torch.float16,
        )
    per_image_steps = max(int(args.steps), 0)
    total_steps = per_image_steps * max(args.n, 1)

    samples = []
    for i, seed in enumerate(seeds):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        if mode == "latent":
            noise = torch.randn((1, image_channels, latent_h, latent_w), device=device, generator=gen)
        else:
            noise = torch.randn((1, 3, h, w), device=device, generator=gen)
        def _progress_cb(step: int, _total: int, image_index: int = i) -> None:
            event_bus.emit({
                "type": "metric",
                "step": image_index * per_image_steps + step,
                "max_steps": total_steps,
                "sampler": sampler,
            })

        shape = (1, image_channels, latent_h, latent_w) if mode == "latent" else (1, 3, h, w)
        sample_kwargs = dict(
            model=model,
            diffusion=diffusion,
            shape=shape,
            txt_ids=cond_ids,
            txt_mask=cond_mask,
            steps=args.steps,
            cfg_scale=cfg_scale,
            uncond_ids=uncond_ids,
            uncond_mask=uncond_mask,
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
        if mode == "latent":
            x = vae.decode(x)
        samples.append(x)

    x = torch.cat(samples, dim=0)

    if mode == "pixel":
        x = (x.clamp(-1, 1) + 1) / 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(x, out, nrow=int((args.n) ** 0.5))
    event_bus.emit({"type": "status", "status": "done", "path": str(out)})
    print(f"[OK] saved {out}")


if __name__ == "__main__":
    main()
