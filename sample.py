#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torchvision.utils import save_image

from ddpm.config import TrainConfig
from ddpm.ddim import (
    ddpm_ancestral_sample,
    ddim_sample,
    dpm_solver_sample,
    euler_sample,
    heun_sample,
)
from ddpm.diffusion import Diffusion, DiffusionConfig
from ddpm.model import UNet, UNetConfig
from ddpm.text import BPETokenizer, TextConfig
from ddpm.utils import EMA, load_ckpt


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
        choices=("ddim", "ddpm", "euler", "heun", "dpm_solver"),
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def _emit_event(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    if args.seed is None:
        base_seed = random.SystemRandom().randint(0, 2**31 - 1)
    else:
        base_seed = int(args.seed)
    seeds = [base_seed + i for i in range(args.n)]
    _emit_event({"type": "status", "status": "start", "seed": base_seed, "n": args.n})

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

    model = UNet(unet_cfg).to(device, memory_format=torch.channels_last)
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
    per_image_steps = max(int(args.steps), 0)
    total_steps = per_image_steps * max(args.n, 1)

    samples = []
    for i, seed in enumerate(seeds):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        noise = torch.randn((1, 3, h, w), device=device, generator=gen)
        def _progress_cb(step: int, _total: int, image_index: int = i) -> None:
            _emit_event({
                "type": "metric",
                "step": image_index * per_image_steps + step,
                "max_steps": total_steps,
                "sampler": sampler,
            })

        sample_kwargs = dict(
            model=model,
            diffusion=diffusion,
            shape=(1, 3, h, w),
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
        elif sampler == "ddpm":
            x = ddpm_ancestral_sample(**sample_kwargs)
        elif sampler == "euler":
            x = euler_sample(**sample_kwargs)
        elif sampler == "dpm_solver":
            x = dpm_solver_sample(**sample_kwargs)
        elif sampler == "heun":
            x = heun_sample(**sample_kwargs)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        samples.append(x)

    x = torch.cat(samples, dim=0)

    x = (x.clamp(-1, 1) + 1) / 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(x, out, nrow=int((args.n) ** 0.5))
    _emit_event({"type": "status", "status": "done", "path": str(out)})
    print(f"[OK] saved {out}")


if __name__ == "__main__":
    main()
