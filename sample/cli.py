from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from data_loader.dataset import load_image_tensor
from diffusion.events import EventBus, StdoutJsonSink
from diffusion.perf import PerfConfig, configure_performance
from diffusion.utils.oom import is_torch_oom_error, print_torch_oom

from .build import build_all
from samplers import (
    ddpm_ancestral_sample,
    ddim_sample,
    dpm_solver_sample,
    euler_sample,
    heun_sample,
    sample_flow_euler,
    sample_flow_heun,
)


def _save_image_grid(x: torch.Tensor, path: str | Path, nrow: int) -> None:
    try:
        from torchvision.utils import save_image
    except Exception as exc:
        raise RuntimeError("Saving sample images requires a working torchvision install.") from exc
    save_image(x, path, nrow=nrow)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


@torch.no_grad()
def _main_impl() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=_positive_int, default=8)
    ap.add_argument("--steps", type=_positive_int, default=30)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--neg", default="", help="Deprecated; use --neg_prompt")
    ap.add_argument("--neg_prompt", default="")
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument(
        "--sampler",
        default="ddim",
        choices=("ddim", "diffusion", "euler", "heun", "dpm_solver", "flow_euler", "flow_heun"),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init-image", default="")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--mask", default="")
    ap.add_argument("--task", default="txt2img", choices=("txt2img", "img2img", "inpaint"))
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

    if str(built.cfg.get("architecture", built.ckpt.get("architecture", "unet_v1"))) == "mmdit_rf":
        if built.text_encoder is None:
            raise RuntimeError("Frozen text encoder bundle is not initialized for mmdit_rf.")
        if built.vae is None or built.latent_h is None or built.latent_w is None:
            raise RuntimeError("mmdit_rf sampling requires latent VAE metadata.")
        prompt = args.prompt.strip()
        neg_prompt = args.neg_prompt if args.neg_prompt else args.neg
        cond = built.text_encoder([prompt])
        uncond = built.text_encoder([neg_prompt.strip() if neg_prompt.strip() else ""])
        sampler = str(args.sampler)
        if sampler == "ddim":
            sampler = "flow_heun"
        if sampler not in {"flow_euler", "flow_heun"}:
            raise RuntimeError("architecture=mmdit_rf supports only flow_euler/flow_heun samplers.")

        source_latent = None
        mask_latent = None
        start_t = 1.0
        if args.init_image:
            img = load_image_tensor(args.init_image).unsqueeze(0).to(device)
            source_latent = built.vae.encode(img)
            start_t = max(0.0, min(float(args.strength), 1.0))
        if args.mask:
            from PIL import Image
            import numpy as np

            with Image.open(args.mask) as im:
                im = im.convert("L").resize((built.latent_w, built.latent_h))
                arr = np.asarray(im, dtype="float32") / 255.0
            mask_latent = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

        samples = []
        for i, seed in enumerate(seeds):
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            shape = (1, built.image_channels, built.latent_h, built.latent_w)
            noise = torch.randn(shape, device=device, generator=gen)
            if source_latent is not None:
                t = torch.tensor(start_t, device=device, dtype=source_latent.dtype).view(1, 1, 1, 1)
                noise = (1.0 - t) * source_latent + t * noise

            def _progress_cb(step: int, _total: int, image_index: int = i) -> None:
                event_bus.emit(
                    {
                        "type": "metric",
                        "step": image_index * int(args.steps) + step,
                        "max_steps": int(args.steps) * max(args.n, 1),
                        "sampler": sampler,
                    }
                )

            kwargs = {
                "model": built.model,
                "shape": shape,
                "text_cond": cond,
                "uncond": uncond,
                "steps": int(args.steps),
                "cfg_scale": float(args.cfg),
                "shift": float(built.cfg.get("sampling_shift", built.cfg.get("sampling", {}).get("shift", 1.0) if isinstance(built.cfg.get("sampling"), dict) else 1.0)),
                "noise": noise,
                "generator": gen,
                "progress_cb": _progress_cb,
                "start_t": start_t,
                "source_latent": source_latent,
                "mask": mask_latent,
                "task": str(args.task),
            }
            z = sample_flow_euler(**kwargs) if sampler == "flow_euler" else sample_flow_heun(**kwargs)
            samples.append(built.vae.decode(z))

        x = torch.cat(samples, dim=0)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        _save_image_grid(x, out, nrow=int((args.n) ** 0.5))
        event_bus.emit({"type": "status", "status": "done", "path": str(out)})
        print(f"[OK] saved {out}")
        return

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

    per_image_steps = int(args.steps)
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
    _save_image_grid(x, out, nrow=int((args.n) ** 0.5))
    event_bus.emit({"type": "status", "status": "done", "path": str(out)})
    print(f"[OK] saved {out}")


def main() -> None:
    try:
        _main_impl()
    except Exception as exc:
        if is_torch_oom_error(exc):
            print_torch_oom(exc, context="sampling")
            raise SystemExit(2) from None
        raise


if __name__ == "__main__":
    main()
