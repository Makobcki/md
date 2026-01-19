from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from torchvision.utils import save_image

from diffusion.core.diffusion import Diffusion
from diffusion.utils import EMA
from diffusion.vae import VAEWrapper
from model.unet.unet import UNet
from text_enc.tokenizer import BPETokenizer

from samplers import (
    ddpm_ancestral_sample,
    ddim_sample,
    dpm_solver_sample,
    euler_sample,
    heun_sample,
)


def _load_eval_prompts(path: str, count: int) -> list[str]:
    prompts_path = Path(path)
    if not prompts_path.exists():
        raise RuntimeError(f"Eval prompts file not found: {prompts_path}")
    lines = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if len(prompts) < count:
        raise RuntimeError(f"Eval prompts file must contain at least {count} non-empty lines.")
    return prompts[:count]


def _select_sampler(name: str) -> Callable[..., torch.Tensor]:
    if name == "ddim":
        return ddim_sample
    if name == "diffusion":
        return ddpm_ancestral_sample
    if name == "euler":
        return euler_sample
    if name == "heun":
        return heun_sample
    if name == "dpm_solver":
        return dpm_solver_sample
    raise RuntimeError(f"Unknown sampler: {name}")


@torch.no_grad()
def _run_eval_sampling(
    *,
    step: int,
    model: UNet,
    ema: EMA,
    diffusion: Diffusion,
    tokenizer: Optional[BPETokenizer],
    out_dir: Path,
    prompts: list[str],
    eval_seed: int,
    eval_sampler: str,
    eval_steps: int,
    eval_cfg: float,
    eval_n: int,
    mode: str,
    image_size: int,
    latent_channels: int,
    latent_downsample_factor: int,
    vae: Optional[VAEWrapper],
    use_text_conditioning: bool,
    self_conditioning: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    if eval_steps <= 0:
        raise RuntimeError("eval_steps must be positive.")
    if eval_n <= 0:
        raise RuntimeError("eval_n must be positive.")
    if mode not in {"pixel", "latent"}:
        raise RuntimeError(f"Unsupported mode={mode}")
    if mode == "latent" and vae is None:
        raise RuntimeError("VAE is required for latent eval sampling.")
    if use_text_conditioning and tokenizer is None:
        raise RuntimeError("Tokenizer is required for text-conditioned eval sampling.")

    eval_dir = out_dir / "eval" / f"step_{step:07d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    saved_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    was_training = model.training
    model.eval()

    sampler_fn = _select_sampler(eval_sampler)
    device = diffusion.device

    if mode == "latent":
        latent_h = image_size // latent_downsample_factor
        latent_w = image_size // latent_downsample_factor
        shape = (1, latent_channels, latent_h, latent_w)
    else:
        shape = (1, 3, image_size, image_size)

    for idx, prompt in enumerate(prompts):
        samples = []
        for i in range(eval_n):
            seed = int(eval_seed) + idx * eval_n + i
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            noise = torch.randn(shape, device=device, generator=gen)
            if use_text_conditioning:
                ids, mask = tokenizer.encode(prompt)
                ids = ids.unsqueeze(0).to(device)
                mask = mask.unsqueeze(0).to(device)
                un_ids, un_mask = tokenizer.encode("")
                un_ids = un_ids.unsqueeze(0).to(device)
                un_mask = un_mask.unsqueeze(0).to(device)
            else:
                ids = None
                mask = None
                un_ids = None
                un_mask = None
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                x = sampler_fn(
                    model=model,
                    diffusion=diffusion,
                    shape=shape,
                    txt_ids=ids,
                    txt_mask=mask,
                    steps=eval_steps,
                    cfg_scale=eval_cfg,
                    uncond_ids=un_ids,
                    uncond_mask=un_mask,
                    self_conditioning=self_conditioning,
                    noise=noise,
                    generator=gen,
                )
            if mode == "latent":
                x = vae.decode(x)
            if mode == "pixel":
                x = (x.clamp(-1, 1) + 1) / 2.0
            samples.append(x.cpu())

        batch = torch.cat(samples, dim=0)
        out_path = eval_dir / f"prompt_{idx:02d}.png"
        save_image(batch, out_path, nrow=eval_n)

    model.load_state_dict(saved_state, strict=True)
    model.train(was_training)
