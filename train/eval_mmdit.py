from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from diffusion.utils import EMA
from diffusion.vae import VAEWrapper
from model.mmdit import MMDiTFlowModel
from model.text.pretrained import FrozenTextEncoderBundle
from samplers import sample_flow_euler, sample_flow_heun
from train.eval import save_image_grid


def _select_flow_sampler(name: str):
    if name == "flow_euler":
        return sample_flow_euler
    if name == "flow_heun":
        return sample_flow_heun
    raise RuntimeError(f"MMDiT eval requires flow_euler or flow_heun, got {name}.")


@torch.no_grad()
def run_mmdit_eval_sampling(
    *,
    step: int,
    model: MMDiTFlowModel,
    ema: EMA,
    vae: VAEWrapper,
    text_encoder: FrozenTextEncoderBundle,
    out_dir: Path,
    prompts: list[str],
    eval_seed: int,
    eval_sampler: str,
    eval_steps: int,
    eval_cfg: float,
    eval_n: int,
    latent_channels: int,
    image_size: int,
    latent_downsample_factor: int,
    shift: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> list[dict]:
    if eval_steps <= 0:
        raise RuntimeError("eval_steps must be positive.")
    if eval_n <= 0:
        raise RuntimeError("eval_n must be positive.")

    sampler = _select_flow_sampler(eval_sampler)
    device = next(model.parameters()).device
    latent_h = int(image_size) // int(latent_downsample_factor)
    latent_w = int(image_size) // int(latent_downsample_factor)
    shape = (1, int(latent_channels), latent_h, latent_w)
    eval_dir = out_dir / "eval" / f"step_{step:07d}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    swapped_ema = False
    try:
        ema.swap_to(model)
        swapped_ema = True
        model.eval()
        uncond = text_encoder([""])

        events: list[dict] = []
        output_records: list[dict] = []
        for prompt_idx, prompt in enumerate(prompts):
            cond = text_encoder([prompt])
            samples = []
            for sample_idx in range(eval_n):
                seed = int(eval_seed) + prompt_idx * eval_n + sample_idx
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                noise = torch.randn(shape, device=device, generator=gen)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    z = sampler(
                        model=model,
                        shape=shape,
                        text_cond=cond,
                        uncond=uncond,
                        steps=int(eval_steps),
                        cfg_scale=float(eval_cfg),
                        shift=float(shift),
                        noise=noise,
                        generator=gen,
                    )
                    x = vae.decode(z)
                samples.append(x.cpu())
            path = eval_dir / f"prompt_{prompt_idx:02d}.png"
            save_image_grid(torch.cat(samples, dim=0), path, nrow=eval_n)
            record = {
                "prompt_index": int(prompt_idx),
                "prompt": str(prompt),
                "sampler": str(eval_sampler),
                "steps": int(eval_steps),
                "cfg": float(eval_cfg),
                "seed": int(eval_seed) + prompt_idx * eval_n,
                "path": str(path),
            }
            output_records.append(record)
            events.append(
                {
                    "type": "sample",
                    "step": int(step),
                    "prompt_set": "eval_prompts_file",
                    "sampler": str(eval_sampler),
                    "steps": int(eval_steps),
                    "cfg": float(eval_cfg),
                    "seed": int(eval_seed) + prompt_idx * eval_n,
                    "path": str(path.relative_to(out_dir)),
                }
            )
        metadata = {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "step": int(step),
            "prompt_set": "eval_prompts_file",
            "sampler": str(eval_sampler),
            "steps": int(eval_steps),
            "cfg": float(eval_cfg),
            "seed": int(eval_seed),
            "shift": float(shift),
            "outputs": output_records,
        }
        (eval_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return events
    finally:
        if swapped_ema:
            ema.restore(model)
        model.train(was_training)

