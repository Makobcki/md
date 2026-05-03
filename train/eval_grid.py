from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from model.text.conditioning import TextConditioning
from sample.build import build_all
from samplers import sample_flow_euler, sample_flow_heun
from train.eval import load_eval_prompt_bank, save_image_grid


DEFAULT_CFG_SWEEP: tuple[float, ...] = (1.0, 2.5, 4.5, 7.0)
DEFAULT_STEP_SWEEP: tuple[int, ...] = (8, 16, 28, 40)
DEFAULT_SAMPLER_SWEEP: tuple[str, ...] = ("flow_euler", "flow_heun")
DEFAULT_SHIFT_SWEEP: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)


def _select_sampler(name: str):
    if name == "flow_euler":
        return sample_flow_euler
    if name == "flow_heun":
        return sample_flow_heun
    raise RuntimeError(f"Unsupported eval sampler: {name}")


def _sanitize_float(value: float) -> str:
    return (f"{float(value):.2f}").replace("-", "m").replace(".", "_")


def _variant_dir_name(
    *,
    sampler: str,
    steps: int,
    cfg: float,
    shift: float,
    base_sampler: str,
    base_steps: int,
    base_cfg: float,
    base_shift: float,
    sweep: bool,
) -> str:
    if not sweep:
        return ""
    parts: list[str] = []
    if sampler != base_sampler:
        parts.append(str(sampler))
    if int(steps) != int(base_steps):
        parts.append(f"steps_{int(steps):03d}")
    if float(cfg) != float(base_cfg):
        parts.append(f"cfg_{_sanitize_float(cfg)}")
    if float(shift) != float(base_shift):
        parts.append(f"shift_{_sanitize_float(shift)}")
    return "_".join(parts) or "base"


def _conditioning_for_prompt(built: Any, prompt: str) -> TextConditioning:
    return built.text_encoder([prompt])


def _empty_conditioning(built: Any) -> TextConditioning:
    if getattr(built, "empty_text", None) is not None:
        return built.empty_text
    return built.text_encoder([""])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class EvalGridResult:
    metadata: dict[str, Any]
    events: list[dict[str, Any]]


@torch.no_grad()
def run_fixed_seed_eval_grids(
    *,
    ckpt: str | Path,
    out_dir: str | Path,
    prompt_root: str | Path = "data/eval_prompts",
    prompt_sets: list[str] | None = None,
    count_per_set: int | None = None,
    seed: int = 42,
    sampler: str = "flow_heun",
    steps: int = 28,
    cfg: float = 4.5,
    n_per_prompt: int = 1,
    shift: float | None = None,
    device: str = "cuda",
    fake_vae: bool = False,
    latent_only: bool = False,
    cfg_values: list[float] | None = None,
    step_values: list[int] | None = None,
    sampler_values: list[str] | None = None,
    shift_values: list[float] | None = None,
    resolution: int | None = None,
    use_ema: bool = True,
) -> EvalGridResult:
    if int(steps) <= 0:
        raise RuntimeError("eval steps must be positive")
    if int(n_per_prompt) <= 0:
        raise RuntimeError("n_per_prompt must be positive")
    if float(cfg) < 0:
        raise RuntimeError("cfg must be non-negative")
    prompt_bank = load_eval_prompt_bank(prompt_sets, root=prompt_root, count_per_set=count_per_set)
    device_obj = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    built = build_all(str(ckpt), device_obj, latent_only=latent_only, fake_vae=fake_vae, use_ema=use_ema)
    if built.vae is None and not latent_only:
        raise RuntimeError("Eval image grids require a VAE, --fake-vae, or --latent-only.")

    ckpt_step = int(getattr(built, "checkpoint_step", 0) or 0)
    if resolution is not None:
        if int(resolution) <= 0:
            raise RuntimeError("eval resolution must be positive")
        if int(resolution) % int(built.cfg.get("latent_downsample_factor", 8)) != 0:
            raise RuntimeError("eval resolution must be divisible by latent_downsample_factor")
        eval_h = eval_w = int(resolution)
        latent_h = latent_w = eval_h // int(built.cfg.get("latent_downsample_factor", 8))
        step_dir = Path(out_dir) / "eval" / f"eval_{eval_h}" / f"step_{ckpt_step:06d}"
    else:
        eval_h = int(built.h)
        eval_w = int(built.w)
        latent_h = int(built.latent_h)
        latent_w = int(built.latent_w)
        step_dir = Path(out_dir) / "eval" / f"step_{ckpt_step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    cfg_sweep = list(cfg_values) if cfg_values is not None else [float(cfg)]
    step_sweep = list(step_values) if step_values is not None else [int(steps)]
    sampler_sweep = list(sampler_values) if sampler_values is not None else [str(sampler)]
    base_shift = float(shift if shift is not None else built.cfg.get("sampling_shift", 1.0))
    shift_sweep = list(shift_values) if shift_values is not None else [base_shift]
    is_sweep = len(cfg_sweep) > 1 or len(step_sweep) > 1 or len(sampler_sweep) > 1 or len(shift_sweep) > 1

    latent_shape = (1, int(built.image_channels), int(latent_h), int(latent_w))
    uncond = _empty_conditioning(built)
    all_runs: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    was_training = bool(getattr(built.model, "training", False))
    built.model.eval()
    try:
        for sampler_name in sampler_sweep:
            sampler_fn = _select_sampler(str(sampler_name))
            for steps_value in step_sweep:
                if int(steps_value) <= 0:
                    raise RuntimeError("step sweep values must be positive")
                for cfg_value in cfg_sweep:
                    if float(cfg_value) < 0:
                        raise RuntimeError("cfg sweep values must be non-negative")
                    for shift_value in shift_sweep:
                        effective_shift = float(shift_value)
                        if effective_shift <= 0:
                            raise RuntimeError("shift sweep values must be positive")
                        variant = _variant_dir_name(
                            sampler=str(sampler_name),
                            steps=int(steps_value),
                            cfg=float(cfg_value),
                            shift=effective_shift,
                            base_sampler=str(sampler),
                            base_steps=int(steps),
                            base_cfg=float(cfg),
                            base_shift=base_shift,
                            sweep=is_sweep,
                        )
                        variant_dir = step_dir / variant if variant else step_dir
                        variant_dir.mkdir(parents=True, exist_ok=True)
                        variant_records: list[dict[str, Any]] = []

                        for set_name, prompts in prompt_bank.items():
                            decoded: list[torch.Tensor] = []
                            latent_outputs: list[torch.Tensor] = []
                            seed_records: list[int] = []
                            for prompt_idx, prompt in enumerate(prompts):
                                cond = _conditioning_for_prompt(built, prompt)
                                for sample_idx in range(int(n_per_prompt)):
                                    sample_seed = int(seed) + len(seed_records)
                                    seed_records.append(sample_seed)
                                    gen = torch.Generator(device=device_obj)
                                    gen.manual_seed(sample_seed)
                                    noise = torch.randn(latent_shape, device=device_obj, generator=gen)
                                    z = sampler_fn(
                                        model=built.model,
                                        shape=latent_shape,
                                        text_cond=cond,
                                        uncond=uncond,
                                        steps=int(steps_value),
                                        cfg_scale=float(cfg_value),
                                        shift=effective_shift,
                                        noise=noise,
                                        generator=gen,
                                    )
                                    latent_outputs.append(z.detach().cpu())
                                    if not latent_only:
                                        assert built.vae is not None
                                        decoded.append(built.vae.decode(z).detach().cpu())

                            if latent_only:
                                grid_path = variant_dir / f"{set_name}_grid.pt"
                                torch.save(torch.cat(latent_outputs, dim=0), grid_path)
                            else:
                                grid_path = variant_dir / f"{set_name}_grid.png"
                                save_image_grid(torch.cat(decoded, dim=0), grid_path, nrow=max(1, int(n_per_prompt)))

                            record = {
                                "prompt_set": str(set_name),
                                "prompts": list(prompts),
                                "seeds": seed_records,
                                "sampler": str(sampler_name),
                                "steps": int(steps_value),
                                "cfg": float(cfg_value),
                                "shift": effective_shift,
                                "path": str(grid_path),
                                "relative_path": str(grid_path.relative_to(Path(out_dir))),
                                "latent_only": bool(latent_only),
                            }
                            variant_records.append(record)
                            all_events.append(
                                {
                                    "type": "sample",
                                    "step": ckpt_step,
                                    "prompt_set": str(set_name),
                                    "sampler": str(sampler_name),
                                    "steps": int(steps_value),
                                    "cfg": float(cfg_value),
                                    "shift": effective_shift,
                                    "seed": int(seed),
                                    "path": str(grid_path.relative_to(Path(out_dir))),
                                    "resolution": [eval_h, eval_w],
                                }
                            )

                        variant_metadata = {
                            "version": 1,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "checkpoint_path": str(ckpt),
                            "checkpoint_step": ckpt_step,
                            "architecture": str(getattr(built, "checkpoint_metadata", {}).get("architecture", built.cfg.get("architecture", "mmdit_rf"))),
                            "objective": str(getattr(built, "checkpoint_metadata", {}).get("objective", built.cfg.get("objective", "rectified_flow"))),
                            "prediction_type": str(getattr(built, "checkpoint_metadata", {}).get("prediction_type", built.cfg.get("prediction_type", "flow_velocity"))),
                            "use_ema": bool(use_ema),
                            "prompt_root": str(prompt_root),
                            "sampler": str(sampler_name),
                            "steps": int(steps_value),
                            "cfg": float(cfg_value),
                            "seed": int(seed),
                            "n_per_prompt": int(n_per_prompt),
                            "shift": effective_shift,
                            "image_size": [eval_h, eval_w],
                            "resolution": [eval_h, eval_w],
                            "latent_shape": [int(built.image_channels), int(latent_h), int(latent_w)],
                            "outputs": variant_records,
                        }
                        _write_json(variant_dir / "metadata.json", variant_metadata)
                        all_runs.append(variant_metadata)
    finally:
        built.model.train(was_training)

    base_meta = {"sampler": str(sampler), "steps": int(steps), "cfg": float(cfg), "seed": int(seed), "shift": base_shift}
    if resolution is not None:
        base_meta["resolution"] = [eval_h, eval_w]
    parent_metadata = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(ckpt),
        "checkpoint_step": ckpt_step,
        "use_ema": bool(use_ema),
        "prompt_root": str(prompt_root),
        "prompt_sets": list(prompt_bank.keys()),
        "base": base_meta,
        "sweep": {
            "samplers": [str(x) for x in sampler_sweep],
            "steps": [int(x) for x in step_sweep],
            "cfg": [float(x) for x in cfg_sweep],
            "shift": [float(x) for x in shift_sweep],
        },
        "runs": all_runs,
    }
    if resolution is not None:
        parent_metadata["resolution"] = [eval_h, eval_w]
    _write_json(step_dir / "metadata.json", parent_metadata)
    _append_jsonl(step_dir / "events.jsonl", all_events)
    return EvalGridResult(metadata=parent_metadata, events=all_events)
