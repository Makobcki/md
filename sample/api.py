from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class SampleOptions:
    ckpt: str
    out: str
    n: int = 8
    steps: int = 30
    prompt: str = ""
    neg_prompt: str = ""
    cfg: float = 5.0
    sampler: Literal["flow_euler", "flow_heun"] = "flow_heun"
    seed: int | None = 42
    shift: float | None = None
    device: str = "cuda"
    init_image: str = ""
    strength: float = 1.0
    mask: str = ""
    task: Literal["txt2img", "img2img", "inpaint", "control"] = "txt2img"
    control_image: str = ""
    control_strength: float = 1.0
    control_type: str = "image"
    latent_only: bool = False
    fake_vae: bool = False
    use_ema: bool = True
    width: int | None = None
    height: int | None = None

    def validate(self) -> None:
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.sampler not in {"flow_euler", "flow_heun"}:
            raise ValueError("sampler must be one of: flow_euler, flow_heun")
        if self.task not in {"txt2img", "img2img", "inpaint", "control"}:
            raise ValueError("task must be one of: txt2img, img2img, inpaint, control")
        if not (0.0 <= float(self.strength) <= 1.0):
            raise ValueError("strength must be in [0, 1]")
        if float(self.cfg) < 0:
            raise ValueError("cfg must be non-negative")
        if self.shift is not None and float(self.shift) <= 0.0:
            raise ValueError("shift must be positive")
        if float(self.control_strength) < 0:
            raise ValueError("control_strength must be non-negative")
        if self.width is not None and int(self.width) < 1:
            raise ValueError("width must be >= 1")
        if self.height is not None and int(self.height) < 1:
            raise ValueError("height must be >= 1")
        if self.control_type not in {"none", "latent_identity", "image", "canny", "depth", "pose", "lineart", "normal"}:
            raise ValueError("control_type must be one of: none, latent_identity, image, canny, depth, pose, lineart, normal")
        if self.task in {"img2img", "inpaint"} and not self.init_image:
            raise RuntimeError(f"task={self.task} requires --init-image")
        if self.task == "inpaint" and not self.mask:
            raise RuntimeError("task=inpaint requires --mask")
        if self.task == "control" and not self.control_image:
            raise RuntimeError("task=control requires --control-image")
        if self.latent_only and (self.init_image or self.control_image):
            raise ValueError("init/control images require a VAE or fake_vae; they are not available in latent_only mode")


def _save_image_grid(x, path: str | Path, nrow: int) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().float().clamp(0.0, 1.0)
    try:
        from torchvision.utils import save_image

        save_image(x, path, nrow=max(int(nrow), 1))
        return
    except Exception:
        pass

    from PIL import Image
    import numpy as np

    b, c, h, w = x.shape
    if c == 1:
        x = x.repeat(1, 3, 1, 1)
    elif c > 3:
        x = x[:, :3]
    nrow = max(min(int(nrow), b), 1)
    ncol = int(math.ceil(b / nrow))
    grid = torch.zeros(3, ncol * h, nrow * w, dtype=x.dtype)
    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = x[idx]
    arr = (grid.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def _metadata_sidecar_path(image_path: str | Path) -> Path:
    return Path(image_path).with_suffix(".json")


def _write_sample_metadata(path: str | Path, metadata: dict[str, object]) -> Path:
    sidecar = _metadata_sidecar_path(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return sidecar


def _cfg_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    return value if isinstance(value, dict) else {}


def _model_config_for_metadata(cfg: dict[str, Any], built: Any) -> dict[str, object]:
    meta_model = getattr(built, "checkpoint_metadata", {}) or {}
    if isinstance(meta_model, dict) and isinstance(meta_model.get("model_config"), dict):
        return dict(meta_model["model_config"])
    model_section = _cfg_section(cfg, "model")
    return {
        "latent_channels": int(cfg.get("latent_channels", 4)),
        "latent_patch_size": int(cfg.get("latent_patch_size", model_section.get("patch_size", 2))),
        "hidden_dim": int(cfg.get("hidden_dim", model_section.get("hidden_dim", 1024))),
        "depth": int(cfg.get("depth", model_section.get("depth", 24))),
        "num_heads": int(cfg.get("num_heads", model_section.get("num_heads", 16))),
        "double_stream_blocks": int(cfg.get("double_stream_blocks", model_section.get("double_stream_blocks", 16))),
        "single_stream_blocks": int(cfg.get("single_stream_blocks", model_section.get("single_stream_blocks", 8))),
        "pos_embed": str(cfg.get("pos_embed", model_section.get("pos_embed", "rope_2d"))),
        "rope_scaling": str(cfg.get("rope_scaling", model_section.get("rope_scaling", _cfg_section(model_section, "rope").get("scaling", "none")))),
        "rope_base_grid_hw": list(cfg.get("rope_base_grid_hw", model_section.get("rope_base_grid_hw", _cfg_section(model_section, "rope").get("base_grid", [32, 32])))),
        "rope_theta": float(cfg.get("rope_theta", model_section.get("rope_theta", _cfg_section(model_section, "rope").get("theta", 10000.0)))),
        "hierarchical_tokens_enabled": bool(cfg.get("hierarchical_tokens_enabled", model_section.get("hierarchical_tokens_enabled", _cfg_section(model_section, "hierarchical").get("enabled", False)))),
        "coarse_patch_size": int(cfg.get("coarse_patch_size", model_section.get("coarse_patch_size", _cfg_section(model_section, "hierarchical").get("coarse_patch_size", 4)))),
        "text_dim": int(cfg.get("text_dim", _cfg_section(cfg, "text").get("text_dim", 1024))),
        "pooled_dim": int(cfg.get("pooled_dim", _cfg_section(cfg, "text").get("pooled_dim", 1024))),
    }


def _vae_config_for_metadata(cfg: dict[str, Any], built: Any, *, latent_only: bool, fake_vae: bool) -> dict[str, object]:
    meta = getattr(built, "checkpoint_metadata", {}) or {}
    if isinstance(meta, dict) and isinstance(meta.get("vae_config"), dict):
        out = dict(meta["vae_config"])
    else:
        vae = _cfg_section(cfg, "vae")
        out = {
            "pretrained": str(cfg.get("vae_pretrained", vae.get("pretrained", ""))),
            "scaling_factor": float(cfg.get("vae_scaling_factor", vae.get("scaling_factor", 0.18215))),
        }
    if latent_only:
        out["runtime_backend"] = "latent_only"
    elif fake_vae:
        out["runtime_backend"] = "fake"
    return out


def _text_config_for_metadata(cfg: dict[str, Any], built: Any) -> dict[str, object]:
    encoder = getattr(built, "text_encoder", None)
    if encoder is not None and hasattr(encoder, "metadata"):
        return dict(encoder.metadata())
    meta = getattr(built, "checkpoint_metadata", {}) or {}
    if isinstance(meta, dict) and isinstance(meta.get("text_config"), dict):
        return dict(meta["text_config"])
    text = _cfg_section(cfg, "text")
    return {
        "backend": str(text.get("backend", "real")),
        "encoders": list(text.get("encoders", [])),
        "text_dim": int(cfg.get("text_dim", text.get("text_dim", 1024))),
        "pooled_dim": int(cfg.get("pooled_dim", text.get("pooled_dim", 1024))),
    }


def _get_opt(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _sample_metadata(
    args: Any,
    built,
    *,
    sampler: str,
    seed: int,
    latent_only: bool | None = None,
    fake_vae: bool | None = None,
) -> dict[str, object]:
    cfg = getattr(built, "cfg", {}) or {}
    latent_h = int(getattr(built, "latent_h", None) or int(cfg.get("image_size", 512)) // int(cfg.get("latent_downsample_factor", 8)))
    latent_w = int(getattr(built, "latent_w", None) or int(cfg.get("image_size", 512)) // int(cfg.get("latent_downsample_factor", 8)))
    latent_channels = int(getattr(built, "image_channels", cfg.get("latent_channels", 4)))
    h = int(getattr(built, "h", cfg.get("image_size", 512)))
    w = int(getattr(built, "w", cfg.get("image_size", 512)))
    meta = getattr(built, "checkpoint_metadata", {}) or {}
    checkpoint_step = int(getattr(built, "checkpoint_step", meta.get("step", 0) if isinstance(meta, dict) else 0) or 0)
    latent_only = bool(_get_opt(args, "latent_only", False) if latent_only is None else latent_only)
    fake_vae = bool(_get_opt(args, "fake_vae", False) if fake_vae is None else fake_vae)
    negative_prompt = str(_get_opt(args, "neg_prompt", ""))
    sampling_shift = float(
        _get_opt(args, "shift", None)
        if _get_opt(args, "shift", None) is not None
        else cfg.get("sampling_shift", _cfg_section(cfg, "sampling").get("shift", 1.0))
    )
    model_config = _model_config_for_metadata(cfg, built)
    vae_config = _vae_config_for_metadata(cfg, built, latent_only=latent_only, fake_vae=fake_vae)
    text_config = _text_config_for_metadata(cfg, built)
    return {
        "checkpoint_path": str(_get_opt(args, "ckpt", "")),
        "checkpoint_step": checkpoint_step,
        "ckpt": str(_get_opt(args, "ckpt", "")),
        "architecture": str(meta.get("architecture", cfg.get("architecture", "mmdit_rf"))) if isinstance(meta, dict) else "mmdit_rf",
        "objective": str(meta.get("objective", cfg.get("objective", "rectified_flow"))) if isinstance(meta, dict) else "rectified_flow",
        "prediction_type": str(meta.get("prediction_type", cfg.get("prediction_type", "flow_velocity"))) if isinstance(meta, dict) else "flow_velocity",
        "prompt": str(_get_opt(args, "prompt", "")),
        "negative_prompt": negative_prompt,
        "negative_prompt_source": str(getattr(built, "empty_text_source", "encoder")) if not negative_prompt else "encoder",
        "sampler": str(sampler),
        "steps": int(_get_opt(args, "steps", 0)),
        "cfg": float(_get_opt(args, "cfg", 0.0)),
        "seed": int(seed),
        "sampling_shift": sampling_shift,
        "image_size": [h, w],
        "latent_shape": [latent_channels, latent_h, latent_w],
        "model_config": model_config,
        "vae_config": vae_config,
        "text_encoder_config": text_config,
        "n": int(_get_opt(args, "n", 1)),
        "task": str(_get_opt(args, "task", "txt2img")),
        "strength": float(_get_opt(args, "strength", 1.0)),
        "init_image": str(_get_opt(args, "init_image", "")),
        "mask": str(_get_opt(args, "mask", "")),
        "control_image": str(_get_opt(args, "control_image", "")),
        "control_strength": float(_get_opt(args, "control_strength", 1.0)),
        "control_type": str(_get_opt(args, "control_type", "image")),
        "control_preprocessing": str(_get_opt(args, "control_type", "image")) if str(_get_opt(args, "control_image", "")) else "none",
        "latent_only": latent_only,
        "use_ema": bool(_get_opt(args, "use_ema", True)),
    }


def _load_latent_mask(path: str, *, latent_h: int, latent_w: int, device):
    from PIL import Image
    import numpy as np
    import torch

    with Image.open(path) as im:
        im = im.convert("L").resize((latent_w, latent_h))
        arr = np.asarray(im, dtype="float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def run_sample(options: SampleOptions, event_callback: Callable[[dict[str, object]], None] | None = None, quiet: bool = False) -> dict[str, object]:
    options.validate()
    import torch
    import numpy as np
    from PIL import Image
    from diffusion.events import EventBus, StdoutJsonSink
    from diffusion.perf import PerfConfig, configure_performance
    from samplers import sample_flow_euler, sample_flow_heun
    from .build import build_all

    device = torch.device(options.device if options.device == "cpu" or torch.cuda.is_available() else "cpu")
    class _CallbackSink:
        def __init__(self, callback: Callable[[dict[str, object]], None]) -> None:
            self.callback = callback

        def emit(self, event: dict[str, object]) -> None:
            self.callback(event)

    sinks = [] if quiet else [StdoutJsonSink()]
    if event_callback is not None:
        sinks.append(_CallbackSink(event_callback))
    event_bus = EventBus(sinks)
    base_seed = random.SystemRandom().randint(0, 2**31 - 1) if options.seed is None else int(options.seed)
    seeds = [base_seed + i for i in range(options.n)]
    event_bus.emit({"type": "status", "status": "start", "seed": base_seed, "n": options.n, "task": options.task})

    built = build_all(
        options.ckpt,
        device,
        latent_only=bool(options.latent_only),
        fake_vae=bool(options.fake_vae),
        use_ema=bool(options.use_ema),
        width=options.width,
        height=options.height,
    )
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

    prompt = options.prompt.strip()
    negative_prompt = options.neg_prompt.strip()
    cond = built.text_encoder([prompt])
    if negative_prompt:
        uncond = built.text_encoder([negative_prompt])
    elif getattr(built, "empty_text", None) is not None:
        uncond = built.empty_text
    else:
        uncond = built.text_encoder([""])

    source_latent = None
    control_latents = None
    mask_latent = None
    start_t = 1.0

    def _load_sample_image_tensor(path: str) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (built.w, built.h):
                im = im.resize((built.w, built.h), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous() * 2.0 - 1.0
    if options.init_image:
        if built.vae is None:
            raise RuntimeError("--init-image requires a VAE or --fake-vae; it is not available in --latent-only mode.")
        img = _load_sample_image_tensor(options.init_image).unsqueeze(0).to(device)
        source_latent = built.vae.encode(img)
        start_t = float(options.strength)
    if options.mask:
        mask_latent = _load_latent_mask(options.mask, latent_h=built.latent_h, latent_w=built.latent_w, device=device)
    if options.control_image:
        if built.vae is None:
            raise RuntimeError("--control-image requires a VAE or --fake-vae; it is not available in --latent-only mode.")
        from control.preprocess import image_control_preprocess

        control_img = _load_sample_image_tensor(options.control_image).unsqueeze(0).to(device)
        control_img = image_control_preprocess(control_img, str(options.control_type))
        control_latents = built.vae.encode(control_img)

    outputs = []
    sampler = str(options.sampler)
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
                    "step": image_index * int(options.steps) + step,
                    "max_steps": int(options.steps) * max(options.n, 1),
                    "sampler": sampler,
                }
            )

        kwargs = {
            "model": built.model,
            "shape": shape,
            "text_cond": cond,
            "uncond": uncond,
            "steps": int(options.steps),
            "cfg_scale": float(options.cfg),
            "shift": float(options.shift if options.shift is not None else built.cfg.get("sampling_shift", 1.0)),
            "noise": noise,
            "generator": gen,
            "progress_cb": _progress_cb,
            "start_t": start_t,
            "source_latent": source_latent,
            "mask": mask_latent,
            "control_latents": control_latents,
            "control_type": str(options.control_type),
            "strength": float(options.strength) if options.task in {"img2img", "inpaint"} else 1.0,
            "control_strength": float(options.control_strength) if control_latents is not None else 0.0,
            "task": str(options.task),
        }
        z = sample_flow_euler(**kwargs) if sampler == "flow_euler" else sample_flow_heun(**kwargs)
        outputs.append(z if options.latent_only else built.vae.decode(z))

    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if options.latent_only:
        torch.save(torch.cat(outputs, dim=0).cpu(), out)
    else:
        x = torch.cat(outputs, dim=0)
        _save_image_grid(x, out, nrow=max(1, int(math.sqrt(options.n))))
    metadata = _sample_metadata(options, built, sampler=sampler, seed=base_seed, latent_only=options.latent_only, fake_vae=options.fake_vae)
    sidecar = _write_sample_metadata(out, metadata)
    event_bus.emit({"type": "status", "status": "done", "path": str(out), "metadata": str(sidecar)})
    if not quiet:
        print(f"[OK] saved {out}")
        print(f"[OK] saved metadata {sidecar}")
    return {"path": str(out), "metadata_path": str(sidecar), "metadata": metadata, "seed": base_seed}
