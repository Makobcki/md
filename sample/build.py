from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from config.train import TrainConfig
from diffusion.utils import EMA, load_ckpt
from diffusion.vae import VAEWrapper
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.pretrained import FrozenTextEncoderBundle
from model.text.conditioning import TextConditioning
from model.text.cache import TextCache
from train.checkpoint_mmdit import validate_mmdit_checkpoint_compatibility


class FakeVAE(torch.nn.Module):
    """Small deterministic VAE replacement for smoke tests and fake sampling.

    ``encode`` maps image tensors to the configured latent shape by adaptive
    average pooling and channel pad/truncate. ``decode`` maps latents to a
    visible RGB tensor in [0, 1]. It is not a training component.
    """

    def __init__(self, *, latent_channels: int, latent_h: int, latent_w: int) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.adaptive_avg_pool2d(x, (self.latent_h, self.latent_w))
        if z.shape[1] > self.latent_channels:
            z = z[:, : self.latent_channels]
        elif z.shape[1] < self.latent_channels:
            pad = torch.zeros(
                z.shape[0],
                self.latent_channels - z.shape[1],
                z.shape[2],
                z.shape[3],
                device=z.device,
                dtype=z.dtype,
            )
            z = torch.cat([z, pad], dim=1)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, :3] if z.shape[1] >= 3 else z.repeat(1, 3, 1, 1)[:, :3]
        return torch.sigmoid(x.float())


@dataclass(frozen=True)
class Built:
    ckpt: Dict[str, Any]
    cfg: Dict[str, Any]
    model: torch.nn.Module
    text_encoder: FrozenTextEncoderBundle
    empty_text: TextConditioning | None
    empty_text_source: str
    mode: str
    image_channels: int
    h: int
    w: int
    latent_h: int
    latent_w: int
    vae: torch.nn.Module | VAEWrapper | None
    checkpoint_step: int
    checkpoint_metadata: dict[str, Any]


def _metadata_from_ckpt(ck: Dict[str, Any]) -> dict[str, Any]:
    meta = ck.get("metadata")
    return meta if isinstance(meta, dict) else {}


def load_checkpoint_and_cfg(ckpt_path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ck = load_ckpt(ckpt_path, device)
    cfg = TrainConfig.from_dict(ck["cfg"]).to_dict()
    validate_mmdit_checkpoint_compatibility(ck, cfg)
    return ck, cfg


def build_vae(
    cfg: Dict[str, Any],
    device: torch.device,
    *,
    latent_h: int,
    latent_w: int,
    latent_only: bool = False,
    fake_vae: bool = False,
) -> torch.nn.Module | VAEWrapper | None:
    vae_section = cfg.get("vae", {}) if isinstance(cfg.get("vae", {}), dict) else {}
    backend = str(vae_section.get("backend", cfg.get("vae_backend", ""))).lower()
    if latent_only:
        return None
    if fake_vae or backend == "fake" or str(cfg.get("vae_pretrained", vae_section.get("pretrained", ""))).lower() == "fake":
        return FakeVAE(
            latent_channels=int(cfg.get("latent_channels", 4)),
            latent_h=int(latent_h),
            latent_w=int(latent_w),
        ).to(device)

    vae_pretrained = str(cfg.get("vae_pretrained", vae_section.get("pretrained", "")))
    if not vae_pretrained:
        raise RuntimeError("MMDiT RF image sampling requires vae_pretrained, --fake-vae, or --latent-only.")

    amp_dtype = str(cfg.get("amp_dtype", "")).lower()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return VAEWrapper(
        pretrained=vae_pretrained,
        freeze=True,
        scaling_factor=float(cfg.get("vae_scaling_factor", vae_section.get("scaling_factor", 0.18215))),
        device=device,
        dtype=dtype,
    )


def _nested_model_int(cfg: Dict[str, Any], key: str, default: int) -> int:
    model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    if key in cfg:
        return int(cfg[key])
    if key in model:
        return int(model[key])
    conditioning = model.get("conditioning_tokens", {}) if isinstance(model.get("conditioning_tokens", {}), dict) else {}
    if key in conditioning:
        return int(conditioning[key])
    hierarchical = model.get("hierarchical", {}) if isinstance(model.get("hierarchical", {}), dict) else {}
    if key == "coarse_patch_size" and "coarse_patch_size" in hierarchical:
        return int(hierarchical["coarse_patch_size"])
    return int(default)


def _nested_model_bool(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    if key in cfg:
        return bool(cfg[key])
    if key in model:
        return bool(model[key])
    hierarchical = model.get("hierarchical", {}) if isinstance(model.get("hierarchical", {}), dict) else {}
    if key == "hierarchical_tokens_enabled" and "enabled" in hierarchical:
        return bool(hierarchical["enabled"])
    return bool(default)


def resolve_shapes(cfg: Dict[str, Any], *, width: int | None = None, height: int | None = None) -> Tuple[int, int, int, int]:
    h = int(height if height is not None else cfg.get("height", cfg.get("image_size", 512)))
    w = int(width if width is not None else cfg.get("width", cfg.get("image_size", 512)))
    downsample = int(cfg.get("latent_downsample_factor", 8))
    if h % downsample != 0 or w % downsample != 0:
        raise RuntimeError("image_size must be divisible by latent_downsample_factor.")
    latent_h, latent_w = h // downsample, w // downsample
    patch_size = _nested_model_int(cfg, "patch_size", int(cfg.get("latent_patch_size", 2)))
    patch_sizes = {
        int(cfg.get("latent_patch_size", patch_size)),
        patch_size,
        _nested_model_int(cfg, "source_patch_size", patch_size),
        _nested_model_int(cfg, "mask_patch_size", patch_size),
        _nested_model_int(cfg, "control_patch_size", patch_size),
    }
    if _nested_model_bool(cfg, "hierarchical_tokens_enabled", False):
        patch_sizes.add(_nested_model_int(cfg, "coarse_patch_size", 4))
    for size in sorted(patch_sizes):
        if size <= 0:
            raise RuntimeError("latent patch sizes must be positive.")
        if latent_h % size != 0 or latent_w % size != 0:
            raise RuntimeError(
                f"requested latent shape {(latent_h, latent_w)} must be divisible by Stage D patch size {size}."
            )
    return h, w, latent_h, latent_w


def _load_empty_text_from_cache(cfg: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> tuple[TextConditioning | None, str]:
    root = cfg.get("data_root")
    cache_dir = cfg.get("text_cache_dir", ".cache/text")
    if not root or not bool(cfg.get("text_cache", True)):
        return None, "encoder"
    cache = TextCache(Path(str(root)) / str(cache_dir), shard_cache_size=int(cfg.get("text_shard_cache_size", 2)))
    if not cache.empty_prompt_path.exists():
        return None, "encoder"
    empty = cache.load_empty().to(device=device, dtype=dtype)
    return empty, "text_cache/empty_prompt"


def build_all(
    ckpt_path: str,
    device: torch.device,
    *,
    latent_only: bool = False,
    fake_vae: bool = False,
    use_ema: bool = True,
    width: int | None = None,
    height: int | None = None,
) -> Built:
    ck, cfg = load_checkpoint_and_cfg(ckpt_path, device)
    architecture = str(ck.get("architecture", cfg.get("architecture", "")))
    if architecture != "mmdit_rf":
        raise RuntimeError("Only architecture=mmdit_rf checkpoints are supported.")
    if str(cfg.get("mode", "latent")) != "latent":
        raise RuntimeError("MMDiT RF sampling requires latent mode.")

    model = MMDiTFlowModel(MMDiTConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ck["model"], strict=True)
    if use_ema and "ema" in ck and isinstance(ck["ema"], dict):
        ema = EMA(model)
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        ema.copy_to(model)
    model.eval()

    h, w, latent_h, latent_w = resolve_shapes(cfg, width=width, height=height)
    text_dtype = torch.bfloat16 if str(cfg.get("amp_dtype", "bf16")) == "bf16" else torch.float16
    text_encoder = FrozenTextEncoderBundle.from_config(
        cfg,
        device=device,
        dtype=text_dtype,
    )
    empty_text, empty_text_source = _load_empty_text_from_cache(cfg, device, text_dtype)
    return Built(
        ckpt=ck,
        cfg=cfg,
        model=model,
        text_encoder=text_encoder,
        empty_text=empty_text,
        empty_text_source=empty_text_source,
        mode="latent",
        image_channels=int(cfg.get("latent_channels", 4)),
        h=h,
        w=w,
        latent_h=latent_h,
        latent_w=latent_w,
        vae=build_vae(cfg, device, latent_h=latent_h, latent_w=latent_w, latent_only=latent_only, fake_vae=fake_vae),
        checkpoint_step=int(ck.get("step", _metadata_from_ckpt(ck).get("step", 0)) or 0),
        checkpoint_metadata=_metadata_from_ckpt(ck),
    )
