from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from config.train import TrainConfig
from diffusion.utils import EMA, load_ckpt
from diffusion.vae import VAEWrapper
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.pretrained import FrozenTextEncoderBundle


@dataclass(frozen=True)
class Built:
    ckpt: Dict[str, Any]
    cfg: Dict[str, Any]
    model: torch.nn.Module
    text_encoder: FrozenTextEncoderBundle
    mode: str
    image_channels: int
    h: int
    w: int
    latent_h: Optional[int]
    latent_w: Optional[int]
    vae: VAEWrapper


def load_checkpoint_and_cfg(ckpt_path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ck = load_ckpt(ckpt_path, device)
    cfg = TrainConfig.from_dict(ck["cfg"]).to_dict()
    return ck, cfg


def build_vae(cfg: Dict[str, Any], device: torch.device) -> VAEWrapper:
    vae_section = cfg.get("vae", {}) if isinstance(cfg.get("vae", {}), dict) else {}
    vae_pretrained = str(cfg.get("vae_pretrained", vae_section.get("pretrained", "")))
    if not vae_pretrained:
        raise RuntimeError("MMDiT RF sampling requires vae_pretrained in checkpoint config.")

    amp_dtype = str(cfg.get("amp_dtype", "")).lower()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return VAEWrapper(
        pretrained=vae_pretrained,
        freeze=True,
        scaling_factor=float(cfg.get("vae_scaling_factor", vae_section.get("scaling_factor", 0.18215))),
        device=device,
        dtype=dtype,
    )


def resolve_shapes(cfg: Dict[str, Any]) -> Tuple[int, int, int, int]:
    h = int(cfg.get("image_size", 512))
    w = int(cfg.get("image_size", 512))
    downsample = int(cfg.get("latent_downsample_factor", 8))
    if h % downsample != 0 or w % downsample != 0:
        raise RuntimeError("image_size must be divisible by latent_downsample_factor.")
    return h, w, h // downsample, w // downsample


def build_all(ckpt_path: str, device: torch.device) -> Built:
    ck, cfg = load_checkpoint_and_cfg(ckpt_path, device)
    architecture = str(ck.get("architecture", cfg.get("architecture", "")))
    if architecture != "mmdit_rf":
        raise RuntimeError("Only architecture=mmdit_rf checkpoints are supported.")
    if str(cfg.get("mode", "latent")) != "latent":
        raise RuntimeError("MMDiT RF sampling requires latent mode.")

    model = MMDiTFlowModel(MMDiTConfig.from_dict(cfg)).to(device)
    model.load_state_dict(ck["model"], strict=True)
    if "ema" in ck and isinstance(ck["ema"], dict):
        ema = EMA(model)
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}
        ema.copy_to(model)
    model.eval()

    h, w, latent_h, latent_w = resolve_shapes(cfg)
    text_encoder = FrozenTextEncoderBundle(
        cfg,
        device=device,
        dtype=torch.bfloat16 if str(cfg.get("amp_dtype", "bf16")) == "bf16" else torch.float16,
    )
    return Built(
        ckpt=ck,
        cfg=cfg,
        model=model,
        text_encoder=text_encoder,
        mode="latent",
        image_channels=int(cfg.get("latent_channels", 4)),
        h=h,
        w=w,
        latent_h=latent_h,
        latent_w=latent_w,
        vae=build_vae(cfg, device),
    )
