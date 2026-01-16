from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from diffusion.config import TrainConfig
from diffusion.diffusion import Diffusion, DiffusionConfig
from diffusion.model import UNet, UNetConfig
from diffusion.text import BPETokenizer, TextConfig
from diffusion.utils import EMA, load_ckpt
from diffusion.vae import VAEWrapper


@dataclass(frozen=True)
class Built:
    ckpt: Dict[str, Any]
    cfg: Dict[str, Any]
    use_text_conditioning: bool
    self_conditioning: bool
    diffusion: Diffusion
    model: UNet
    tokenizer: Optional[BPETokenizer]
    mode: str
    image_channels: int
    h: int
    w: int
    latent_h: Optional[int]
    latent_w: Optional[int]
    vae: Optional[VAEWrapper]


def load_checkpoint_and_cfg(ckpt_path: str, device: torch.device) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    ck = load_ckpt(ckpt_path, device)
    cfg = TrainConfig.from_dict(ck["cfg"]).to_dict()
    return ck, cfg


def resolve_flags(ck: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, bool]:
    use_text_conditioning = bool(cfg.get("use_text_conditioning", True))
    meta_flag = ck.get("meta", {}).get("use_text_conditioning")
    if isinstance(meta_flag, bool):
        use_text_conditioning = meta_flag

    self_conditioning = bool(cfg.get("self_conditioning", False))
    meta_self = ck.get("meta", {}).get("self_conditioning")
    if isinstance(meta_self, bool):
        self_conditioning = meta_self

    return use_text_conditioning, self_conditioning


def build_diffusion(cfg: Dict[str, Any], device: torch.device) -> Diffusion:
    return Diffusion(
        DiffusionConfig(
            timesteps=int(cfg.get("timesteps", 1000)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 2e-2)),
            prediction_type=str(cfg.get("prediction_type", "v")),
            noise_schedule=str(cfg.get("noise_schedule", "linear")),
            cosine_s=float(cfg.get("cosine_s", 0.008)),
        ),
        device=device,
    )


def build_tokenizer(cfg: Dict[str, Any]) -> BPETokenizer:
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
    return BPETokenizer.from_files(vocab_path, merges_path, text_cfg)


def build_unet_config(
    cfg: Dict[str, Any],
    image_channels: int,
    tokenizer: Optional[BPETokenizer],
    use_text_conditioning: bool,
    self_conditioning: bool,
) -> UNetConfig:
    return UNetConfig(
        image_channels=image_channels,
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 3, 4])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [32, 16])),
        attn_heads=int(cfg.get("attn_heads", 4)),
        attn_head_dim=int(cfg.get("attn_head_dim", 32)),
        vocab_size=len(tokenizer.vocab) if tokenizer is not None else 0,
        text_dim=int(cfg.get("text_dim", 256)),
        text_layers=int(cfg.get("text_layers", 4)),
        text_heads=int(cfg.get("text_heads", 4)),
        text_max_len=int(cfg.get("text_max_len", 64)),
        use_text_conditioning=use_text_conditioning,
        use_scale_shift_norm=bool(cfg.get("use_scale_shift_norm", False)),
        self_conditioning=self_conditioning,
    )


def build_model(
    ck: Dict[str, Any],
    unet_cfg: UNetConfig,
    device: torch.device,
    channels_last: bool,
) -> UNet:
    if channels_last:
        model = UNet(unet_cfg).to(device, memory_format=torch.channels_last)
    else:
        model = UNet(unet_cfg).to(device)

    model.load_state_dict(ck["model"], strict=True)

    if "ema" in ck:
        ema = EMA(model)
        ema.shadow = ck["ema"]
        ema.copy_to(model)

    model.eval()
    return model


def build_vae_if_needed(cfg: Dict[str, Any], device: torch.device) -> VAEWrapper:
    vae_pretrained = str(cfg.get("vae_pretrained", ""))
    if not vae_pretrained:
        raise RuntimeError("latent mode requires vae_pretrained in checkpoint config.")

    amp_dtype = str(cfg.get("amp_dtype", "")).lower()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    return VAEWrapper(
        pretrained=vae_pretrained,
        freeze=True,
        scaling_factor=float(cfg.get("vae_scaling_factor", 0.18215)),
        device=device,
        dtype=dtype,
    )


def resolve_shapes(cfg: Dict[str, Any], mode: str) -> Tuple[int, int, Optional[int], Optional[int]]:
    h = int(cfg.get("image_size", 512))
    w = int(cfg.get("image_size", 512))

    if mode != "latent":
        return h, w, None, None

    downsample = int(cfg.get("latent_downsample_factor", 8))
    if h % downsample != 0 or w % downsample != 0:
        raise RuntimeError("image_size must be divisible by latent_downsample_factor.")
    return h, w, h // downsample, w // downsample


def build_all(ckpt_path: str, device: torch.device) -> Built:
    ck, cfg = load_checkpoint_and_cfg(ckpt_path, device)
    use_text_conditioning, self_conditioning = resolve_flags(ck, cfg)

    diffusion = build_diffusion(cfg, device)

    tokenizer: Optional[BPETokenizer] = None
    if use_text_conditioning:
        tokenizer = build_tokenizer(cfg)

    mode = str(cfg.get("mode", "pixel"))
    image_channels = 3 if mode == "pixel" else int(cfg.get("latent_channels", 4))

    unet_cfg = build_unet_config(
        cfg=cfg,
        image_channels=image_channels,
        tokenizer=tokenizer,
        use_text_conditioning=use_text_conditioning,
        self_conditioning=self_conditioning,
    )

    channels_last = bool(cfg.get("channels_last", True))
    model = build_model(ck, unet_cfg, device, channels_last=channels_last)

    h, w, latent_h, latent_w = resolve_shapes(cfg, mode)

    vae: Optional[VAEWrapper] = None
    if mode == "latent":
        vae = build_vae_if_needed(cfg, device)

    return Built(
        ckpt=ck,
        cfg=cfg,
        use_text_conditioning=use_text_conditioning,
        self_conditioning=self_conditioning,
        diffusion=diffusion,
        model=model,
        tokenizer=tokenizer,
        mode=mode,
        image_channels=image_channels,
        h=h,
        w=w,
        latent_h=latent_h,
        latent_w=latent_w,
        vae=vae,
    )
