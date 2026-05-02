from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VAEConfig:
    mode: str
    latent_channels: int
    latent_downsample_factor: int
    latent_cache: bool
    latent_cache_dir: str
    latent_cache_sharded: bool
    latent_cache_index: str
    latent_dtype: str
    latent_precompute: bool
    latent_cache_fallback: bool
    latent_cache_strict: bool
    latent_shard_cache_size: int
    vae_pretrained: str
    vae_freeze: bool
    vae_scaling_factor: float
