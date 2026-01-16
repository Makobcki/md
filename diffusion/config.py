from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


def _as_tuple(value: Iterable[int] | Iterable[float]) -> Tuple:
    return tuple(value)


@dataclass(frozen=True)
class TrainConfig:
    data_root: str = "./data/raw/Danbooru"
    image_dir: str = "image_512"
    meta_dir: str = "meta"
    tags_dir: str = "tags"
    caption_field: str = "caption_llava_34b_no_tags_short"
    images_only: bool = False
    min_tag_count: int = 8
    require_512: bool = True
    val_ratio: float = 0.01
    cache_dir: str = ".cache"
    failed_list: str = "failed/md5.txt"

    seed: int = 42
    out_dir: str = "./runs/danbooru_512"

    batch_size: int = 1
    grad_accum_steps: int = 8
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1_000
    min_lr_ratio: float = 0.1
    decay_steps: int = 0
    max_steps: int = 120_000
    log_every: int = 50
    save_every: int = 2_000

    timesteps: int = 1_000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    min_snr_gamma: float = 5.0
    prediction_type: str = "v"

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    dropout: float = 0.10
    attn_resolutions: Tuple[int, ...] = (32, 16)
    attn_heads: int = 4
    attn_head_dim: int = 32
    text_dim: int = 256
    text_layers: int = 4
    text_heads: int = 4
    text_max_len: int = 128
    use_text_conditioning: bool = True
    use_scale_shift_norm: bool = True
    grad_checkpointing: bool = False

    text_vocab_path: str = "diffusion/bpe/vocab.json"
    text_merges_path: str = "diffusion/bpe/merges.txt"
    tokenizer_type: str = "bpe"

    cond_drop_prob: float = 0.15
    token_drop_prob: float = 0.0
    tag_drop_prob: float = 0.0
    caption_drop_prob: float = 0.0

    amp: bool = True
    amp_dtype: str = "fp16"
    compile: bool = False
    compile_warmup_steps: int = 2
    compile_cudagraphs: bool = True
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.999
    ema_decay_fast: float = 0.999
    ema_decay_slow: float = 0.9999
    ema_switch_step: int = 10_000
    resume_ckpt: str = ""
    deterministic: bool = False

    tf32: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = True
    enable_flash_sdp: bool = True
    enable_mem_efficient_sdp: bool = True
    enable_math_sdp: bool = False

    mode: str = "pixel"
    latent_channels: int = 4
    latent_downsample_factor: int = 8
    latent_cache: bool = False
    latent_cache_dir: str = ".cache/latents"
    latent_cache_sharded: bool = False
    latent_cache_index: str = "index.jsonl"
    latent_dtype: str = "bf16"
    latent_precompute: bool = False
    latent_cache_fallback: bool = False
    latent_cache_strict: bool = True
    latent_shard_cache_size: int = 2
    vae_pretrained: str = ""
    vae_freeze: bool = True
    vae_scaling_factor: float = 0.18215

    ckpt_keep_last: int = 5

    sanity_overfit_steps: int = 0
    sanity_overfit_images: int = 0
    sanity_overfit_max_loss: float = 0.1

    curriculum_enabled: bool = True
    curriculum_steps: int = 20_000
    curriculum_require_one_person: bool = True
    curriculum_prefer_solo: bool = True
    curriculum_exclude_multi: bool = True
    curriculum_solo_weight: float = 2.0
    curriculum_non_solo_weight: float = 1.0

    self_conditioning: bool = True
    self_cond_prob: float = 0.5

    noise_schedule: str = "linear"
    cosine_s: float = 0.008

    eval_prompts_file: str = "./data/raw/Danbooru/prompts.txt"
    eval_every: int = 500
    eval_seed: int = 42
    eval_sampler: str = "ddim"
    eval_steps: int = 30
    eval_cfg: float = 5.0
    eval_n: int = 1

    image_size: int = 512

    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.prediction_type != "v":
            raise ValueError("Only v-prediction is supported.")
        if self.text_max_len <= 0:
            raise ValueError("text_max_len must be positive.")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.min_lr_ratio <= 0 or self.min_lr_ratio > 1:
            raise ValueError("min_lr_ratio must be in (0, 1].")
        if self.decay_steps < 0:
            raise ValueError("decay_steps must be non-negative.")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive.")
        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError("amp_dtype must be 'fp16' or 'bf16'.")
        if self.tokenizer_type != "bpe":
            raise ValueError("tokenizer_type must be 'bpe'.")
        if self.lr_scheduler not in {"cosine", "linear"}:
            raise ValueError("lr_scheduler must be 'cosine' or 'linear'.")
        if self.mode not in {"pixel", "latent"}:
            raise ValueError("mode must be 'pixel' or 'latent'.")
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")
        if self.latent_downsample_factor <= 0:
            raise ValueError("latent_downsample_factor must be positive.")
        if self.latent_dtype not in {"fp16", "bf16"}:
            raise ValueError("latent_dtype must be 'fp16' or 'bf16'.")
        if self.latent_cache_sharded and not self.latent_cache:
            raise ValueError("latent_cache_sharded requires latent_cache=true.")
        if self.latent_shard_cache_size <= 0:
            raise ValueError("latent_shard_cache_size must be positive.")
        if self.ckpt_keep_last < 0:
            raise ValueError("ckpt_keep_last must be non-negative.")
        if self.curriculum_steps < 0:
            raise ValueError("curriculum_steps must be non-negative.")
        if self.curriculum_solo_weight <= 0 or self.curriculum_non_solo_weight <= 0:
            raise ValueError("curriculum_solo_weight and curriculum_non_solo_weight must be positive.")
        if self.self_cond_prob < 0 or self.self_cond_prob > 1:
            raise ValueError("self_cond_prob must be in [0, 1].")
        if self.noise_schedule not in {"linear", "cosine"}:
            raise ValueError("noise_schedule must be 'linear' or 'cosine'.")
        if self.cosine_s <= 0:
            raise ValueError("cosine_s must be positive.")
        if self.eval_every < 0:
            raise ValueError("eval_every must be non-negative.")
        if self.eval_steps < 0:
            raise ValueError("eval_steps must be non-negative.")
        if self.eval_cfg < 0:
            raise ValueError("eval_cfg must be non-negative.")
        if self.eval_n <= 0:
            raise ValueError("eval_n must be positive.")
        if self.eval_sampler not in {"ddim", "diffusion", "euler", "heun", "dpm_solver"}:
            raise ValueError("eval_sampler must be one of: ddim, diffusion, euler, heun, dpm_solver.")
        if not (0.0 <= self.cond_drop_prob <= 1.0):
            raise ValueError("cond_drop_prob must be in [0, 1].")
        if not (0.0 <= self.token_drop_prob <= 1.0):
            raise ValueError("token_drop_prob must be in [0, 1].")
        if not (0.0 <= self.tag_drop_prob <= 1.0):
            raise ValueError("tag_drop_prob must be in [0, 1].")
        if not (0.0 <= self.caption_drop_prob <= 1.0):
            raise ValueError("caption_drop_prob must be in [0, 1].")
        if not (0.0 < self.ema_decay_fast <= 1.0):
            raise ValueError("ema_decay_fast must be in (0, 1].")
        if not (0.0 < self.ema_decay_slow <= 1.0):
            raise ValueError("ema_decay_slow must be in (0, 1].")
        if self.ema_switch_step < 0:
            raise ValueError("ema_switch_step must be non-negative.")

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in fields}
        extra = {k: v for k, v in data.items() if k not in fields}
        if "channel_mults" in kwargs:
            kwargs["channel_mults"] = _as_tuple(kwargs["channel_mults"])
        if "attn_resolutions" in kwargs:
            kwargs["attn_resolutions"] = _as_tuple(kwargs["attn_resolutions"])
        cfg = cls(**kwargs)
        return replace(cfg, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        return data
