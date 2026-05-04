from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict

TEXT_ENCODER_PRESETS: Dict[str, Dict[str, Any]] = {
    "clip_l_t5_base": {
        "text_dim": 1024,
        "pooled_dim": 1024,
        "encoders": [
            {"name": "clip_l", "model_name": "openai/clip-vit-large-patch14", "max_length": 77, "trainable": False, "cache": True},
            {"name": "t5_base", "model_name": "google/t5-v1_1-base", "max_length": 256, "trainable": False, "cache": True},
        ],
    },
    "clip_l_t5_large": {
        "text_dim": 1024,
        "pooled_dim": 1024,
        "encoders": [
            {"name": "clip_l", "model_name": "openai/clip-vit-large-patch14", "max_length": 77, "trainable": False, "cache": True},
            {"name": "t5_large", "model_name": "google/t5-v1_1-large", "max_length": 256, "trainable": False, "cache": True},
        ],
    },
    "clip_l_clip_bigG_t5_large": {
        "text_dim": 1024,
        "pooled_dim": 1024,
        "encoders": [
            {"name": "clip_l", "model_name": "openai/clip-vit-large-patch14", "max_length": 77, "trainable": False, "cache": True},
            {"name": "clip_bigG", "model_name": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "max_length": 77, "trainable": False, "cache": True},
            {"name": "t5_large", "model_name": "google/t5-v1_1-large", "max_length": 256, "trainable": False, "cache": True},
        ],
    },
}


def _apply_text_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    preset_name = str(data.get("text_preset", data.get("text", {}).get("preset", "")) or "").strip()
    if not preset_name:
        return dict(data)
    if preset_name not in TEXT_ENCODER_PRESETS:
        allowed = ", ".join(sorted(TEXT_ENCODER_PRESETS))
        raise ValueError(f"Unsupported text_preset={preset_name!r}. Allowed: {allowed}.")
    out = dict(data)
    preset = TEXT_ENCODER_PRESETS[preset_name]
    text = dict(out.get("text") or {})
    text.setdefault("preset", preset_name)
    text.setdefault("text_dim", int(preset["text_dim"]))
    text.setdefault("pooled_dim", int(preset["pooled_dim"]))
    if not text.get("encoders"):
        text["encoders"] = [dict(item) for item in preset["encoders"]]
    out["text"] = text
    return out


def _flatten_nested_config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Accept the current nested YAML shape while exposing a flat TrainConfig."""
    data = _apply_text_preset(data)
    flat = dict(data)

    training = data.get("training")
    if isinstance(training, dict):
        for key in (
            "batch_size",
            "grad_accum_steps",
            "lr",
            "weight_decay",
            "optimizer",
            "warmup_steps",
            "max_steps",
            "log_every",
            "save_every",
            "val_every",
            "val_batches",
            "eval_every",
            "eval_steps",
            "eval_cfg",
            "eval_seed",
            "eval_n",
            "ema_decay",
            "grad_clip_norm",
            "amp",
            "amp_dtype",
            "tf32",
            "compile",
        ):
            if key in training and key not in flat:
                flat[key] = training[key]

    model = data.get("model")
    if isinstance(model, dict):
        for key in (
            "hidden_dim",
            "depth",
            "num_heads",
            "mlp_ratio",
            "qk_norm",
            "rms_norm",
            "swiglu",
            "adaln_zero",
            "pos_embed",
            "rope_scaling",
            "rope_base_grid_hw",
            "rope_theta",
            "double_stream_blocks",
            "single_stream_blocks",
            "dropout",
            "attn_dropout",
            "gradient_checkpointing",
            "zero_init_final",
            "attention_schedule",
            "early_joint_blocks",
            "late_joint_blocks",
            "source_patch_size",
            "mask_patch_size",
            "control_patch_size",
            "mask_as_source_channel",
            "conditioning_rope",
            "strength_embed",
            "control_type_embed",
            "control_adapter",
            "control_adapter_ratio",
            "hierarchical_tokens_enabled",
            "coarse_patch_size",
        ):
            if key in model and key not in flat:
                flat[key] = model[key]
        rope = model.get("rope")
        if isinstance(rope, dict):
            mapping = {
                "scaling": "rope_scaling",
                "base_grid": "rope_base_grid_hw",
                "theta": "rope_theta",
            }
            for src, dst in mapping.items():
                if src in rope and dst not in flat:
                    flat[dst] = rope[src]
        hierarchical = model.get("hierarchical")
        if isinstance(hierarchical, dict):
            mapping = {
                "enabled": "hierarchical_tokens_enabled",
                "coarse_patch_size": "coarse_patch_size",
            }
            for src, dst in mapping.items():
                if src in hierarchical and dst not in flat:
                    flat[dst] = hierarchical[src]
    text = data.get("text")
    if isinstance(text, dict):
        for key in ("text_dim", "pooled_dim"):
            if key in text and key not in flat:
                flat[key] = text[key]
        if "cache" in text and "text_cache" not in flat:
            flat["text_cache"] = text["cache"]
        resampler = text.get("resampler")
        if isinstance(resampler, dict):
            mapping = {
                "enabled": "text_resampler_enabled",
                "num_tokens": "text_resampler_num_tokens",
                "depth": "text_resampler_depth",
                "mlp_ratio": "text_resampler_mlp_ratio",
            }
            for src, dst in mapping.items():
                if src in resampler and dst not in flat:
                    flat[dst] = resampler[src]

    dataset = data.get("dataset")
    if isinstance(dataset, dict):
        mapping = {
            "text_field": "text_field",
            "text_fields": "text_fields",
            "caption_field": "caption_field",
            "prompt_field": "text_field",
            "image_dir": "image_dir",
            "meta_dir": "meta_dir",
            "tags_dir": "tags_dir",
            "min_tag_count": "min_tag_count",
            "require_512": "require_512",
            "val_ratio": "val_ratio",
            "dataset_limit": "dataset_limit",
            "aspect_buckets_enabled": "aspect_buckets_enabled",
            "aspect_buckets": "aspect_buckets",
        }
        for src, dst in mapping.items():
            if src in dataset and dst not in flat:
                flat[dst] = dataset[src]

    img2img = data.get("img2img")
    if isinstance(img2img, dict):
        mapping = {
            "strength_min": "img2img_strength_min",
            "strength_max": "img2img_strength_max",
        }
        for src, dst in mapping.items():
            if src in img2img and dst not in flat:
                flat[dst] = img2img[src]

    inpaint = data.get("inpaint")
    if isinstance(inpaint, dict):
        mapping = {
            "mask_min_area": "inpaint_mask_min_area",
            "mask_max_area": "inpaint_mask_max_area",
            "mask_modes": "inpaint_mask_modes",
            "loss_mask_weight": "inpaint_loss_mask_weight",
            "loss_unmask_weight": "inpaint_loss_unmask_weight",
            "strength_min": "inpaint_strength_min",
            "strength_max": "inpaint_strength_max",
        }
        for src, dst in mapping.items():
            if src in inpaint and dst not in flat:
                flat[dst] = inpaint[src]

    conditioning = data.get("conditioning")
    if isinstance(conditioning, dict):
        mapping = {
            "cfg_drop_prob": "cond_drop_prob",
            "token_drop_prob": "token_drop_prob",
            "caption_drop_prob": "caption_drop_prob",
            "tag_drop_prob": "tag_drop_prob",
        }
        for src, dst in mapping.items():
            if src in conditioning and dst not in flat:
                flat[dst] = conditioning[src]

    control = data.get("control")
    if isinstance(control, dict):
        mapping = {
            "enabled": "control_enabled",
            "types": "control_types",
            "strength": "control_strength",
            "num_streams": "control_num_streams",
            "type_embed": "control_type_embed",
            "adapter": "control_adapter",
            "adapter_ratio": "control_adapter_ratio",
        }
        for src, dst in mapping.items():
            if src in control and dst not in flat:
                flat[dst] = control[src]

    hierarchical = data.get("hierarchical")
    if isinstance(hierarchical, dict):
        mapping = {
            "enabled": "hierarchical_tokens_enabled",
            "coarse_patch_size": "coarse_patch_size",
        }
        for src, dst in mapping.items():
            if src in hierarchical and dst not in flat:
                flat[dst] = hierarchical[src]

    loss = data.get("loss")
    if isinstance(loss, dict):
        if "x0_aux_weight" in loss and "x0_aux_weight" not in flat:
            flat["x0_aux_weight"] = loss["x0_aux_weight"]

    flow = data.get("flow")
    if isinstance(flow, dict):
        mapping = {
            "timestep_sampling": "flow_timestep_sampling",
            "logit_mean": "flow_logit_mean",
            "logit_std": "flow_logit_std",
            "loss_weighting": "flow_loss_weighting",
            "timestep_shift": "flow_timestep_shift",
            "shift": "flow_timestep_shift",
            "train_t_min": "flow_train_t_min",
            "train_t_max": "flow_train_t_max",
        }
        for src, dst in mapping.items():
            if src in flow and dst not in flat:
                flat[dst] = flow[src]

    sampling = data.get("sampling")
    if isinstance(sampling, dict):
        mapping = {
            "sampler": "sampling_sampler",
            "steps": "sampling_steps",
            "cfg_scale": "sampling_cfg_scale",
            "shift": "sampling_shift",
        }
        for src, dst in mapping.items():
            if src in sampling and dst not in flat:
                flat[dst] = sampling[src]
        if "sampler" in sampling and "eval_sampler" not in flat:
            flat["eval_sampler"] = sampling["sampler"]
        if "steps" in sampling and "eval_steps" not in flat:
            flat["eval_steps"] = sampling["steps"]
        if "cfg_scale" in sampling and "eval_cfg" not in flat:
            flat["eval_cfg"] = sampling["cfg_scale"]

    vae = data.get("vae")
    if isinstance(vae, dict):
        mapping = {
            "pretrained": "vae_pretrained",
            "freeze": "vae_freeze",
            "scaling_factor": "vae_scaling_factor",
        }
        for src, dst in mapping.items():
            if src in vae and dst not in flat:
                flat[dst] = vae[src]

    cache = data.get("cache")
    if isinstance(cache, dict):
        mapping = {
            "latent_cache": "latent_cache",
            "text_cache": "text_cache",
            "auto_prepare": "cache_auto_prepare",
            "rebuild_if_stale": "cache_rebuild_if_stale",
            "sharded": "latent_cache_sharded",
            "dtype": "latent_dtype",
            "text_shard_cache_size": "text_shard_cache_size",
            "allow_on_the_fly_text": "allow_on_the_fly_text",
            "strict": "cache_strict",
            "validate_on_start": "cache_validate_on_start",
        }
        for src, dst in mapping.items():
            if src in cache and dst not in flat:
                flat[dst] = cache[src]
        if "strict" in cache and "latent_cache_strict" not in flat:
            flat["latent_cache_strict"] = cache["strict"]

    performance = data.get("performance")
    if isinstance(performance, dict):
        mapping = {
            "flash_sdp": "enable_flash_sdp",
            "mem_efficient_sdp": "enable_mem_efficient_sdp",
            "math_sdp": "enable_math_sdp",
            "tf32": "tf32",
            "cudnn_benchmark": "cudnn_benchmark",
            "channels_last": "channels_last",
        }
        for src, dst in mapping.items():
            if src in performance and dst not in flat:
                flat[dst] = performance[src]

    distributed = data.get("distributed")
    if isinstance(distributed, dict):
        mapping = {
            "backend": "distributed_backend",
            "save_on_rank0_only": "save_on_rank0_only",
            "metrics_aggregation": "distributed_metrics_aggregation",
            "find_unused_parameters": "ddp_find_unused_parameters",
            "gradient_as_bucket_view": "ddp_gradient_as_bucket_view",
        }
        for src, dst in mapping.items():
            if src in distributed and dst not in flat:
                flat[dst] = distributed[src]

    fsdp = data.get("fsdp")
    if isinstance(fsdp, dict):
        mapping = {
            "enabled": "fsdp_enabled",
            "min_hidden_dim": "fsdp_min_hidden_dim",
            "min_num_params": "fsdp_min_num_params",
            "sharding_strategy": "fsdp_sharding_strategy",
            "auto_wrap_policy": "fsdp_auto_wrap_policy",
            "cpu_offload": "fsdp_cpu_offload",
        }
        for src, dst in mapping.items():
            if src in fsdp and dst not in flat:
                flat[dst] = fsdp[src]

    debug = data.get("debug")
    if isinstance(debug, dict) and "dataset_limit" in debug and "dataset_limit" not in flat:
        flat["dataset_limit"] = debug["dataset_limit"]

    return flat


@dataclass(frozen=True)
class TrainConfig:
    architecture: str = "mmdit_rf"
    objective: str = "rectified_flow"
    prediction_type: str = "flow_velocity"

    data_root: str = "./data/dataset/pixso_512"
    image_dir: str = "images"
    meta_dir: str = ""
    tags_dir: str = ""
    caption_field: str = "caption_llava_34b_no_tags_short"
    text_field: str = ""
    text_fields: list[str] = field(default_factory=list)
    images_only: bool = False
    min_tag_count: int = 0
    require_512: bool = True
    val_ratio: float = 0.01
    cache_dir: str = ".cache"
    failed_list: str = "failed/md5.txt"
    dataset_limit: int = 0
    aspect_buckets_enabled: bool = False
    aspect_buckets: list[Any] = field(default_factory=list)
    dataset_tasks: Dict[str, float] = field(
        default_factory=lambda: {"txt2img": 1.0, "img2img": 0.0, "inpaint": 0.0, "control": 0.0}
    )
    img2img_strength_min: float = 1.0
    img2img_strength_max: float = 1.0
    inpaint_strength_min: float = 1.0
    inpaint_strength_max: float = 1.0

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
    val_every: int = 500
    val_batches: int = 8
    optimizer: str = "adamw"
    grad_clip_norm: float = 1.0
    fail_on_nonfinite_grad: bool = False
    ema_decay: float = 0.999
    ema_decay_fast: float = 0.999
    ema_decay_slow: float = 0.9999
    ema_switch_step: int = 10_000
    resume_ckpt: str = ""
    deterministic: bool = False

    amp: bool = True
    amp_dtype: str = "fp16"
    compile: bool = False
    compile_warmup_steps: int = 2
    compile_cudagraphs: bool = True
    tf32: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = True
    enable_flash_sdp: bool = True
    enable_mem_efficient_sdp: bool = True
    enable_math_sdp: bool = False

    distributed_backend: str = "none"
    save_on_rank0_only: bool = True
    distributed_metrics_aggregation: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True

    fsdp_enabled: bool = False
    fsdp_min_hidden_dim: int = 1024
    fsdp_min_num_params: int = 500_000_000
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_auto_wrap_policy: str = "transformer_block"
    fsdp_cpu_offload: bool = False

    mode: str = "latent"
    image_size: int = 512
    latent_channels: int = 4
    latent_downsample_factor: int = 8
    latent_patch_size: int = 2
    latent_cache: bool = True
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

    hidden_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    rms_norm: bool = True
    swiglu: bool = True
    adaln_zero: bool = True
    pos_embed: str = "rope_2d"
    rope_scaling: str = "none"
    rope_base_grid_hw: tuple[int, int] = (32, 32)
    rope_theta: float = 10000.0
    double_stream_blocks: int = 16
    single_stream_blocks: int = 8
    dropout: float = 0.0
    attn_dropout: float = 0.0
    gradient_checkpointing: bool = True
    zero_init_final: bool = True
    attention_schedule: str = "full"
    early_joint_blocks: int = 0
    late_joint_blocks: int = 0
    source_patch_size: int = 2
    mask_patch_size: int = 2
    control_patch_size: int = 2
    mask_as_source_channel: bool = False
    conditioning_rope: bool = True
    strength_embed: bool = False
    control_type_embed: bool = False
    control_adapter: bool = False
    control_adapter_ratio: float = 0.25
    hierarchical_tokens_enabled: bool = False
    coarse_patch_size: int = 4

    text_resampler_enabled: bool = False
    text_resampler_num_tokens: int = 128
    text_resampler_depth: int = 2
    text_resampler_mlp_ratio: float = 4.0

    x0_aux_weight: float = 0.0

    text_preset: str = ""
    text_dim: int = 1024
    pooled_dim: int = 1024
    text_cache: bool = True
    allow_on_the_fly_text: bool = False
    text_cache_dir: str = ".cache/text"
    text_shard_cache_size: int = 2
    cache_auto_prepare: bool = True
    cache_rebuild_if_stale: bool = False
    cache_strict: bool = True
    cache_validate_on_start: bool = True

    cond_drop_prob: float = 0.15
    token_drop_prob: float = 0.0
    tag_drop_prob: float = 0.0
    caption_drop_prob: float = 0.0

    inpaint_mask_min_area: float = 0.05
    inpaint_mask_max_area: float = 0.60
    inpaint_mask_modes: Dict[str, float] = field(
        default_factory=lambda: {"rectangle": 0.5, "brush": 0.3, "random_blocks": 0.2}
    )
    inpaint_loss_mask_weight: float = 1.0
    inpaint_loss_unmask_weight: float = 0.1

    control_enabled: bool = False
    control_types: Dict[str, bool] = field(default_factory=lambda: {"canny": True, "depth": False, "pose": False})
    control_strength: float = 1.0
    control_num_streams: int = 1

    flow_timestep_sampling: str = "logit_normal"
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_loss_weighting: str = "none"
    flow_timestep_shift: float = 1.0
    flow_train_t_min: float = 0.0
    flow_train_t_max: float = 1.0

    sampling_sampler: str = "flow_heun"
    sampling_steps: int = 28
    sampling_cfg_scale: float = 4.5
    sampling_shift: float = 3.0

    eval_prompts_file: str = "./data/eval_prompts/core.txt"
    eval_every: int = 500
    eval_seed: int = 42
    eval_sampler: str = "flow_heun"
    eval_steps: int = 30
    eval_cfg: float = 5.0
    eval_n: int = 1

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

    extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.architecture != "mmdit_rf":
            raise ValueError("Only architecture=mmdit_rf is supported.")
        if self.objective != "rectified_flow":
            raise ValueError("Only objective=rectified_flow is supported.")
        if self.prediction_type != "flow_velocity":
            raise ValueError("Only prediction_type=flow_velocity is supported.")
        if self.mode != "latent":
            raise ValueError("Only mode=latent is supported.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")
        if self.latent_downsample_factor <= 0:
            raise ValueError("latent_downsample_factor must be positive.")
        if self.latent_patch_size <= 0:
            raise ValueError("latent_patch_size must be positive.")
        if self.image_size % self.latent_downsample_factor != 0:
            raise ValueError("image_size must be divisible by latent_downsample_factor.")
        latent_side = self.image_size // self.latent_downsample_factor
        if latent_side % self.latent_patch_size != 0:
            raise ValueError("latent side must be divisible by latent_patch_size.")
        if self.latent_dtype not in {"fp16", "bf16"}:
            raise ValueError("latent_dtype must be 'fp16' or 'bf16'.")
        if self.latent_cache_sharded and not self.latent_cache:
            raise ValueError("latent_cache_sharded requires latent_cache=true.")
        if self.latent_shard_cache_size <= 0:
            raise ValueError("latent_shard_cache_size must be positive.")
        if self.text_shard_cache_size <= 0:
            raise ValueError("text_shard_cache_size must be positive.")
        if not self.text_cache and not self.allow_on_the_fly_text:
            raise ValueError("text_cache=false is only allowed when allow_on_the_fly_text=true.")
        if not isinstance(self.text_field, str):
            raise ValueError("text_field must be a string.")
        if not isinstance(self.text_fields, list) or any(not isinstance(x, str) for x in self.text_fields):
            raise ValueError("text_fields must be a list of strings.")
        if self.text_field.strip() and self.text_field.strip() not in [x.strip() for x in self.text_fields if x.strip()]:
            # text_field is valid as a shortcut and will be prepended to text_fields at read time.
            pass

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.min_lr_ratio <= 0 or self.min_lr_ratio > 1:
            raise ValueError("min_lr_ratio must be in (0, 1].")
        if self.decay_steps < 0:
            raise ValueError("decay_steps must be non-negative.")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive.")
        if self.dataset_limit < 0:
            raise ValueError("dataset_limit must be non-negative.")
        if self.val_every < 0 or self.val_batches < 0:
            raise ValueError("validation cadence and batch counts must be non-negative.")
        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError("amp_dtype must be 'fp16' or 'bf16'.")
        if self.distributed_backend not in {"none", "accelerate"}:
            raise ValueError("distributed_backend must be one of: none, accelerate.")
        if not isinstance(self.save_on_rank0_only, bool):
            raise ValueError("save_on_rank0_only must be boolean.")
        if not isinstance(self.distributed_metrics_aggregation, bool):
            raise ValueError("distributed_metrics_aggregation must be boolean.")
        if not isinstance(self.ddp_find_unused_parameters, bool):
            raise ValueError("ddp_find_unused_parameters must be boolean.")
        if not isinstance(self.ddp_gradient_as_bucket_view, bool):
            raise ValueError("ddp_gradient_as_bucket_view must be boolean.")
        if self.fsdp_min_hidden_dim <= 0:
            raise ValueError("fsdp_min_hidden_dim must be positive.")
        if self.fsdp_min_num_params <= 0:
            raise ValueError("fsdp_min_num_params must be positive.")
        if self.fsdp_sharding_strategy not in {"full_shard", "shard_grad_op", "hybrid_shard"}:
            raise ValueError("fsdp_sharding_strategy must be one of: full_shard, shard_grad_op, hybrid_shard.")
        if self.fsdp_auto_wrap_policy not in {"transformer_block", "size_based", "none"}:
            raise ValueError("fsdp_auto_wrap_policy must be one of: transformer_block, size_based, none.")
        if bool(self.fsdp_enabled):
            raise ValueError("fsdp.enabled=true is reserved for future large-model runs and is not supported by this trainer yet.")
        if self.lr_scheduler not in {"cosine", "linear"}:
            raise ValueError("lr_scheduler must be 'cosine' or 'linear'.")
        if self.optimizer not in {"adamw", "adamw_8bit"}:
            raise ValueError("optimizer must be one of: adamw, adamw_8bit.")
        if self.ckpt_keep_last < 0:
            raise ValueError("ckpt_keep_last must be non-negative.")

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.double_stream_blocks < 0 or self.single_stream_blocks < 0:
            raise ValueError("block counts must be non-negative.")
        if self.double_stream_blocks + self.single_stream_blocks != self.depth:
            raise ValueError("double_stream_blocks + single_stream_blocks must equal depth.")
        if self.pos_embed not in {"rope_2d", "sincos_2d", "none"}:
            raise ValueError("pos_embed must be one of: rope_2d, sincos_2d, none.")
        if self.text_dim <= 0 or self.pooled_dim <= 0:
            raise ValueError("text_dim and pooled_dim must be positive.")

        allowed_tasks = {"txt2img", "img2img", "inpaint", "control"}
        unknown_tasks = sorted(set(self.dataset_tasks) - allowed_tasks)
        if unknown_tasks:
            raise ValueError("dataset_tasks contains unsupported task(s): " + ", ".join(unknown_tasks))
        if any(float(v) < 0 for v in self.dataset_tasks.values()):
            raise ValueError("dataset_tasks weights must be non-negative.")
        if sum(float(v) for v in self.dataset_tasks.values()) <= 0:
            raise ValueError("dataset_tasks must have at least one positive weight.")
        if float(self.dataset_tasks.get("control", 0.0)) > 0 and not bool(self.control_enabled):
            raise ValueError("dataset_tasks.control is reserved unless control.enabled=true.")
        if self.control_strength < 0:
            raise ValueError("control_strength must be non-negative.")
        if self.control_num_streams <= 0:
            raise ValueError("control_num_streams must be positive.")
        if any(not isinstance(v, bool) for v in self.control_types.values()):
            raise ValueError("control_types values must be boolean.")
        if bool(self.control_enabled) and not any(bool(v) for v in self.control_types.values()):
            raise ValueError("control.enabled=true requires at least one enabled control type.")

        if not (0.0 <= float(self.inpaint_mask_min_area) <= float(self.inpaint_mask_max_area) <= 1.0):
            raise ValueError("inpaint mask area must satisfy 0 <= min <= max <= 1.")
        allowed_mask_modes = {"rectangle", "brush", "center_rectangle", "full", "small", "large", "random_blocks"}
        unknown_mask_modes = sorted(set(self.inpaint_mask_modes) - allowed_mask_modes)
        if unknown_mask_modes:
            raise ValueError("inpaint_mask_modes contains unsupported mode(s): " + ", ".join(unknown_mask_modes))
        if any(float(v) < 0 for v in self.inpaint_mask_modes.values()):
            raise ValueError("inpaint_mask_modes weights must be non-negative.")
        if sum(float(v) for v in self.inpaint_mask_modes.values()) <= 0:
            raise ValueError("inpaint_mask_modes must include at least one positive weight.")
        if float(self.inpaint_mask_modes.get("full", 0.0)) > 0 and float(self.inpaint_mask_max_area) < 1.0:
            raise ValueError("inpaint_mask_modes.full requires inpaint_mask_max_area=1.0.")
        if self.inpaint_loss_mask_weight < 0 or self.inpaint_loss_unmask_weight < 0:
            raise ValueError("inpaint loss weights must be non-negative.")

        for name, value in (
            ("cond_drop_prob", self.cond_drop_prob),
            ("token_drop_prob", self.token_drop_prob),
            ("tag_drop_prob", self.tag_drop_prob),
            ("caption_drop_prob", self.caption_drop_prob),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")
        if not (0.0 <= self.flow_train_t_min <= self.flow_train_t_max <= 1.0):
            raise ValueError("flow_train_t_min/max must satisfy 0 <= min <= max <= 1.")
        if self.flow_timestep_sampling not in {"uniform", "logit_normal", "shifted_logit_normal", "cosmap", "cosmap_like"}:
            raise ValueError("flow_timestep_sampling must be one of: uniform, logit_normal, shifted_logit_normal, cosmap, cosmap_like.")
        if self.flow_timestep_shift <= 0:
            raise ValueError("flow_timestep_shift must be positive.")
        if self.flow_loss_weighting not in {"none"}:
            raise ValueError("flow_loss_weighting must be 'none'.")
        if self.sampling_sampler not in {"flow_euler", "flow_heun"}:
            raise ValueError("sampling_sampler must be one of: flow_euler, flow_heun.")
        if self.sampling_steps <= 0:
            raise ValueError("sampling_steps must be positive.")
        if self.sampling_cfg_scale < 0:
            raise ValueError("sampling_cfg_scale must be non-negative.")
        if self.sampling_shift <= 0:
            raise ValueError("sampling_shift must be positive.")
        if self.eval_every < 0:
            raise ValueError("eval_every must be non-negative.")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive.")
        if self.eval_cfg < 0:
            raise ValueError("eval_cfg must be non-negative.")
        if self.eval_n <= 0:
            raise ValueError("eval_n must be positive.")
        if self.eval_sampler not in {"flow_euler", "flow_heun"}:
            raise ValueError("eval_sampler must be one of: flow_euler, flow_heun.")
        if not (0.0 < self.ema_decay_fast <= 1.0):
            raise ValueError("ema_decay_fast must be in (0, 1].")
        if not (0.0 < self.ema_decay_slow <= 1.0):
            raise ValueError("ema_decay_slow must be in (0, 1].")
        if self.ema_switch_step < 0:
            raise ValueError("ema_switch_step must be non-negative.")
        if self.curriculum_steps < 0:
            raise ValueError("curriculum_steps must be non-negative.")
        if self.curriculum_solo_weight <= 0 or self.curriculum_non_solo_weight <= 0:
            raise ValueError("curriculum_solo_weight and curriculum_non_solo_weight must be positive.")

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        from .loader import load_yaml

        return cls.from_dict(load_yaml(path))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        data = _flatten_nested_config(data)
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in fields}
        extra = {k: v for k, v in data.items() if k not in fields}
        cfg = cls(**kwargs)
        return replace(cfg, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        return data
