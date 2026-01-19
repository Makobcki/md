from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from config.train import TrainConfig
from data_loader import (
    DataConfig,
    ImageTextDataset,
    LatentCacheMetadata,
    ShardAwareBatchSampler,
    build_or_load_index,
    build_token_cache_key,
    collate_with_tokenizer,
)
from diffusion.core.diffusion import Diffusion, DiffusionConfig
from diffusion.domains.latent import LatentDomain
from diffusion.domains.pixel import PixelDomain
from diffusion.perf import PerfConfig, configure_performance
from diffusion.utils import EMA, build_run_metadata, seed_everything
from diffusion.vae import VAEWrapper
from model.unet.unet import UNet, UNetConfig
from text_enc import BPETokenizer
from text_enc.build import build_tokenizer
from text_enc.tokenizer import TextConfig
from train.checkpoint import (
    load_ckpt,
    normalize_state_dict_for_keys,
    normalize_state_dict_for_model,
    resolve_resume_path,
)
from train.curriculum import _build_curriculum_weights
from train.eval import _load_eval_prompts
from train.loop import run_training_loop
from train.sanity import _sanity_overfit


def _resolve_num_workers(requested: int) -> int:
    if requested == 0:
        return 0
    cpu_count = os.cpu_count() or 0
    max_workers = max(cpu_count - 1, 1) if cpu_count else 0
    if requested < 0:
        return max_workers
    if max_workers:
        return min(requested, max_workers)
    return requested


def _assert_divisible(value: int, divisor: int, name: str) -> None:
    if value % divisor != 0:
        raise RuntimeError(f"{name} must be divisible by {divisor}, got {value}.")


def run(cfg: TrainConfig) -> None:
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perf_active = configure_performance(
        PerfConfig(
            tf32=bool(cfg.tf32),
            cudnn_benchmark=bool(cfg.cudnn_benchmark),
            channels_last=bool(cfg.channels_last),
            enable_flash_sdp=bool(cfg.enable_flash_sdp),
            enable_mem_efficient_sdp=bool(cfg.enable_mem_efficient_sdp),
            enable_math_sdp=bool(cfg.enable_math_sdp),
        ),
        device,
    )

    seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))

    resume = str(cfg.resume_ckpt).strip()
    resume_path = ""
    ck = None
    model_cfg = cfg
    if resume:
        resume_path = resolve_resume_path(resume, out_dir)
        ck = load_ckpt(resume_path, device)
        if "cfg" in ck and isinstance(ck["cfg"], dict):
            model_cfg = TrainConfig.from_dict(ck["cfg"])

    use_text_conditioning = bool(model_cfg.use_text_conditioning)
    if ck is not None:
        meta_flag = ck.get("meta", {}).get("use_text_conditioning")
        if isinstance(meta_flag, bool):
            use_text_conditioning = meta_flag
    images_only = bool(cfg.images_only)
    if images_only:
        use_text_conditioning = False
    effective_cond_drop_prob = float(cfg.cond_drop_prob) if use_text_conditioning else 1.0
    effective_token_drop_prob = float(cfg.token_drop_prob) if use_text_conditioning else 0.0
    effective_tag_drop_prob = float(cfg.tag_drop_prob) if use_text_conditioning else 0.0
    effective_caption_drop_prob = float(cfg.caption_drop_prob) if use_text_conditioning else 0.0

    self_conditioning = bool(model_cfg.self_conditioning)
    if ck is not None:
        meta_flag = ck.get("meta", {}).get("self_conditioning")
        if isinstance(meta_flag, bool):
            self_conditioning = meta_flag
    self_cond_prob = float(cfg.self_cond_prob)
    if self_cond_prob > 0 and not self_conditioning:
        raise RuntimeError("self_cond_prob > 0 requires self_conditioning=true.")

    cfg_dict = model_cfg.to_dict()
    cfg_dict.update({
        "data_root": cfg.data_root,
        "image_dir": cfg.image_dir,
        "meta_dir": cfg.meta_dir,
        "tags_dir": cfg.tags_dir,
        "caption_field": cfg.caption_field,
        "images_only": cfg.images_only,
        "min_tag_count": cfg.min_tag_count,
        "require_512": cfg.require_512,
        "val_ratio": cfg.val_ratio,
        "cache_dir": cfg.cache_dir,
        "failed_list": cfg.failed_list,
        "seed": cfg.seed,
        "out_dir": cfg.out_dir,
        "batch_size": cfg.batch_size,
        "grad_accum_steps": cfg.grad_accum_steps,
        "num_workers": cfg.num_workers,
        "prefetch_factor": cfg.prefetch_factor,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler": cfg.lr_scheduler,
        "warmup_steps": cfg.warmup_steps,
        "min_lr_ratio": cfg.min_lr_ratio,
        "decay_steps": cfg.decay_steps,
        "max_steps": cfg.max_steps,
        "log_every": cfg.log_every,
        "save_every": cfg.save_every,
        "use_text_conditioning": use_text_conditioning,
        "cond_drop_prob": effective_cond_drop_prob,
        "token_drop_prob": effective_token_drop_prob,
        "tag_drop_prob": effective_tag_drop_prob,
        "caption_drop_prob": effective_caption_drop_prob,
        "amp": cfg.amp,
        "amp_dtype": cfg.amp_dtype,
        "compile": cfg.compile,
        "compile_warmup_steps": cfg.compile_warmup_steps,
        "compile_cudagraphs": cfg.compile_cudagraphs,
        "grad_clip_norm": cfg.grad_clip_norm,
        "ema_decay": cfg.ema_decay,
        "ema_decay_fast": cfg.ema_decay_fast,
        "ema_decay_slow": cfg.ema_decay_slow,
        "ema_switch_step": cfg.ema_switch_step,
        "resume_ckpt": resume_path or cfg.resume_ckpt,
        "deterministic": cfg.deterministic,
        "sanity_overfit_steps": cfg.sanity_overfit_steps,
        "sanity_overfit_images": cfg.sanity_overfit_images,
        "sanity_overfit_max_loss": cfg.sanity_overfit_max_loss,
        "tf32": cfg.tf32,
        "cudnn_benchmark": cfg.cudnn_benchmark,
        "channels_last": cfg.channels_last,
        "enable_flash_sdp": cfg.enable_flash_sdp,
        "enable_mem_efficient_sdp": cfg.enable_mem_efficient_sdp,
        "enable_math_sdp": cfg.enable_math_sdp,
        "mode": cfg.mode,
        "latent_channels": cfg.latent_channels,
        "latent_downsample_factor": cfg.latent_downsample_factor,
        "latent_cache": cfg.latent_cache,
        "latent_cache_dir": cfg.latent_cache_dir,
        "latent_cache_sharded": cfg.latent_cache_sharded,
        "latent_cache_index": cfg.latent_cache_index,
        "latent_dtype": cfg.latent_dtype,
        "latent_precompute": cfg.latent_precompute,
        "latent_cache_fallback": cfg.latent_cache_fallback,
        "latent_cache_strict": cfg.latent_cache_strict,
        "vae_pretrained": cfg.vae_pretrained,
        "vae_freeze": cfg.vae_freeze,
        "vae_scaling_factor": cfg.vae_scaling_factor,
        "ckpt_keep_last": cfg.ckpt_keep_last,
        "curriculum_enabled": cfg.curriculum_enabled,
        "curriculum_steps": cfg.curriculum_steps,
        "curriculum_require_one_person": cfg.curriculum_require_one_person,
        "curriculum_prefer_solo": cfg.curriculum_prefer_solo,
        "curriculum_exclude_multi": cfg.curriculum_exclude_multi,
        "curriculum_solo_weight": cfg.curriculum_solo_weight,
        "curriculum_non_solo_weight": cfg.curriculum_non_solo_weight,
        "self_conditioning": self_conditioning,
        "self_cond_prob": cfg.self_cond_prob,
        "noise_schedule": model_cfg.noise_schedule,
        "cosine_s": model_cfg.cosine_s,
        "eval_prompts_file": cfg.eval_prompts_file,
        "eval_every": cfg.eval_every,
        "eval_seed": cfg.eval_seed,
        "eval_sampler": cfg.eval_sampler,
        "eval_steps": cfg.eval_steps,
        "eval_cfg": cfg.eval_cfg,
        "eval_n": cfg.eval_n,
    })

    dcfg = DataConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        images_only=images_only,
        use_text_conditioning=use_text_conditioning,
        min_tag_count=int(cfg.min_tag_count),
        require_512=bool(cfg.require_512),
        val_ratio=float(cfg.val_ratio),
        seed=int(cfg.seed),
        cache_dir=str(cfg.cache_dir),
        failed_list=str(cfg.failed_list),
    )

    text_cfg: Optional[TextConfig] = None
    tokenizer: Optional[BPETokenizer] = None
    if use_text_conditioning:
        text_cfg = TextConfig(
            vocab_path=str(model_cfg.text_vocab_path),
            merges_path=str(model_cfg.text_merges_path),
            max_len=int(model_cfg.text_max_len),
            lowercase=True,
            strip_punct=True,
        )
        tokenizer = build_tokenizer(text_cfg)
        cfg_dict["text_max_len"] = int(text_cfg.max_len)

    mode = str(cfg.mode)
    train_entries, _val_entries = build_or_load_index(dcfg)
    with open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)
    run_meta = build_run_metadata(perf_active)
    run_meta["use_text_conditioning"] = use_text_conditioning
    run_meta["self_conditioning"] = self_conditioning
    with open(out_dir / "run_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(run_meta, f, sort_keys=False, allow_unicode=True)

    latent_dtype = torch.bfloat16 if cfg.latent_dtype == "bf16" else torch.float16
    latent_expected_meta: LatentCacheMetadata | None = None
    if mode == "latent":
        _assert_divisible(int(cfg.image_size), int(cfg.latent_downsample_factor), "image_size")
        if not bool(cfg.latent_cache_strict) and not bool(cfg.latent_cache_fallback):
            raise RuntimeError("latent_cache_strict=false requires latent_cache_fallback=true.")
        latent_side = int(cfg.image_size) // int(cfg.latent_downsample_factor)
        latent_expected_meta = LatentCacheMetadata(
            vae_pretrained=str(cfg.vae_pretrained),
            scaling_factor=float(cfg.vae_scaling_factor),
            latent_shape=(int(cfg.latent_channels), latent_side, latent_side),
            dtype=str(cfg.latent_dtype),
        )
    ds = ImageTextDataset(
        entries=train_entries,
        tokenizer=tokenizer,
        cond_drop_prob=effective_cond_drop_prob,
        token_drop_prob=effective_token_drop_prob,
        tag_drop_prob=effective_tag_drop_prob,
        caption_drop_prob=effective_caption_drop_prob,
        seed=int(cfg.seed),
        cache_dir=str(Path(cfg.data_root) / cfg.cache_dir),
        token_cache_key=(
            build_token_cache_key(
                vocab_path=model_cfg.text_vocab_path,
                merges_path=model_cfg.text_merges_path,
                caption_field=cfg.caption_field,
                max_len=int(text_cfg.max_len),
                lowercase=bool(text_cfg.lowercase),
                strip_punct=bool(text_cfg.strip_punct),
            )
            if text_cfg is not None
            else None
        ),
        latent_cache_dir=str(Path(cfg.data_root) / cfg.latent_cache_dir),
        latent_cache_sharded=bool(cfg.latent_cache_sharded),
        latent_cache_index_path=str(cfg.latent_cache_index),
        latent_dtype=latent_dtype,
        return_latents=(mode == "latent"),
        latent_cache_strict=bool(cfg.latent_cache_strict),
        latent_cache_fallback=bool(cfg.latent_cache_fallback),
        latent_expected_meta=latent_expected_meta,
        include_is_latent=bool(cfg.latent_cache_fallback),
        latent_missing_log_path=out_dir / "latent_missing.txt" if mode == "latent" else None,
        latent_shard_cache_size=int(cfg.latent_shard_cache_size),
    )
    if mode == "latent" and len(ds) == 0:
        raise RuntimeError("latent cache is empty; run scripts/prepare_latents.py before training.")

    latent_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    if mode == "latent":
        if not bool(cfg.latent_cache):
            raise RuntimeError("mode=latent requires latent_cache=true (use scripts/prepare_latents.py).")
        if bool(cfg.latent_cache_fallback):
            if not str(cfg.vae_pretrained):
                raise RuntimeError("latent_cache_fallback requires vae_pretrained.")
            vae = VAEWrapper(
                pretrained=str(cfg.vae_pretrained),
                freeze=bool(cfg.vae_freeze),
                scaling_factor=float(cfg.vae_scaling_factor),
                device=device,
                dtype=latent_dtype,
            )

            def _encode_latents(x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    z = vae.encode(x.to(device=device, dtype=latent_dtype))
                return z.cpu()

            latent_encoder = _encode_latents

    nw = _resolve_num_workers(int(cfg.num_workers))
    if latent_encoder is not None:
        nw = 0
    curriculum_steps = int(cfg.curriculum_steps)
    curriculum_enabled = bool(cfg.curriculum_enabled) and use_text_conditioning and curriculum_steps > 0
    use_shard_sampler = mode == "latent" and bool(cfg.latent_cache_sharded)
    if use_shard_sampler and curriculum_enabled:
        curriculum_enabled = False
        print("[WARN] curriculum disabled for shard-aware latent sampling")
    if use_shard_sampler:
        shard_to_indices = ds.shard_to_entry_indices()
        if not shard_to_indices:
            raise RuntimeError("Shard-aware sampling requires a valid shard index.")
        batch_sampler = ShardAwareBatchSampler(
            shard_to_entry_indices=shard_to_indices,
            batch_size=int(cfg.batch_size),
            drop_last=True,
            seed=int(cfg.seed),
        )
        dl_full = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=nw,
            pin_memory=bool(cfg.pin_memory),
            persistent_workers=bool(cfg.persistent_workers) and nw > 0,
            prefetch_factor=int(cfg.prefetch_factor) if nw > 0 else None,
            collate_fn=lambda batch: collate_with_tokenizer(batch, latent_encoder=latent_encoder),
        )
    else:
        dl_full = DataLoader(
            ds,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            num_workers=nw,
            pin_memory=bool(cfg.pin_memory),
            drop_last=True,
            persistent_workers=bool(cfg.persistent_workers) and nw > 0,
            prefetch_factor=int(cfg.prefetch_factor) if nw > 0 else None,
            collate_fn=lambda batch: collate_with_tokenizer(batch, latent_encoder=latent_encoder),
        )
    dl_curr = None
    if curriculum_enabled:
        weights = _build_curriculum_weights(
            train_entries,
            require_one_person=bool(cfg.curriculum_require_one_person),
            prefer_solo=bool(cfg.curriculum_prefer_solo),
            exclude_multi=bool(cfg.curriculum_exclude_multi),
            solo_weight=float(cfg.curriculum_solo_weight),
            non_solo_weight=float(cfg.curriculum_non_solo_weight),
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dl_curr = DataLoader(
            ds,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            sampler=sampler,
            num_workers=nw,
            pin_memory=bool(cfg.pin_memory),
            drop_last=True,
            persistent_workers=bool(cfg.persistent_workers) and nw > 0,
            prefetch_factor=int(cfg.prefetch_factor) if nw > 0 else None,
            collate_fn=lambda batch: collate_with_tokenizer(batch, latent_encoder=latent_encoder),
        )

    if mode == "latent":
        _assert_divisible(int(cfg.image_size), int(cfg.latent_downsample_factor), "image_size")

    image_channels = 4 if mode == "pixel" else int(cfg.latent_channels)
    unet_cfg = UNetConfig(
        image_channels=image_channels,
        base_channels=int(model_cfg.base_channels),
        channel_mults=tuple(model_cfg.channel_mults),
        num_res_blocks=int(model_cfg.num_res_blocks),
        dropout=float(model_cfg.dropout),
        attn_resolutions=tuple(model_cfg.attn_resolutions),
        attn_heads=int(model_cfg.attn_heads),
        attn_head_dim=int(model_cfg.attn_head_dim),
        vocab_size=len(tokenizer.vocab) if tokenizer is not None else 0,
        text_dim=int(model_cfg.text_dim),
        text_layers=int(model_cfg.text_layers),
        text_heads=int(model_cfg.text_heads),
        text_max_len=int(model_cfg.text_max_len),
        use_text_conditioning=use_text_conditioning,
        self_conditioning=self_conditioning,
        use_scale_shift_norm=bool(model_cfg.use_scale_shift_norm),
        grad_checkpointing=bool(cfg.grad_checkpointing),
    )

    model = UNet(unet_cfg).to(device)
    if bool(cfg.channels_last):
        model = model.to(memory_format=torch.channels_last)

    if bool(cfg.compile) and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        fused=(device.type == "cuda"),
    )

    use_amp = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_decay_fast = float(cfg.ema_decay_fast)
    ema_decay_slow = float(cfg.ema_decay_slow)
    ema_switch_step = int(cfg.ema_switch_step)
    if ema_switch_step <= 0:
        ema_decay_fast = float(cfg.ema_decay)
        ema_decay_slow = float(cfg.ema_decay)
    ema = EMA(model, decay=ema_decay_fast)

    prediction_type = str(model_cfg.prediction_type)
    diff = Diffusion(
        DiffusionConfig(
            timesteps=int(model_cfg.timesteps),
            beta_start=float(model_cfg.beta_start),
            beta_end=float(model_cfg.beta_end),
            prediction_type=prediction_type,
            noise_schedule=str(model_cfg.noise_schedule),
            cosine_s=float(model_cfg.cosine_s),
        ),
        device=device,
    )
    if mode == "latent":
        domain = LatentDomain(
            diffusion=diff,
            device=device,
            channels_last=bool(cfg.channels_last),
        )
    else:
        domain = PixelDomain(
            diffusion=diff,
            device=device,
            channels_last=bool(cfg.channels_last),
        )

    start_step = 0
    if resume:
        if ck is None:
            ck = load_ckpt(resume, device)
        model_state = normalize_state_dict_for_model(ck["model"], model, name="model")
        model.load_state_dict(model_state, strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        elif "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        if "ema" in ck:
            ema_state = normalize_state_dict_for_keys(ck["ema"], ema.shadow.keys(), name="ema")
            ema.shadow = {k: v.to(device) for k, v in ema_state.items()}
        for group in opt.param_groups:
            group["lr"] = float(cfg.lr)
        start_step = int(ck.get("step", 0)) + 1

    eval_every = int(cfg.eval_every)
    eval_prompts: list[str] | None = None
    eval_sampler = str(cfg.eval_sampler)
    eval_steps = int(cfg.eval_steps)
    eval_cfg = float(cfg.eval_cfg)
    eval_seed = int(cfg.eval_seed)
    eval_n = int(cfg.eval_n)
    eval_vae: Optional[VAEWrapper] = None
    if eval_every > 0:
        eval_prompts = _load_eval_prompts(str(cfg.eval_prompts_file), count=5)
        if mode == "latent":
            if not str(cfg.vae_pretrained):
                raise RuntimeError("eval in latent mode requires vae_pretrained.")
            eval_vae = VAEWrapper(
                pretrained=str(cfg.vae_pretrained),
                freeze=True,
                scaling_factor=float(cfg.vae_scaling_factor),
                device=device,
                dtype=latent_dtype,
            )

    sanity_steps = int(cfg.sanity_overfit_steps)
    sanity_images = int(cfg.sanity_overfit_images)
    sanity_max_loss = float(cfg.sanity_overfit_max_loss)
    _sanity_overfit(
        model=model,
        tokenizer=tokenizer,
        entries=train_entries,
        diff=diff,
        domain=domain,
        latent_mode=(mode == "latent"),
        latent_cache_dir=str(Path(cfg.data_root) / cfg.latent_cache_dir) if mode == "latent" else None,
        latent_cache_sharded=bool(cfg.latent_cache_sharded),
        latent_cache_index_path=str(cfg.latent_cache_index),
        latent_dtype=latent_dtype if mode == "latent" else None,
        latent_cache_strict=bool(cfg.latent_cache_strict),
        latent_cache_fallback=bool(cfg.latent_cache_fallback),
        latent_expected_meta=latent_expected_meta,
        latent_encoder=latent_encoder,
        use_text_conditioning=use_text_conditioning,
        self_conditioning=self_conditioning,
        self_cond_prob=self_cond_prob,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        steps=sanity_steps,
        max_images=sanity_images,
        max_loss=sanity_max_loss,
        opt=opt,
        scaler=scaler,
        ema=ema,
        ema_switch_step=ema_switch_step,
        ema_decay_fast=ema_decay_fast,
        ema_decay_slow=ema_decay_slow,
        log_fn=print,
    )

    run_training_loop(
        run_cfg=cfg,
        cfg_dict=cfg_dict,
        run_meta=run_meta,
        out_dir=out_dir,
        device=device,
        perf_active=perf_active,
        use_text_conditioning=use_text_conditioning,
        self_conditioning=self_conditioning,
        self_cond_prob=self_cond_prob,
        effective_cond_drop_prob=effective_cond_drop_prob,
        tokenizer=tokenizer,
        ds=ds,
        dl_full=dl_full,
        dl_curr=dl_curr,
        diff=diff,
        domain=domain,
        model=model,
        opt=opt,
        scaler=scaler,
        ema=ema,
        start_step=start_step,
        eval_prompts=eval_prompts,
        eval_sampler=eval_sampler,
        eval_steps=eval_steps,
        eval_cfg=eval_cfg,
        eval_seed=eval_seed,
        eval_n=eval_n,
        eval_vae=eval_vae,
        compile_cudagraphs=bool(cfg.compile_cudagraphs),
        amp_dtype=amp_dtype,
    )
