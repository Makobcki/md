from __future__ import annotations

import logging
import os
import tempfile
import hashlib
from pathlib import Path
import shutil
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
import yaml

from config.train import TrainConfig
from data_loader import (
    DataConfig,
    ImageTextDataset,
    LatentCacheMetadata,
    build_or_load_index,
    latent_cache_path,
    load_latent_shard_index,
)
from diffusion.objectives import RectifiedFlowObjective
from diffusion.perf import PerfConfig, configure_performance
from diffusion.perf.triton_compat import patch_triton_cuda_python_include_order
from diffusion.utils import EMA, build_run_metadata, seed_everything
from diffusion.vae import VAEWrapper
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.cache import TextCache
from model.text.conditioning import TextConditioning, TrainBatch
from model.text.pretrained import FrozenTextEncoderBundle
from train.checkpoint import (
    load_ckpt,
    normalize_state_dict_for_keys,
    normalize_state_dict_for_model,
    resolve_resume_path,
)
from train.eval import _resolve_eval_prompts
from train.loop_mmdit_full import run_mmdit_training_loop
from train.checkpoint_mmdit import validate_mmdit_checkpoint_compatibility


_SMALL_GPU_MAX_AUTOTUNE_WARNING = "Not enough SMs to use max_autotune_gemm mode"
_SMALL_GPU_MAX_AUTOTUNE_MIN_SMS = 68
_inductor_warning_filter_installed = False


class _SmallGpuMaxAutotuneWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _SMALL_GPU_MAX_AUTOTUNE_WARNING


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


def _atomic_write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _build_optimizer(cfg: TrainConfig, model: torch.nn.Module, device: torch.device) -> torch.optim.Optimizer:
    name = str(cfg.optimizer)
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            fused=(device.type == "cuda"),
        )
    if name == "adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("optimizer=adamw_8bit requires bitsandbytes to be installed.") from exc
        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )
    raise RuntimeError(f"Unsupported optimizer: {name}")


def _mmdit_entry_text(entry: dict) -> str:
    caption = str(entry.get("caption", "") or "")
    if caption:
        return caption
    tags = list(entry.get("tags_primary", [])) + list(entry.get("tags_gender", []))
    return " ".join(str(x) for x in tags)


def _mmdit_dataset_hash(entries: list[dict]) -> str:
    h = hashlib.sha256()
    for entry in entries:
        h.update(str(entry.get("md5", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(_mmdit_entry_text(entry).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _validate_text_cache_for_mmdit(cache: TextCache, cfg: TrainConfig, entries: list[dict]) -> None:
    if not cache.index_path.exists():
        raise RuntimeError(f"Missing text cache index: {cache.index_path}")
    if not cache.metadata:
        raise RuntimeError(f"Missing text cache metadata: {cache.metadata_path}")

    meta = cache.metadata
    if int(meta.get("text_dim", -1)) != int(cfg.text_dim):
        raise RuntimeError(f"text cache text_dim mismatch: {meta.get('text_dim')} != {cfg.text_dim}")
    if int(meta.get("pooled_dim", -1)) != int(cfg.pooled_dim):
        raise RuntimeError(f"text cache pooled_dim mismatch: {meta.get('pooled_dim')} != {cfg.pooled_dim}")

    expected_encoders = cfg.extra.get("text", {}).get("encoders", [])
    actual_encoders = meta.get("encoders", [])
    if expected_encoders and actual_encoders:
        expected = [
            {
                "name": str(item.get("name", "")),
                "model_name": str(item.get("model_name", "")),
                "max_length": int(item.get("max_length", 0)),
            }
            for item in expected_encoders
        ]
        actual = [
            {
                "name": str(item.get("name", "")),
                "model_name": str(item.get("model_name", "")),
                "max_length": int(item.get("max_length", 0)),
            }
            for item in actual_encoders
        ]
        if expected != actual:
            raise RuntimeError(f"text cache encoder metadata mismatch: cache={actual!r}, config={expected!r}")

    expected_hash = meta.get("dataset_hash")
    if isinstance(expected_hash, str) and expected_hash:
        actual_hash = _mmdit_dataset_hash(entries)
        if actual_hash != expected_hash:
            raise RuntimeError(f"text cache dataset_hash mismatch: {expected_hash} != {actual_hash}")

    missing = [str(entry.get("md5", "")) for entry in entries if str(entry.get("md5", "")) not in cache.entries]
    if missing:
        examples = ", ".join(missing[:10])
        raise RuntimeError(f"text cache missing {len(missing)} md5 keys used by dataset. Examples: {examples}")


def _prepare_section(cfg: TrainConfig, name: str) -> dict[str, Any]:
    section = cfg.extra.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RuntimeError(f"{name} config section must be a mapping.")
    return {str(k).replace("-", "_"): v for k, v in section.items()}


def _resolve_auto_device(value: object) -> torch.device:
    requested = str(value or "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise RuntimeError("prepare device must be one of: auto, cpu, cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    return torch.device(requested)


def _resolve_text_prepare_dtype(value: object, cfg: TrainConfig, device: torch.device) -> torch.dtype:
    requested = str(value or "auto")
    if requested == "auto":
        if device.type == "cpu":
            return torch.float32
        requested = str(cfg.latent_dtype)
    if requested in {"fp32", "float32"}:
        return torch.float32
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    raise RuntimeError("prepare_text.dtype must be one of: auto, fp32, bf16, fp16.")


def _prepare_text_cache_for_training(cfg: TrainConfig, device: torch.device) -> None:
    from scripts.prepare_text_cache import prepare_text_cache

    section = _prepare_section(cfg, "prepare_text")
    prepare_device = _resolve_auto_device(section.get("device", "auto"))
    prepare_text_cache(
        cfg=cfg,
        out_dir=Path(cfg.data_root) / str(cfg.text_cache_dir),
        batch_size=int(section.get("batch_size", 8)),
        shard_size=int(section.get("shard_size", 1024)),
        limit=None,
        device=prepare_device,
        dtype=_resolve_text_prepare_dtype(section.get("dtype", "auto"), cfg, prepare_device),
    )


def _ensure_text_cache_ready(cfg: TrainConfig, entries: list[dict], device: torch.device) -> None:
    if not bool(cfg.text_cache):
        raise RuntimeError("architecture=mmdit_rf requires cache.text_cache=true.")

    cache_root = Path(cfg.data_root) / str(cfg.text_cache_dir)
    cache = TextCache(cache_root, shard_cache_size=int(cfg.text_shard_cache_size))
    missing = not cache.index_path.exists() or not cache.metadata_path.exists() or not (cache.root / "empty_prompt.safetensors").exists()

    if missing:
        if not bool(cfg.cache_auto_prepare):
            raise RuntimeError(f"Missing text cache at {cache_root}; set cache.auto_prepare=true or run scripts/prepare_text_cache.py.")
        print(f"[INFO] Missing text cache at {cache_root}; preparing before training.", flush=True)
        _prepare_text_cache_for_training(cfg, device)
        cache = TextCache(cache_root, shard_cache_size=int(cfg.text_shard_cache_size))

    try:
        _validate_text_cache_for_mmdit(cache, cfg, entries)
    except RuntimeError as exc:
        if not bool(cfg.cache_rebuild_if_stale):
            raise RuntimeError(
                f"Text cache is stale or incompatible: {exc}\n"
                "Set cache.rebuild_if_stale=true to rebuild it automatically."
            ) from exc
        print(f"[INFO] Rebuilding stale text cache at {cache_root}: {exc}", flush=True)
        shutil.rmtree(cache_root, ignore_errors=True)
        _prepare_text_cache_for_training(cfg, device)
        _validate_text_cache_for_mmdit(
            TextCache(cache_root, shard_cache_size=int(cfg.text_shard_cache_size)),
            cfg,
            entries,
        )


def _latent_expected_metadata(cfg: TrainConfig) -> dict[str, Any]:
    latent_side = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    return {
        "format_version": 3,
        "vae_pretrained": str(cfg.vae_pretrained),
        "scaling_factor": float(cfg.vae_scaling_factor),
        "latent_shape": [int(cfg.latent_channels), latent_side, latent_side],
        "dtype": str(cfg.latent_dtype),
    }


def _latent_cache_state(cfg: TrainConfig, entries: list[dict]) -> tuple[str, str]:
    cache_dir = _resolve_latent_cache_dir(cfg)
    if bool(cfg.latent_cache_sharded):
        index_path = _resolve_latent_shard_index_path(cfg)
        if not index_path.exists():
            return "missing", f"missing sharded latent index: {index_path}"
        from scripts.prepare_latents import _sharded_cache_mismatch_reason

        mismatch = _sharded_cache_mismatch_reason(
            index_path=index_path,
            shard_dir=cache_dir / "shards",
            expected_meta=_latent_expected_metadata(cfg),
        )
        if mismatch is not None:
            return "stale", mismatch
        try:
            index = load_latent_shard_index(index_path)
        except Exception as exc:
            return "stale", f"cannot read sharded latent index: {exc}"
        missing = [str(entry.get("md5", "")) for entry in entries if str(entry.get("md5", "")) not in index]
        if missing:
            return "missing", f"latent cache missing {len(missing)} md5 keys. Examples: {', '.join(missing[:10])}"
        return "ready", ""

    missing_paths = [
        str(latent_cache_path(cache_dir, str(entry.get("md5", ""))))
        for entry in entries
        if not latent_cache_path(cache_dir, str(entry.get("md5", ""))).exists()
    ]
    if missing_paths:
        return "missing", f"latent cache missing {len(missing_paths)} files. Examples: {', '.join(missing_paths[:3])}"
    return "ready", ""


def _prepare_latent_cache_for_training(cfg: TrainConfig, *, rebuild: bool) -> None:
    from scripts.prepare_latents import prepare_latent_cache_for_config

    prepare_latent_cache_for_config(cfg, overwrite=rebuild)


def _ensure_latent_cache_ready_for_mmdit(cfg: TrainConfig, entries: list[dict], device: torch.device) -> None:
    del device
    if not bool(cfg.latent_cache):
        raise RuntimeError("architecture=mmdit_rf requires cache.latent_cache=true.")
    if not str(cfg.vae_pretrained):
        raise RuntimeError("architecture=mmdit_rf requires vae.pretrained/vae_pretrained for latent cache preparation.")

    state, reason = _latent_cache_state(cfg, entries)
    if state == "ready":
        return
    if state == "stale" and not bool(cfg.cache_rebuild_if_stale):
        raise RuntimeError(
            f"Latent cache is stale or incompatible: {reason}\n"
            "Set cache.rebuild_if_stale=true to rebuild it automatically."
        )
    if not bool(cfg.cache_auto_prepare):
        raise RuntimeError(f"Latent cache is {state}: {reason}; set cache.auto_prepare=true or run scripts/prepare_latents.py.")

    rebuild = state == "stale"
    if rebuild:
        print(f"[INFO] Rebuilding stale latent cache at {_resolve_latent_cache_dir(cfg)}: {reason}", flush=True)
        shutil.rmtree(_resolve_latent_cache_dir(cfg), ignore_errors=True)
    else:
        print(f"[INFO] Missing latent cache at {_resolve_latent_cache_dir(cfg)}; preparing before training.", flush=True)
    _prepare_latent_cache_for_training(cfg, rebuild=rebuild)
    state, reason = _latent_cache_state(cfg, entries)
    if state != "ready":
        raise RuntimeError(f"Latent cache preparation finished but cache is still {state}: {reason}")


def _ensure_mmdit_caches_ready(cfg: TrainConfig, entries: list[dict], device: torch.device) -> None:
    _ensure_text_cache_ready(cfg, entries, device)
    _ensure_latent_cache_ready_for_mmdit(cfg, entries, device)


class _MMDiTCachedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        latent_ds: ImageTextDataset,
        text_cache: TextCache,
        *,
        dataset_tasks: dict[str, float] | None = None,
    ) -> None:
        self.latent_ds = latent_ds
        self.text_cache = text_cache
        self.entries = latent_ds.entries
        task_weights = dict(dataset_tasks or {"txt2img": 1.0})
        self.task_names = [name for name, weight in task_weights.items() if float(weight) > 0]
        weights = torch.tensor([float(task_weights[name]) for name in self.task_names], dtype=torch.float32)
        if not self.task_names or float(weights.sum().item()) <= 0:
            raise RuntimeError("MMDiT dataset_tasks must include at least one positive task weight.")
        self.task_probs = weights / weights.sum()

    def __len__(self) -> int:
        return len(self.latent_ds)

    def _sample_task(self) -> str:
        if len(self.task_names) == 1:
            return self.task_names[0]
        idx = int(torch.multinomial(self.task_probs, 1).item())
        return self.task_names[idx]

    def _random_mask(self, x0: torch.Tensor) -> torch.Tensor:
        _, h, w = x0.shape
        mask = torch.zeros((1, h, w), dtype=x0.dtype, device=x0.device)
        mh_min = max(h // 8, 1)
        mw_min = max(w // 8, 1)
        mh_max = max(h // 2, mh_min)
        mw_max = max(w // 2, mw_min)
        mh = int(torch.randint(mh_min, mh_max + 1, ()).item())
        mw = int(torch.randint(mw_min, mw_max + 1, ()).item())
        y0 = int(torch.randint(0, max(h - mh + 1, 1), ()).item())
        x0_pos = int(torch.randint(0, max(w - mw + 1, 1), ()).item())
        mask[:, y0 : y0 + mh, x0_pos : x0_pos + mw] = 1
        return mask

    def __getitem__(self, idx: int) -> TrainBatch:
        raw = self.latent_ds[idx]
        x0 = raw[0] if isinstance(raw, tuple) else raw
        key = str(self.entries[idx].get("md5", idx))
        task = self._sample_task()
        source_latent = None
        mask = None
        if task == "img2img":
            source_latent = x0.clone()
        elif task == "inpaint":
            mask = self._random_mask(x0)
            source_latent = x0 * (1.0 - mask)
        elif task != "txt2img":
            raise RuntimeError(f"Unsupported MMDiT dataset task: {task}")
        return TrainBatch(
            x0=x0,
            text=self.text_cache.load(key),
            source_latent=source_latent,
            mask=mask,
            task=task,
            metadata={"key": key, "task": task},
        )


def _collate_mmdit(batch: list[TrainBatch]) -> TrainBatch:
    has_source = any(item.source_latent is not None for item in batch)
    has_mask = any(item.mask is not None for item in batch)
    source_latent = None
    mask = None
    if has_source:
        source_latent = torch.stack(
            [
                item.source_latent if item.source_latent is not None else torch.zeros_like(item.x0)
                for item in batch
            ],
            dim=0,
        )
    if has_mask:
        mask = torch.stack(
            [
                item.mask if item.mask is not None else torch.zeros((1, *item.x0.shape[-2:]), dtype=item.x0.dtype)
                for item in batch
            ],
            dim=0,
        )
    tasks = [item.task for item in batch]
    return TrainBatch(
        x0=torch.stack([item.x0 for item in batch], dim=0),
        text=TextConditioning(
            tokens=torch.stack([item.text.tokens for item in batch], dim=0)
            if batch[0].text.tokens.dim() == 2
            else torch.cat([item.text.tokens for item in batch], dim=0),
            mask=torch.stack([item.text.mask for item in batch], dim=0)
            if batch[0].text.mask.dim() == 1
            else torch.cat([item.text.mask for item in batch], dim=0),
            pooled=torch.stack([item.text.pooled for item in batch], dim=0)
            if batch[0].text.pooled.dim() == 1
            else torch.cat([item.text.pooled for item in batch], dim=0),
            is_uncond=None,
        ),
        source_latent=source_latent,
        mask=mask,
        task=tasks[0] if all(task == tasks[0] for task in tasks) else "mixed",
        metadata={
            "keys": [item.metadata.get("key") if item.metadata else None for item in batch],
            "tasks": tasks,
        },
    )


def _run_mmdit_rf(cfg: TrainConfig, *, device: torch.device, perf_active: dict) -> None:
    if str(cfg.mode) != "latent":
        raise RuntimeError("architecture=mmdit_rf requires mode=latent.")
    if not bool(cfg.latent_cache):
        raise RuntimeError("architecture=mmdit_rf requires latent_cache=true.")
    if not bool(cfg.text_cache):
        raise RuntimeError("architecture=mmdit_rf requires text_cache=true; run scripts/prepare_text_cache.py.")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))
    cfg_dict = cfg.to_dict()
    _atomic_write_yaml(out_dir / "config_snapshot.yaml", cfg_dict)
    run_meta = build_run_metadata(perf_active)
    run_meta["architecture"] = "mmdit_rf"
    _atomic_write_yaml(out_dir / "run_meta.yaml", run_meta)

    dcfg = DataConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        images_only=False,
        use_text_conditioning=True,
        min_tag_count=int(cfg.min_tag_count),
        require_512=bool(cfg.require_512),
        val_ratio=float(cfg.val_ratio),
        seed=int(cfg.seed),
        cache_dir=str(cfg.cache_dir),
        failed_list=str(cfg.failed_list),
    )
    train_entries, val_entries = build_or_load_index(dcfg)
    if int(cfg.dataset_limit) > 0:
        train_entries = train_entries[: int(cfg.dataset_limit)]
        val_entries = []
    _ensure_mmdit_caches_ready(cfg, train_entries + val_entries, device)
    latent_dtype = torch.bfloat16 if cfg.latent_dtype == "bf16" else torch.float16
    latent_side = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    latent_expected_meta = LatentCacheMetadata(
        vae_pretrained=str(cfg.vae_pretrained),
        scaling_factor=float(cfg.vae_scaling_factor),
        latent_shape=(int(cfg.latent_channels), latent_side, latent_side),
        dtype=str(cfg.latent_dtype),
    )
    latent_ds = ImageTextDataset(
        entries=train_entries,
        tokenizer=None,
        cond_drop_prob=1.0,
        seed=int(cfg.seed),
        latent_cache_dir=str(Path(cfg.data_root) / cfg.latent_cache_dir),
        latent_cache_sharded=bool(cfg.latent_cache_sharded),
        latent_cache_index_path=str(cfg.latent_cache_index),
        latent_dtype=latent_dtype,
        return_latents=True,
        latent_cache_strict=bool(cfg.latent_cache_strict),
        latent_cache_fallback=False,
        latent_expected_meta=latent_expected_meta,
        include_is_latent=False,
        latent_missing_log_path=out_dir / "latent_missing.txt",
        latent_shard_cache_size=int(cfg.latent_shard_cache_size),
    )
    if len(latent_ds) == 0:
        raise RuntimeError("latent cache is empty; run scripts/prepare_latents.py first.")
    text_cache = TextCache(
        Path(cfg.data_root) / str(cfg.text_cache_dir),
        shard_cache_size=int(cfg.text_shard_cache_size),
    )
    _validate_text_cache_for_mmdit(text_cache, cfg, train_entries + val_entries)
    ds = _MMDiTCachedDataset(latent_ds, text_cache, dataset_tasks=cfg.dataset_tasks)
    if len(ds) == 0:
        raise RuntimeError("MMDiT dataset is empty after latent/text cache filtering.")
    dl = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=_resolve_num_workers(int(cfg.num_workers)),
        pin_memory=bool(cfg.pin_memory),
        drop_last=True,
        persistent_workers=bool(cfg.persistent_workers) and int(cfg.num_workers) != 0,
        prefetch_factor=int(cfg.prefetch_factor) if int(cfg.num_workers) != 0 else None,
        collate_fn=_collate_mmdit,
    )
    dl_val = None
    if val_entries and int(cfg.val_every) > 0 and int(cfg.val_batches) > 0:
        val_latent_ds = ImageTextDataset(
            entries=val_entries,
            tokenizer=None,
            cond_drop_prob=1.0,
            seed=int(cfg.seed),
            latent_cache_dir=str(Path(cfg.data_root) / cfg.latent_cache_dir),
            latent_cache_sharded=bool(cfg.latent_cache_sharded),
            latent_cache_index_path=str(cfg.latent_cache_index),
            latent_dtype=latent_dtype,
            return_latents=True,
            latent_cache_strict=bool(cfg.latent_cache_strict),
            latent_cache_fallback=False,
            latent_expected_meta=latent_expected_meta,
            include_is_latent=False,
            latent_missing_log_path=out_dir / "latent_missing_val.txt",
            latent_shard_cache_size=int(cfg.latent_shard_cache_size),
        )
        if len(val_latent_ds) > 0:
            dl_val = DataLoader(
                _MMDiTCachedDataset(val_latent_ds, text_cache, dataset_tasks={"txt2img": 1.0}),
                batch_size=int(cfg.batch_size),
                shuffle=False,
                num_workers=_resolve_num_workers(int(cfg.num_workers)),
                pin_memory=bool(cfg.pin_memory),
                drop_last=False,
                persistent_workers=bool(cfg.persistent_workers) and int(cfg.num_workers) != 0,
                prefetch_factor=int(cfg.prefetch_factor) if int(cfg.num_workers) != 0 else None,
                collate_fn=_collate_mmdit,
            )

    mmdit_cfg = MMDiTConfig(
        latent_channels=int(cfg.latent_channels),
        patch_size=int(cfg.latent_patch_size),
        hidden_dim=int(cfg.hidden_dim),
        depth=int(cfg.depth),
        num_heads=int(cfg.num_heads),
        mlp_ratio=float(cfg.mlp_ratio),
        qk_norm=bool(cfg.qk_norm),
        rms_norm=bool(cfg.rms_norm),
        swiglu=bool(cfg.swiglu),
        adaln_zero=bool(cfg.adaln_zero),
        pos_embed=str(cfg.pos_embed),
        double_stream_blocks=int(cfg.double_stream_blocks),
        single_stream_blocks=int(cfg.single_stream_blocks),
        dropout=float(cfg.dropout),
        attn_dropout=float(cfg.attn_dropout),
        gradient_checkpointing=bool(cfg.grad_checkpointing),
        text_dim=int(cfg.text_dim),
        pooled_dim=int(cfg.pooled_dim),
    )
    model = MMDiTFlowModel(mmdit_cfg).to(device)
    opt = _build_optimizer(cfg, model, device)
    ema = EMA(model, decay=float(cfg.ema_decay))
    use_amp = bool(cfg.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    objective = RectifiedFlowObjective(
        timestep_sampling=str(cfg.flow_timestep_sampling),
        logit_mean=float(cfg.flow_logit_mean),
        logit_std=float(cfg.flow_logit_std),
        train_t_min=float(cfg.flow_train_t_min),
        train_t_max=float(cfg.flow_train_t_max),
        loss_weighting=str(cfg.flow_loss_weighting),
    )
    empty_text = text_cache.load_empty()

    start_step = 0
    resume_path = ""
    if str(cfg.resume_ckpt).strip():
        resume_path = resolve_resume_path(str(cfg.resume_ckpt), out_dir)
        ck = _load_resume_checkpoint(resume_path)
        validate_mmdit_checkpoint_compatibility(ck, cfg_dict)
        model_state = normalize_state_dict_for_model(ck["model"], model, name="model")
        model.load_state_dict(model_state, strict=True)
        if "optimizer" in ck and ck["optimizer"] is not None:
            opt.load_state_dict(ck["optimizer"])
        elif "opt" in ck and ck["opt"] is not None:
            opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        if "ema" in ck and isinstance(ck["ema"], dict):
            ema_state = normalize_state_dict_for_keys(ck["ema"], ema.shadow.keys(), name="ema")
            ema.shadow = {k: v.to(device) for k, v in ema_state.items()}
        start_step = int(ck.get("step", 0))
        print(f"[INFO] Resumed mmdit_rf from {resume_path} at step {start_step}", flush=True)

    eval_prompts = None
    eval_text_encoder = None
    eval_vae = None
    if int(cfg.eval_every) > 0:
        eval_prompts = _resolve_eval_prompts(str(cfg.eval_prompts_file), count=5, use_text_conditioning=True)
        eval_text_encoder = FrozenTextEncoderBundle(
            cfg_dict,
            device=device,
            dtype=torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16,
        )
        if not str(cfg.vae_pretrained):
            raise RuntimeError("MMDiT eval sampling requires vae_pretrained.")
        eval_vae = VAEWrapper(
            pretrained=str(cfg.vae_pretrained),
            freeze=True,
            scaling_factor=float(cfg.vae_scaling_factor),
            device=device,
            dtype=latent_dtype,
        )

    run_mmdit_training_loop(
        cfg=cfg,
        cfg_dict=cfg_dict,
        model=model,
        dataloader=dl,
        val_dataloader=dl_val,
        optimizer=opt,
        scaler=scaler,
        objective=objective,
        device=device,
        ema=ema,
        out_dir=out_dir,
        empty_text=empty_text,
        start_step=start_step,
        text_metadata=text_cache.metadata,
        eval_prompts=eval_prompts,
        eval_text_encoder=eval_text_encoder,
        eval_vae=eval_vae,
    )


def _configure_inductor_for_compile(device: torch.device) -> dict[str, bool | int | None]:
    active: dict[str, bool | int | None] = {
        "compile_small_gpu": False,
        "compile_max_autotune_disabled": False,
        "compile_warning_suppressed": False,
        "compile_gpu_sms": None,
    }
    if device.type != "cuda":
        return active

    try:
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    except Exception:
        return active

    active["compile_gpu_sms"] = sm_count
    if sm_count >= _SMALL_GPU_MAX_AUTOTUNE_MIN_SMS:
        return active

    active["compile_small_gpu"] = True
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

    cfg = getattr(getattr(torch, "_inductor", None), "config", None)
    if cfg is not None:
        for name in ("max_autotune", "max_autotune_gemm"):
            if hasattr(cfg, name):
                setattr(cfg, name, False)
                active["compile_max_autotune_disabled"] = True

    global _inductor_warning_filter_installed
    if not _inductor_warning_filter_installed:
        logging.getLogger("torch._inductor.utils").addFilter(_SmallGpuMaxAutotuneWarningFilter())
        _inductor_warning_filter_installed = True
    active["compile_warning_suppressed"] = True
    return active


def _attention_token_counts(cfg: TrainConfig) -> dict[str, int]:
    side = int(cfg.image_size)
    if str(cfg.mode) == "latent":
        side = side // int(cfg.latent_downsample_factor)
    resolutions = sorted(set(int(r) for r in cfg.attn_resolutions), reverse=True)
    return {str(r): r * r for r in resolutions if r > 0 and r <= side}


def _validate_resume_compatibility(cfg: TrainConfig, model_cfg: TrainConfig) -> None:
    fields = (
        "mode",
        "latent_channels",
        "image_size",
        "self_conditioning",
        "use_text_conditioning",
    )
    mismatches = [
        f"{name}: ckpt={getattr(model_cfg, name)!r}, config={getattr(cfg, name)!r}"
        for name in fields
        if getattr(model_cfg, name) != getattr(cfg, name)
    ]
    if mismatches:
        raise RuntimeError("resume config mismatch: " + "; ".join(mismatches))


def _resolve_latent_cache_dir(cfg: TrainConfig) -> Path:
    return Path(cfg.data_root) / str(cfg.latent_cache_dir)


def _resolve_latent_shard_index_path(cfg: TrainConfig) -> Path:
    index_path = Path(str(cfg.latent_cache_index))
    if index_path.is_absolute():
        return index_path
    return _resolve_latent_cache_dir(cfg) / index_path


def _load_resume_checkpoint(path: str) -> dict:
    return load_ckpt(path, torch.device("cpu"))


def dry_run(cfg: TrainConfig) -> None:
    if cfg.architecture != "mmdit_rf":
        raise RuntimeError("Only architecture=mmdit_rf is supported.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = DataConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        images_only=False,
        use_text_conditioning=True,
        min_tag_count=int(cfg.min_tag_count),
        require_512=bool(cfg.require_512),
        val_ratio=float(cfg.val_ratio),
        seed=int(cfg.seed),
        cache_dir=str(cfg.cache_dir),
        failed_list=str(cfg.failed_list),
    )
    train_entries, val_entries = build_or_load_index(dcfg)
    if int(cfg.dataset_limit) > 0:
        train_entries = train_entries[: int(cfg.dataset_limit)]
        val_entries = []
    entries = train_entries + val_entries
    _ensure_mmdit_caches_ready(cfg, entries, device)
    mmdit_cfg = MMDiTConfig.from_dict(cfg.to_dict())
    model = MMDiTFlowModel(mmdit_cfg)
    print(
        "[DRY-RUN] "
        f"architecture={cfg.architecture} objective={cfg.objective} mode={cfg.mode} "
        f"train_entries={len(train_entries)} val_entries={len(val_entries)} "
        f"params={_count_params(model)}",
        flush=True,
    )


def run(cfg: TrainConfig) -> None:
    if cfg.architecture != "mmdit_rf":
        raise RuntimeError("Only architecture=mmdit_rf is supported.")

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
    perf_active["triton_python_cuda_include_patch"] = (
        patch_triton_cuda_python_include_order() if bool(cfg.compile) and device.type == "cuda" else False
    )
    _run_mmdit_rf(cfg, device=device, perf_active=perf_active)
