#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion.config import TrainConfig
from diffusion.events import EventBus, JsonlFileSink, StdoutJsonSink
from diffusion.data import DanbooruConfig, build_or_load_index, latent_cache_path, load_image_tensor
from diffusion.utils import build_run_metadata
from diffusion.vae import VAEWrapper


def _latent_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("latent_dtype must be 'fp16' or 'bf16'.")


def _init_latent_stats() -> dict:
    return {
        "count": 0,
        "mean": 0.0,
        "m2": 0.0,
        "min": None,
        "max": None,
    }


def _update_latent_stats(stats: dict, batch: torch.Tensor) -> None:
    if batch.numel() == 0:
        return
    values = batch.detach().to(device="cpu", dtype=torch.float32).numpy().ravel()
    batch_count = values.size
    batch_mean = float(np.mean(values, dtype=np.float64))
    batch_var = float(np.var(values, dtype=np.float64))
    batch_m2 = batch_var * batch_count

    stats["min"] = float(np.min(values)) if stats["min"] is None else float(min(stats["min"], float(np.min(values))))
    stats["max"] = float(np.max(values)) if stats["max"] is None else float(max(stats["max"], float(np.max(values))))

    count = int(stats["count"])
    mean = float(stats["mean"])
    m2 = float(stats["m2"])
    if count == 0:
        stats["count"] = batch_count
        stats["mean"] = batch_mean
        stats["m2"] = batch_m2
        return

    total = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * (batch_count / total)
    m2 = m2 + batch_m2 + delta * delta * (count * batch_count / total)

    stats["count"] = total
    stats["mean"] = mean
    stats["m2"] = m2


def _finalize_latent_stats(stats: dict) -> dict:
    count = int(stats["count"])
    if count <= 1:
        std = 0.0
    else:
        std = float(np.sqrt(float(stats["m2"]) / (count - 1)))
    return {
        "count": count,
        "mean": float(stats["mean"]),
        "std": std,
        "min": stats["min"],
        "max": stats["max"],
    }


class _LatentPrepDataset(Dataset):
    def __init__(self, entries: list[dict], limit: Optional[int] = None) -> None:
        self.entries = entries[:limit] if limit is not None else entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        md5 = entry.get("md5", "")
        path = entry.get("img", "")
        try:
            x = load_image_tensor(path)
            return {"md5": md5, "x": x, "error": None}
        except Exception as e:
            return {"md5": md5, "x": None, "error": str(e)}


def _collate_items(batch: list[dict]) -> list[dict]:
    return batch


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/train.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--latent-dtype", default=None, choices=("fp16", "bf16"))
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if cfg.mode != "latent":
        raise RuntimeError("prepare_latents requires mode=latent in config.")
    if not cfg.vae_pretrained:
        raise RuntimeError("vae_pretrained is required to prepare latents.")

    root = Path(cfg.data_root)
    cache_dir = root / (cfg.latent_cache_dir or ".cache/latents")
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed_path = root / "failed_latents.txt"
    run_dir = os.environ.get("WEBUI_RUN_DIR")
    metrics_dir = Path(run_dir) / "metrics" if run_dir else Path(cfg.out_dir) / "metrics"
    metrics_path = metrics_dir / "latent_prepare.jsonl"
    sinks = [JsonlFileSink(metrics_path), StdoutJsonSink()]
    event_bus = EventBus(sinks)

    code_version = build_run_metadata().get("git_commit")

    dcfg = DanbooruConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        images_only=bool(cfg.images_only),
        use_text_conditioning=bool(cfg.use_text_conditioning),
        min_tag_count=int(cfg.min_tag_count),
        require_512=bool(cfg.require_512),
        val_ratio=float(cfg.val_ratio),
        seed=int(cfg.seed),
        cache_dir=str(cfg.cache_dir),
        failed_list=str(cfg.failed_list),
    )
    train_entries, _ = build_or_load_index(dcfg)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    dtype = _latent_dtype(args.latent_dtype or cfg.latent_dtype)
    vae = VAEWrapper(
        pretrained=str(cfg.vae_pretrained),
        freeze=True,
        scaling_factor=float(cfg.vae_scaling_factor),
        device=device,
        dtype=dtype,
    )

    dataset = _LatentPrepDataset(train_entries, limit=args.limit)
    dl = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        shuffle=False,
        collate_fn=_collate_items,
    )

    total = len(dataset)
    saved = 0
    skipped = 0
    errors = 0
    stats = _init_latent_stats()
    start = time.perf_counter()
    last_log = start

    event_bus.emit({"type": "status", "status": "start", "total": total})

    webui_mode = os.environ.get("WEBUI") == "1"
    for batch in tqdm(dl, desc="prepare latents", disable=webui_mode):
        batch_items = []
        for item in batch:
            if item["error"] is not None:
                errors += 1
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{item['md5']}\tload\t{item['error']}\n")
                continue
            md5 = item["md5"]
            out_path = latent_cache_path(cache_dir, md5)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            batch_items.append((md5, out_path, item["x"]))

        if not batch_items:
            continue

        md5s, paths, xs = zip(*batch_items)
        x = torch.stack(xs, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        # load_image_tensor возвращает [-1, 1], это ожидаемый диапазон входа VAE.

        try:
            z_batch = vae.encode(x)
            _update_latent_stats(stats, z_batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("OOM during VAE encode; reduce --batch-size or use fewer workers.") from e
            for md5, path, x_single in batch_items:
                try:
                    z_single = vae.encode(x_single.unsqueeze(0).to(device=device, dtype=dtype)).squeeze(0)
                    _update_latent_stats(stats, z_single.unsqueeze(0))
                    _save_latent(z_single, path, cfg, dtype, code_version)
                    saved += 1
                except Exception as inner:
                    errors += 1
                    with open(failed_path, "a", encoding="utf-8") as f:
                        f.write(f"{md5}\tencode\t{inner}\n")
            continue

        for md5, out_path, z in zip(md5s, paths, z_batch):
            try:
                _save_latent(z, out_path, cfg, dtype, code_version)
                saved += 1
            except Exception as e:
                errors += 1
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{md5}\tsave\t{e}\n")

        now = time.perf_counter()
        if now - last_log >= 5.0:
            processed = saved + skipped + errors
            rate = (saved + skipped) / max(now - start, 1e-9)
            event_bus.emit({
                "type": "metric",
                "processed": processed,
                "saved": saved,
                "skipped": skipped,
                "errors": errors,
                "items_per_sec": rate,
                "max_steps": total,
            })
            last_log = now

    total_time = time.perf_counter() - start
    latent_stats = _finalize_latent_stats(stats)
    event_bus.emit({
        "type": "status",
        "status": "done",
        "saved": saved,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": total_time,
        "latent_stats": latent_stats,
    })


def _save_latent(
    z: torch.Tensor,
    out_path: Path,
    cfg: TrainConfig,
    dtype: torch.dtype,
    code_version: Optional[str],
) -> None:
    if z.dim() != 3:
        raise RuntimeError(f"latent must be 3D, got {tuple(z.shape)}")
    if z.shape[0] != int(cfg.latent_channels):
        raise RuntimeError(f"latent_channels mismatch: {z.shape[0]} != {cfg.latent_channels}")
    h = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    w = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    if z.shape[-2:] != (h, w):
        raise RuntimeError(f"latent spatial mismatch: {tuple(z.shape[-2:])} != {(h, w)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": 1,
        "vae_id": str(cfg.vae_pretrained),
        "vae_pretrained": str(cfg.vae_pretrained),
        "scaling_factor": float(cfg.vae_scaling_factor),
        "latent_shape": list(z.shape),
        "dtype": str(cfg.latent_dtype),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code_version": code_version,
    }
    payload = {
        "latent": z.to(dtype=dtype, device="cpu"),
        "meta": meta,
    }
    torch.save(payload, out_path)


if __name__ == "__main__":
    main()
