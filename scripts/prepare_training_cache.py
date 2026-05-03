from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from config.train import TrainConfig
from data_loader import DataConfig, build_or_load_index
from data_loader.dataset import latent_cache_path, load_latent_shard_index
from scripts.prepare_text_cache import _dataset_hash as _text_dataset_hash


def _json_hash(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_entries(cfg: TrainConfig) -> list[dict[str, Any]]:
    dcfg = DataConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        text_field=str(cfg.text_field),
        text_fields=list(cfg.text_fields),
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
    return train_entries + val_entries


def _latent_manifest(cfg: TrainConfig, entries: list[dict[str, Any]]) -> dict[str, Any]:
    root = Path(cfg.data_root) / str(cfg.latent_cache_dir)
    if bool(cfg.latent_cache_sharded):
        index_path = Path(str(cfg.latent_cache_index))
        if not index_path.is_absolute():
            index_path = root / index_path
        shards: list[dict[str, Any]] = []
        if index_path.exists():
            index = load_latent_shard_index(index_path)
            shard_to_ids: dict[str, list[str]] = {}
            for key, loc in index.items():
                shard_to_ids.setdefault(loc.shard_path.name, []).append(key)
            for name, ids in sorted(shard_to_ids.items()):
                path = root / "shards" / name
                shards.append({"name": name, "sample_ids": sorted(ids), "exists": path.exists()})
        return {"mode": "sharded", "root": str(root), "index": str(index_path), "shards": shards}

    files = []
    for entry in entries:
        md5 = str(entry.get("md5", ""))
        path = latent_cache_path(root, md5)
        files.append({"sample_id": md5, "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path), "exists": path.exists()})
    return {"mode": "files", "root": str(root), "files": files}



def _read_json_strict(path: Path, *, label: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Broken {label}: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Broken {label}: {path}: expected JSON object")
    return data


def _validate_text_cache_for_repair(cfg: TrainConfig, entries: list[dict[str, Any]], *, force: bool, rebuild: bool) -> str:
    root = Path(cfg.data_root) / str(cfg.text_cache_dir)
    current_hash = _text_dataset_hash(entries, str(cfg.caption_field)) if entries else ""
    if not root.exists():
        return "missing"
    try:
        metadata = _read_json_strict(root / "metadata.json", label="text cache metadata")
        manifest = _read_json_strict(root / "manifest.json", label="text cache manifest")
    except RuntimeError:
        if force or rebuild:
            return "broken_metadata"
        raise
    for required in ("dataset_hash", "text_dim", "pooled_dim", "encoders"):
        if required not in metadata:
            if force or rebuild:
                return "broken_metadata"
            raise RuntimeError(f"Broken text cache metadata: missing required field {required!r}")
    cached_hash = str(metadata.get("dataset_hash", manifest.get("dataset_hash", "")))
    if cached_hash and current_hash and cached_hash != current_hash:
        if not rebuild:
            raise RuntimeError(
                "Text cache dataset changed. Use --rebuild to regenerate cache for the current dataset.\n"
                f"cache: {cached_hash}\ncurrent: {current_hash}"
            )
        return "changed_dataset"
    try:
        from model.text.cache import TextCache

        TextCache(root, shard_cache_size=int(cfg.text_shard_cache_size)).validate_files_readable()
    except Exception as exc:
        msg = str(exc).lower()
        if "missing text cache tensor file" in msg or "missing empty prompt" in msg or "missing text cache index" in msg:
            return "missing_shard"
        if force or rebuild:
            return "broken_metadata"
        raise RuntimeError(f"Text cache validation failed: {exc}") from exc
    return "ready"


def _validate_latent_cache_for_repair(cfg: TrainConfig, entries: list[dict[str, Any]], *, force: bool, rebuild: bool) -> str:
    root = Path(cfg.data_root) / str(cfg.latent_cache_dir)
    if not root.exists():
        return "missing"
    if bool(cfg.latent_cache_sharded):
        index_path = Path(str(cfg.latent_cache_index))
        if not index_path.is_absolute():
            index_path = root / index_path
        if not index_path.exists():
            return "missing_shard"
        try:
            index = load_latent_shard_index(index_path)
        except Exception as exc:
            if force or rebuild:
                return "broken_metadata"
            raise RuntimeError(f"Broken latent shard index: {index_path}: {exc}") from exc
        for loc in index.values():
            if not loc.shard_path.exists():
                return "missing_shard"
        return "ready"
    missing = []
    for entry in entries:
        key = str(entry.get("md5", ""))
        if key and not latent_cache_path(root, key).exists():
            missing.append(key)
            if len(missing) >= 3:
                break
    return "missing_shard" if missing else "ready"


def _delete_cache_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)

def build_training_cache_manifest(cfg: TrainConfig, *, entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    entries = _dataset_entries(cfg) if entries is None else entries
    latent_side = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    text_root = Path(cfg.data_root) / str(cfg.text_cache_dir)
    text_manifest = _load_json(text_root / "manifest.json")
    text_metadata = _load_json(text_root / "metadata.json")
    dataset_hash = _text_dataset_hash(entries, str(cfg.caption_field)) if entries else ""
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": _json_hash(cfg.to_dict()),
        "dataset_hash": dataset_hash,
        "num_samples": len(entries),
        "image_size": int(cfg.image_size),
        "latent_shape": [int(cfg.latent_channels), latent_side, latent_side],
        "vae": {
            "pretrained": str(cfg.vae_pretrained),
            "scaling_factor": float(cfg.vae_scaling_factor),
        },
        "text": {
            "root": str(text_root),
            "metadata": text_metadata,
            "manifest": text_manifest,
            "encoders": text_metadata.get("encoders", text_manifest.get("encoders", [])),
        },
        "shards": {
            "latents": _latent_manifest(cfg, entries),
            "text": text_manifest.get("shards", []),
        },
    }


def write_training_cache_manifest(cfg: TrainConfig, *, entries: list[dict[str, Any]] | None = None) -> Path:
    cache_root = Path(cfg.data_root) / str(cfg.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = build_training_cache_manifest(cfg, entries=entries)
    path = cache_root / "training_cache_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare MMDiT RF text and latent caches.")
    ap.add_argument("--config", default="config/train.yaml")
    ap.add_argument("--text-batch-size", type=int, default=8)
    ap.add_argument("--text-shard-size", type=int, default=1024)
    ap.add_argument("--text-limit", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--text-dtype", default="auto", choices=("auto", "fp32", "bf16", "fp16"))
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--skip-latents", action="store_true")
    ap.add_argument("--overwrite-latents", action="store_true")
    ap.add_argument("--repair", action="store_true", help="Validate cache and regenerate missing shards without ignoring metadata problems.")
    ap.add_argument("--force", action="store_true", help="Allow repair to replace broken metadata/cache files.")
    ap.add_argument("--rebuild", action="store_true", help="Delete stale cache and rebuild it for the current dataset.")
    args = ap.parse_args()

    from scripts.prepare_latents import prepare_latent_cache_for_config
    from scripts.prepare_text_cache import _resolve_prepare_dtype, prepare_text_cache

    cfg = TrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    entries = _dataset_entries(cfg)
    text_root = Path(cfg.data_root) / str(cfg.text_cache_dir)
    latent_root = Path(cfg.data_root) / str(cfg.latent_cache_dir)

    if args.rebuild:
        if not args.skip_text:
            _delete_cache_path(text_root)
        if not args.skip_latents:
            _delete_cache_path(latent_root)

    if args.repair:
        if not args.skip_text:
            text_state = _validate_text_cache_for_repair(cfg, entries, force=bool(args.force), rebuild=bool(args.rebuild))
            if text_state in {"broken_metadata", "changed_dataset"}:
                _delete_cache_path(text_root)
            elif text_state == "ready":
                print(f"[OK] text cache ready: {text_root}")
            else:
                print(f"[INFO] repairing text cache ({text_state}): {text_root}")
        if not args.skip_latents:
            latent_state = _validate_latent_cache_for_repair(cfg, entries, force=bool(args.force), rebuild=bool(args.rebuild))
            if latent_state == "broken_metadata" and not args.force and not args.rebuild:
                raise RuntimeError("Latent cache metadata is broken. Use --force to replace it or --rebuild to start over.")
            if latent_state in {"broken_metadata", "changed_dataset"}:
                _delete_cache_path(latent_root)
            elif latent_state == "ready":
                print(f"[OK] latent cache ready: {latent_root}")
            else:
                print(f"[INFO] repairing latent cache ({latent_state}): {latent_root}")

    if not args.skip_text:
        prepare_text_cache(
            cfg=cfg,
            out_dir=text_root,
            batch_size=int(args.text_batch_size),
            shard_size=int(args.text_shard_size),
            limit=args.text_limit,
            device=device,
            dtype=_resolve_prepare_dtype(args.text_dtype, str(cfg.latent_dtype), device),
        )
    if not args.skip_latents:
        prepare_latent_cache_for_config(cfg, overwrite=bool(args.overwrite_latents or args.repair or args.rebuild))
    manifest_path = write_training_cache_manifest(cfg, entries=entries)
    print(f"[OK] wrote training cache manifest: {manifest_path}")


if __name__ == "__main__":
    main()
