from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from config.train import TrainConfig
from data_loader import DataConfig, build_or_load_index
from data_loader.indexing import resolve_text_fields
from model.text.pretrained import FrozenTextEncoderBundle


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("dtype must be bf16, fp16, or fp32.")


def _resolve_prepare_dtype(args_dtype: str | None, cfg_dtype: str, device: torch.device) -> torch.dtype:
    if args_dtype == "auto":
        args_dtype = None
    if device.type == "cpu" and args_dtype is None:
        return torch.float32
    return _dtype(str(args_dtype or cfg_dtype))


def _entry_text(entry: dict[str, Any], caption_field: str = "") -> str:
    text = str(entry.get("text", "") or entry.get("prompt", "") or entry.get("caption", "") or "")
    if not text and caption_field:
        text = str(entry.get(caption_field, "") or "")
    if text:
        return text
    tags = list(entry.get("tags_primary", [])) + list(entry.get("tags_gender", []))
    return " ".join(str(x) for x in tags)


def _entry_text_hash(entry: dict[str, Any], caption_field: str = "") -> str:
    h = hashlib.sha256()
    h.update(_entry_text(entry, caption_field).encode("utf-8"))
    h.update(b"\0")
    h.update(str(entry.get("text_source", "")).encode("utf-8"))
    h.update(b"\0")
    return h.hexdigest()


def _dataset_hash(entries: list[dict[str, Any]], caption_field: str = "") -> str:
    h = hashlib.sha256()
    for entry in entries:
        h.update(str(entry.get("md5", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(_entry_text_hash(entry, caption_field).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _save_shard(path: Path, payload: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        from safetensors.torch import save_file

        save_file(payload, str(tmp))
    except ImportError:
        torch.save(payload, tmp)
    tmp.replace(path)


@torch.no_grad()
def prepare_text_cache(
    *,
    cfg: TrainConfig,
    out_dir: Path,
    batch_size: int,
    shard_size: int,
    limit: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
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
    entries = train_entries + val_entries
    if limit is None and int(cfg.dataset_limit) > 0:
        limit = int(cfg.dataset_limit)
    if limit is not None:
        entries = entries[:limit]
    if not entries:
        raise RuntimeError(
            "No dataset entries found for text cache. "
            f"Check data_root={cfg.data_root!r}, image_dir={cfg.image_dir!r}, "
            f"meta_dir={cfg.meta_dir!r}, metadata.jsonl, caption_field={cfg.caption_field!r}, "
            f"text_field={cfg.text_field!r}, text_fields={list(cfg.text_fields)!r}, "
            f"and min_tag_count={cfg.min_tag_count}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = FrozenTextEncoderBundle(cfg.to_dict(), device=device, dtype=dtype)
    bundle_meta = bundle.metadata()
    metadata = {
        "encoders": [
            {**encoder, "dtype": str(dtype).replace("torch.", "")}
            for encoder in bundle_meta["encoders"]
        ],
        "text_dim": int(bundle_meta["text_dim"]),
        "pooled_dim": int(bundle_meta["pooled_dim"]),
        "projection": "not_cached",
        "dataset_hash": _dataset_hash(entries, str(cfg.caption_field)),
        "caption_field": str(cfg.caption_field),
        "text_field": str(cfg.text_field),
        "text_fields": list(cfg.text_fields),
        "resolved_text_fields": resolve_text_fields(
            caption_field=str(cfg.caption_field),
            text_field=str(cfg.text_field),
            text_fields=list(cfg.text_fields),
        ),
        "dtype": str(dtype).replace("torch.", ""),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    empty = bundle([""])
    _save_shard(
        out_dir / "empty_prompt.safetensors",
        {
            "tokens": empty.tokens.detach().cpu(),
            "mask": empty.mask.detach().cpu().to(torch.uint8),
            "pooled": empty.pooled.detach().cpu(),
            "is_uncond": torch.ones(1, dtype=torch.uint8),
            **({"token_types": empty.token_types.detach().cpu().to(torch.long)} if empty.token_types is not None else {}),
        },
    )

    index_lines: list[str] = []
    shard_manifests: list[dict[str, Any]] = []
    shard_sample_ids: list[str] = []
    shard_tokens: list[torch.Tensor] = []
    shard_masks: list[torch.Tensor] = []
    shard_pooled: list[torch.Tensor] = []
    shard_uncond: list[torch.Tensor] = []
    shard_token_types: list[torch.Tensor] = []
    shard_no = 0

    def flush() -> None:
        nonlocal shard_no, shard_sample_ids, shard_tokens, shard_masks, shard_pooled, shard_uncond, shard_token_types
        if not shard_tokens:
            return
        shard_name = f"text_{shard_no:05d}.safetensors"
        payload = {
            "tokens": torch.cat(shard_tokens, dim=0).cpu(),
            "mask": torch.cat(shard_masks, dim=0).cpu().to(torch.uint8),
            "pooled": torch.cat(shard_pooled, dim=0).cpu(),
            "is_uncond": torch.cat(shard_uncond, dim=0).cpu().to(torch.uint8),
        }
        if shard_token_types:
            payload["token_types"] = torch.cat(shard_token_types, dim=0).cpu().to(torch.long)
        _save_shard(out_dir / "shards" / shard_name, payload)
        shard_manifests.append(
            {
                "name": shard_name,
                "sample_ids": list(shard_sample_ids),
                "num_samples": int(payload["tokens"].shape[0]),
                "tokens_shape": list(payload["tokens"].shape),
                "mask_shape": list(payload["mask"].shape),
                "pooled_shape": list(payload["pooled"].shape),
                "token_types_shape": list(payload["token_types"].shape) if "token_types" in payload else None,
                "dtype": str(payload["tokens"].dtype).replace("torch.", ""),
                "encoders": metadata["encoders"],
                "max_lengths": [int(e.get("max_length", 0)) for e in metadata["encoders"]],
            }
        )
        shard_no += 1
        shard_sample_ids = []
        shard_tokens = []
        shard_masks = []
        shard_pooled = []
        shard_uncond = []
        shard_token_types = []

    for start in tqdm(range(0, len(entries), batch_size), desc="text-cache"):
        batch_entries = entries[start : start + batch_size]
        prompts = [_entry_text(entry, str(cfg.caption_field)) for entry in batch_entries]
        cond = bundle(prompts)
        local_start = sum(x.shape[0] for x in shard_tokens)
        shard_name = f"text_{shard_no:05d}.safetensors"
        for offset, entry in enumerate(batch_entries):
            key = str(entry.get("md5", start + offset))
            index_lines.append(
                json.dumps(
                    {
                        "key": key,
                        "shard": shard_name,
                        "idx": local_start + offset,
                        "text_hash": _entry_text_hash(entry, str(cfg.caption_field)),
                    },
                    ensure_ascii=False,
                )
            )
            shard_sample_ids.append(key)
        shard_tokens.append(cond.tokens.detach().cpu())
        shard_masks.append(cond.mask.detach().cpu())
        shard_pooled.append(cond.pooled.detach().cpu())
        shard_uncond.append((cond.is_uncond if cond.is_uncond is not None else torch.zeros(len(prompts), dtype=torch.bool, device=device)).detach().cpu())
        if cond.token_types is not None:
            shard_token_types.append(cond.token_types.detach().cpu())
        if sum(x.shape[0] for x in shard_tokens) >= shard_size:
            flush()
    flush()
    (out_dir / "index.jsonl").write_text("\n".join(index_lines) + ("\n" if index_lines else ""), encoding="utf-8")
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(entries),
        "dataset_hash": metadata["dataset_hash"],
        "caption_field": metadata["caption_field"],
        "text_field": metadata["text_field"],
        "text_fields": metadata["text_fields"],
        "resolved_text_fields": metadata["resolved_text_fields"],
        "dtype": metadata["dtype"],
        "text_dim": metadata["text_dim"],
        "pooled_dim": metadata["pooled_dim"],
        "encoders": metadata["encoders"],
        "empty_prompt": {
            "path": "empty_prompt.safetensors",
            "tokens_shape": list(empty.tokens.detach().cpu().shape),
            "mask_shape": list(empty.mask.detach().cpu().shape),
            "pooled_shape": list(empty.pooled.detach().cpu().shape),
            "token_types_shape": list(empty.token_types.detach().cpu().shape) if empty.token_types is not None else None,
        },
        "shards": shard_manifests,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--out", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("auto", "fp32", "bf16", "fp16"), default=None)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    out = Path(args.out) if args.out else Path(cfg.data_root) / str(cfg.text_cache_dir)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    prepare_text_cache(
        cfg=cfg,
        out_dir=out,
        batch_size=int(args.batch_size),
        shard_size=int(args.shard_size),
        limit=args.limit,
        device=device,
        dtype=_resolve_prepare_dtype(args.dtype, str(cfg.latent_dtype), device),
    )


if __name__ == "__main__":
    main()
