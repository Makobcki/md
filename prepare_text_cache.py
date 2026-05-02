from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from config.train import TrainConfig
from data_loader import DataConfig, build_or_load_index
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
    if device.type == "cpu" and args_dtype is None:
        return torch.float32
    return _dtype(str(args_dtype or cfg_dtype))


def _entry_text(entry: dict[str, Any], caption_field: str) -> str:
    caption = str(entry.get("caption", "") or entry.get(caption_field, "") or "")
    if caption:
        return caption
    tags = list(entry.get("tags_primary", [])) + list(entry.get("tags_gender", []))
    return " ".join(str(x) for x in tags)


def _dataset_hash(entries: list[dict[str, Any]], caption_field: str) -> str:
    h = hashlib.sha256()
    for entry in entries:
        h.update(str(entry.get("md5", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(_entry_text(entry, caption_field).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _save_shard(path: Path, payload: dict[str, torch.Tensor]) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("scripts/prepare_text_cache.py requires safetensors to be installed.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file(payload, str(tmp))
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
) -> None:
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
    if limit is None and int(cfg.dataset_limit) > 0:
        limit = int(cfg.dataset_limit)
    if limit is not None:
        entries = entries[:limit]
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
        },
    )

    index_lines: list[str] = []
    shard_tokens: list[torch.Tensor] = []
    shard_masks: list[torch.Tensor] = []
    shard_pooled: list[torch.Tensor] = []
    shard_uncond: list[torch.Tensor] = []
    shard_no = 0

    def flush() -> None:
        nonlocal shard_no, shard_tokens, shard_masks, shard_pooled, shard_uncond
        if not shard_tokens:
            return
        shard_name = f"text_{shard_no:05d}.safetensors"
        _save_shard(
            out_dir / "shards" / shard_name,
            {
                "tokens": torch.cat(shard_tokens, dim=0).cpu(),
                "mask": torch.cat(shard_masks, dim=0).cpu().to(torch.uint8),
                "pooled": torch.cat(shard_pooled, dim=0).cpu(),
                "is_uncond": torch.cat(shard_uncond, dim=0).cpu().to(torch.uint8),
            },
        )
        shard_no += 1
        shard_tokens = []
        shard_masks = []
        shard_pooled = []
        shard_uncond = []

    for start in tqdm(range(0, len(entries), batch_size), desc="text-cache"):
        batch_entries = entries[start : start + batch_size]
        prompts = [_entry_text(entry, str(cfg.caption_field)) for entry in batch_entries]
        cond = bundle(prompts)
        local_start = sum(x.shape[0] for x in shard_tokens)
        shard_name = f"text_{shard_no:05d}.safetensors"
        for offset, entry in enumerate(batch_entries):
            key = str(entry.get("md5", start + offset))
            index_lines.append(json.dumps({"key": key, "shard": shard_name, "idx": local_start + offset}, ensure_ascii=False))
        shard_tokens.append(cond.tokens.detach().cpu())
        shard_masks.append(cond.mask.detach().cpu())
        shard_pooled.append(cond.pooled.detach().cpu())
        shard_uncond.append((cond.is_uncond if cond.is_uncond is not None else torch.zeros(len(prompts), dtype=torch.bool, device=device)).detach().cpu())
        if sum(x.shape[0] for x in shard_tokens) >= shard_size:
            flush()
    flush()
    (out_dir / "index.jsonl").write_text("\n".join(index_lines) + ("\n" if index_lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_mmdit_rf.yaml")
    parser.add_argument("--out", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default=None)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    out = Path(args.out) if args.out else Path(cfg.data_root) / str(cfg.text_cache_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
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
