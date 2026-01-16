from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch


def latent_cache_path(cache_dir: str | Path, md5: str) -> Path:
    sub_a = md5[:2] if len(md5) >= 2 else "xx"
    sub_b = md5[2:4] if len(md5) >= 4 else "yy"
    return Path(cache_dir) / sub_a / sub_b / f"{md5}.pt"


def latent_shard_index_path(cache_dir: str | Path, index_name: str) -> Path:
    return Path(cache_dir) / index_name


@dataclass(frozen=True)
class LatentShardLocation:
    shard_path: Path
    index: int


def load_latent_shard_index(path: Path) -> dict[str, LatentShardLocation]:
    if not path.exists():
        raise FileNotFoundError(f"Missing shard index: {path}")
    index: dict[str, LatentShardLocation] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        md5 = obj.get("md5")
        shard = obj.get("shard")
        idx = obj.get("idx")
        if not isinstance(md5, str) or not isinstance(shard, str) or not isinstance(idx, int):
            raise ValueError(f"Invalid shard index entry at line {line_no}: {line}")
        if idx < 0:
            raise ValueError(f"Shard index entry has negative idx at line {line_no}: {line}")
        if md5 in index:
            raise ValueError(f"Duplicate md5 in shard index at line {line_no}: {md5}")
        index[md5] = LatentShardLocation(path.parent / "shards" / shard, idx)
    return index


@dataclass(frozen=True)
class LatentCacheMetadata:
    # Метаданные кеша латентов для проверки совместимости.
    vae_pretrained: str
    scaling_factor: float
    latent_shape: Tuple[int, int, int]
    dtype: str
    format_version: int = 1


def _parse_latent_payload(payload: Any) -> Tuple[torch.Tensor, Optional[LatentCacheMetadata]]:
    if isinstance(payload, torch.Tensor):
        return payload, None
    if isinstance(payload, dict):
        latent = payload.get("latent")
        meta = payload.get("meta")
        if isinstance(latent, torch.Tensor) and isinstance(meta, dict):
            meta_obj = LatentCacheMetadata(
                vae_pretrained=str(meta.get("vae_pretrained", "")),
                scaling_factor=float(meta.get("scaling_factor", 0.0)),
                latent_shape=tuple(meta.get("latent_shape", ())),
                dtype=str(meta.get("dtype", "")),
                format_version=int(meta.get("format_version", 1)),
            )
            return latent, meta_obj
    raise RuntimeError("Invalid latent cache payload.")


def _validate_latent_meta(
    *,
    expected: Optional[LatentCacheMetadata],
    actual: Optional[LatentCacheMetadata],
    strict: bool,
    cache_path: Path,
) -> None:
    if expected is None:
        return
    if actual is None:
        if strict:
            raise RuntimeError(f"Missing metadata in latent cache: {cache_path}")
        return
    if expected.vae_pretrained and actual.vae_pretrained != expected.vae_pretrained:
        msg = (
            "Latent cache VAE mismatch: "
            f"{actual.vae_pretrained} != {expected.vae_pretrained} ({cache_path})"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
    if abs(actual.scaling_factor - expected.scaling_factor) > 1e-6:
        msg = (
            "Latent cache scaling_factor mismatch: "
            f"{actual.scaling_factor} != {expected.scaling_factor} ({cache_path})"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
    if tuple(actual.latent_shape) != tuple(expected.latent_shape):
        msg = (
            "Latent cache shape mismatch: "
            f"{actual.latent_shape} != {expected.latent_shape} ({cache_path})"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
    if expected.dtype and actual.dtype != expected.dtype:
        msg = f"Latent cache dtype mismatch: {actual.dtype} != {expected.dtype} ({cache_path})"
        if strict:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}")
