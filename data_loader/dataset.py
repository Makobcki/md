from __future__ import annotations

import random
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image

from diffusion.text import BPETokenizer

from .types import LatentCacheMetadata, LatentShardLocation


def load_image_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (512, 512):
            raise RuntimeError(f"Unexpected image size: {im.size}")
        # PIL -> torch float32 in [0,1], CHW
        arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        # [0,1] -> [-1,1]
        return x * 2.0 - 1.0


def latent_cache_path(cache_dir: str | Path, md5: str) -> Path:
    sub_a = md5[:2] if len(md5) >= 2 else "xx"
    sub_b = md5[2:4] if len(md5) >= 4 else "yy"
    return Path(cache_dir) / sub_a / sub_b / f"{md5}.pt"


def latent_shard_index_path(cache_dir: str | Path, index_name: str) -> Path:
    return Path(cache_dir) / index_name


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


class ImageTextDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        tokenizer: Optional[BPETokenizer],
        cond_drop_prob: float,
        token_drop_prob: float = 0.0,
        tag_drop_prob: float = 0.0,
        caption_drop_prob: float = 0.0,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        token_cache_key: Optional[str] = None,
        latent_cache_dir: Optional[str] = None,
        latent_cache_sharded: bool = False,
        latent_cache_index_path: Optional[str] = None,
        latent_dtype: Optional[torch.dtype] = None,
        return_latents: bool = False,
        latent_cache_strict: bool = True,
        latent_cache_fallback: bool = False,
        latent_expected_meta: Optional[LatentCacheMetadata] = None,
        include_is_latent: bool = False,
        latent_missing_log_path: Optional[Path] = None,
        latent_shard_cache_size: int = 2,
    ):
        self.entries = entries
        self.tokenizer = tokenizer
        self.cond_drop_prob = float(cond_drop_prob)
        self.token_drop_prob = float(token_drop_prob)
        self.tag_drop_prob = float(tag_drop_prob)
        self.caption_drop_prob = float(caption_drop_prob)
        self.rng = random.Random(int(seed))
        self.latent_cache_dir = str(latent_cache_dir) if latent_cache_dir else None
        self.latent_cache_sharded = bool(latent_cache_sharded)
        self.latent_cache_index_path = str(latent_cache_index_path) if latent_cache_index_path else None
        self.latent_dtype = latent_dtype
        self.return_latents = bool(return_latents)
        self.latent_cache_strict = bool(latent_cache_strict)
        self.latent_cache_fallback = bool(latent_cache_fallback)
        self.latent_expected_meta = latent_expected_meta
        self.include_is_latent = bool(include_is_latent)
        self.latent_missing_log_path = latent_missing_log_path
        self.latent_shard_cache_size = int(latent_shard_cache_size)
        self.latent_cache_total = len(entries)
        self.latent_cache_hits = 0
        self.latent_cache_missing = 0
        self._token_cache_path: Optional[Path] = None
        self._token_cache_ids: Optional[torch.LongTensor] = None
        self._token_cache_mask: Optional[torch.BoolTensor] = None
        self._token_cache_md5: Optional[list[str]] = None
        self._empty_ids: Optional[torch.LongTensor] = None
        self._empty_mask: Optional[torch.BoolTensor] = None
        self._shard_index: dict[str, LatentShardLocation] | None = None
        self._entry_shard_locations: Optional[list[Optional[LatentShardLocation]]] = None
        self._shard_to_entry_indices: Optional[dict[Path, list[int]]] = None
        self._shard_tensor_cache: "OrderedDict[Path, torch.Tensor]" = OrderedDict()
        self.current_shard_id: Optional[str] = None
        self.current_shard_tensor: Optional[torch.Tensor] = None

        if self.latent_shard_cache_size <= 0:
            raise ValueError("latent_shard_cache_size must be positive.")

        if self.tokenizer is not None:
            empty_ids, empty_mask = self.tokenizer.encode("")
            self._empty_ids = empty_ids
            self._empty_mask = empty_mask

        if (
            self.tokenizer is not None
            and cache_dir
            and token_cache_key
            and self.token_drop_prob <= 0
            and self.tag_drop_prob <= 0
            and self.caption_drop_prob <= 0
        ):
            cache_path = Path(cache_dir) / f"token_cache_{token_cache_key}.pt"
            self._token_cache_path = cache_path
            self._load_or_build_token_cache()

        if self.return_latents:
            if not self.latent_cache_dir:
                raise RuntimeError("latent_cache_dir is required for latent mode.")
            if self.latent_cache_sharded:
                if self.latent_cache_index_path:
                    index_path = Path(self.latent_cache_index_path)
                    if not index_path.is_absolute():
                        index_path = Path(self.latent_cache_dir) / index_path
                else:
                    index_path = latent_shard_index_path(self.latent_cache_dir, "index.jsonl")
                self._shard_index = load_latent_shard_index(index_path)
            filtered = []
            shard_locations: list[Optional[LatentShardLocation]] = []
            for entry in self.entries:
                md5 = entry.get("md5", "")
                if not md5:
                    self._log_missing_latent(md5, "missing_md5")
                    self.latent_cache_missing += 1
                    shard_locations.append(None)
                    continue
                if self.latent_cache_sharded:
                    shard_path = self._shard_index.get(md5) if self._shard_index else None
                    if shard_path is not None and shard_path.shard_path.exists():
                        filtered.append(entry)
                        shard_locations.append(shard_path)
                        self.latent_cache_hits += 1
                    else:
                        self._log_missing_latent(md5, "missing_shard_index")
                        self.latent_cache_missing += 1
                        shard_locations.append(None)
                    continue
                cache_path = latent_cache_path(self.latent_cache_dir, md5)
                if cache_path.exists():
                    filtered.append(entry)
                    self.latent_cache_hits += 1
                else:
                    self._log_missing_latent(md5, f"missing_cache:{cache_path}")
                    self.latent_cache_missing += 1
            if self.latent_cache_sharded:
                if self.latent_cache_strict:
                    kept_locations = [
                        loc for entry, loc in zip(self.entries, shard_locations) if loc is not None
                    ]
                    self.entries = filtered
                    self._entry_shard_locations = kept_locations
                else:
                    self._entry_shard_locations = shard_locations
                shard_to_indices: dict[Path, list[int]] = {}
                for idx, loc in enumerate(self._entry_shard_locations or []):
                    if loc is None:
                        continue
                    shard_to_indices.setdefault(loc.shard_path, []).append(idx)
                self._shard_to_entry_indices = shard_to_indices
            if self.latent_cache_strict:
                self.entries = filtered

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, path: str) -> torch.Tensor:
        return load_image_tensor(path)

    def _load_latent(self, md5: str) -> torch.Tensor:
        if not self.latent_cache_dir:
            raise RuntimeError("latent_cache_dir is required for latent mode.")
        if self.latent_cache_sharded:
            raise RuntimeError("Use _load_latent_sharded for sharded latent cache.")
        cache_path = latent_cache_path(self.latent_cache_dir, md5)
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing latent cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        latent, meta = _parse_latent_payload(payload)
        _validate_latent_meta(
            expected=self.latent_expected_meta,
            actual=meta,
            strict=self.latent_cache_strict,
            cache_path=cache_path,
        )
        if self.latent_dtype is not None:
            latent = latent.to(dtype=self.latent_dtype)
        return latent

    def _load_latent_sharded(self, entry_idx: int) -> torch.Tensor:
        if not self.latent_cache_dir:
            raise RuntimeError("latent_cache_dir is required for latent mode.")
        if self._entry_shard_locations is None:
            raise RuntimeError("Shard index is not loaded.")
        if entry_idx < 0 or entry_idx >= len(self.entries):
            raise IndexError("entry_idx out of range.")
        location = self._entry_shard_locations[entry_idx]
        if location is None:
            raise FileNotFoundError(f"Missing shard entry for idx={entry_idx}")
        shard_tensor = self._get_shard_tensor(location.shard_path)
        if shard_tensor.dim() != 4:
            raise RuntimeError(f"Shard tensor must be 4D, got {tuple(shard_tensor.shape)}")
        if location.index >= shard_tensor.shape[0]:
            raise RuntimeError(
                f"Shard index out of range: {location.index} >= {shard_tensor.shape[0]}"
            )
        return shard_tensor[location.index]

    def _get_shard_tensor(self, shard_path: Path) -> torch.Tensor:
        cached = self._shard_tensor_cache.get(shard_path)
        if cached is not None:
            self._shard_tensor_cache.move_to_end(shard_path)
            self.current_shard_id = shard_path.name
            self.current_shard_tensor = cached
            return cached
        payload = torch.load(shard_path, map_location="cpu")
        if not isinstance(payload, dict) or int(payload.get("format_version", 0)) != 3:
            raise RuntimeError(f"Invalid shard payload format: {shard_path}")
        meta_common = payload.get("meta_common")
        latents = payload.get("latents")
        if not isinstance(meta_common, dict) or not isinstance(latents, torch.Tensor):
            raise RuntimeError(f"Invalid shard payload contents: {shard_path}")
        meta_obj = LatentCacheMetadata(
            vae_pretrained=str(meta_common.get("vae_pretrained", "")),
            scaling_factor=float(meta_common.get("scaling_factor", 0.0)),
            latent_shape=tuple(meta_common.get("latent_shape", ())),
            dtype=str(meta_common.get("dtype", "")),
            format_version=int(meta_common.get("format_version", 3)),
        )
        _validate_latent_meta(
            expected=self.latent_expected_meta,
            actual=meta_obj,
            strict=self.latent_cache_strict,
            cache_path=shard_path,
        )
        if self.latent_dtype is not None and latents.dtype != self.latent_dtype:
            latents = latents.to(dtype=self.latent_dtype)
        latents = latents.contiguous()
        if len(self._shard_tensor_cache) >= self.latent_shard_cache_size:
            self._shard_tensor_cache.popitem(last=False)
        self._shard_tensor_cache[shard_path] = latents
        self.current_shard_id = shard_path.name
        self.current_shard_tensor = latents
        return latents

    def _log_missing_latent(self, md5: str, reason: str) -> None:
        if self.latent_missing_log_path is None:
            return
        self.latent_missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.latent_missing_log_path.open("a", encoding="utf-8") as f:
            f.write(f"{md5}\t{reason}\n")

    def _build_text(self, entry: dict) -> str:
        cap = entry.get("caption", "") or ""
        tags_primary = list(entry.get("tags_primary", []))
        tags_gender = list(entry.get("tags_gender", []))

        if self.caption_drop_prob > 0 and self.rng.random() < self.caption_drop_prob:
            cap = ""
        if self.tag_drop_prob > 0 and self.rng.random() < self.tag_drop_prob:
            tags_primary = []
            tags_gender = []

        if cap:
            ids_cap, mask_cap = self.tokenizer.encode(cap)
            cap_len = int(mask_cap.sum().item()) - 2  # exclude BOS/EOS
            extra_tags = tags_primary[:5] if cap_len < 40 else []
            all_tags = extra_tags + tags_gender
            if all_tags:
                tag_text = " ".join(all_tags).strip()
                return f"{tag_text} {cap}".strip()
            return cap
        if tags_gender:
            return " ".join(tags_gender).strip()
        return ""

    def _load_or_build_token_cache(self) -> None:
        if self._token_cache_path is None or self.tokenizer is None:
            return
        cache_path = self._token_cache_path
        if cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu")
                md5_list = payload.get("md5", [])
                ids = payload.get("ids")
                mask = payload.get("mask")
                if (
                    isinstance(md5_list, list)
                    and ids is not None
                    and mask is not None
                    and len(md5_list) == len(self.entries)
                    and all(md5 == e.get("md5") for md5, e in zip(md5_list, self.entries))
                ):
                    self._token_cache_md5 = md5_list
                    self._token_cache_ids = ids
                    self._token_cache_mask = mask
                    return
            except Exception:
                pass

        ids_list: list[torch.LongTensor] = []
        mask_list: list[torch.BoolTensor] = []
        md5_list = []
        for entry in self.entries:
            text = self._build_text(entry)
            ids, mask = self.tokenizer.encode(text)
            ids_list.append(ids)
            mask_list.append(mask)
            md5_list.append(entry.get("md5", ""))
        ids_tensor = torch.stack(ids_list, dim=0)
        mask_tensor = torch.stack(mask_list, dim=0)
        payload = {
            "md5": md5_list,
            "ids": ids_tensor,
            "mask": mask_tensor,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
        self._token_cache_md5 = md5_list
        self._token_cache_ids = ids_tensor
        self._token_cache_mask = mask_tensor

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        md5 = e.get("md5", "")
        is_latent = False
        if self.return_latents:
            if self.latent_cache_sharded:
                try:
                    img = self._load_latent_sharded(idx)
                    is_latent = True
                except FileNotFoundError:
                    if self.latent_cache_fallback:
                        self._log_missing_latent(md5, "fallback_encode")
                        img = self._load_image(e["img"])
                        is_latent = False
                    else:
                        raise
            else:
                cache_path = latent_cache_path(self.latent_cache_dir or "", md5)
                if cache_path.exists():
                    img = self._load_latent(md5)
                    is_latent = True
                elif self.latent_cache_fallback:
                    self._log_missing_latent(md5, "fallback_encode")
                    img = self._load_image(e["img"])
                    is_latent = False
                else:
                    raise FileNotFoundError(f"Missing latent cache: {cache_path}")
        else:
            img = self._load_image(e["img"])

        if self.tokenizer is None:
            if self.include_is_latent:
                return img, is_latent
            return (img,)

        drop_cond = self.cond_drop_prob > 0 and self.rng.random() < self.cond_drop_prob
        if drop_cond:
            if self._empty_ids is None or self._empty_mask is None:
                empty_ids, empty_mask = self.tokenizer.encode("")
                self._empty_ids = empty_ids
                self._empty_mask = empty_mask
            if self.include_is_latent:
                return img, self._empty_ids, self._empty_mask, is_latent
            return img, self._empty_ids, self._empty_mask

        if self._token_cache_ids is not None and self._token_cache_mask is not None:
            if self.include_is_latent:
                return img, self._token_cache_ids[idx], self._token_cache_mask[idx], is_latent
            return img, self._token_cache_ids[idx], self._token_cache_mask[idx]

        text = self._build_text(e)
        ids, mask = self.tokenizer.encode(text)
        ids, mask = self._apply_token_dropout(ids, mask)
        if self.include_is_latent:
            return img, ids, mask, is_latent
        return img, ids, mask

    def _apply_token_dropout(
        self,
        ids: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        if self.token_drop_prob <= 0:
            return ids, mask
        ids = ids.clone()
        mask = mask.clone()
        max_len = ids.shape[0]
        for i in range(1, max_len - 1):
            if not mask[i]:
                continue
            if self.rng.random() < self.token_drop_prob:
                ids[i] = self.tokenizer.pad_id
                mask[i] = False
        return ids, mask

    def latent_cache_hit_rate(self) -> float | None:
        if not self.return_latents or self.latent_cache_total == 0:
            return None
        return self.latent_cache_hits / max(self.latent_cache_total, 1)

    def shard_to_entry_indices(self) -> Optional[dict[Path, list[int]]]:
        if not self.latent_cache_sharded:
            return None
        if self._shard_to_entry_indices is None:
            return None
        return {path: list(indices) for path, indices in self._shard_to_entry_indices.items()}
