from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from diffusion.text import BPETokenizer

# ----------------------------
# Configs
# ----------------------------
# ----------------------------
# Configs
# ----------------------------

@dataclass(frozen=True)
class DanbooruConfig:
    root: str  # ./data/raw/Danbooru
    image_dir: str = "image_512"
    meta_dir: str = "meta"
    tags_dir: str = "tags"
    caption_field: str = "caption_llava_34b_no_tags_short"
    images_only: bool = False
    use_text_conditioning: bool = True
    min_tag_count: int = 8          # danbooru_post.tag_count >= min_tag_count
    require_512: bool = True        # пропускать всё, что не 512x512
    val_ratio: float = 0.01         # 99/1
    seed: int = 42
    cache_dir: str = ".cache"       # внутри root
    failed_list: str = "failed/md5.txt"


_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_GENDER_TAG_RE = re.compile(r"^\d+(?:boy|boys|girl|girls)$")


# ----------------------------
# Index building / caching
# ----------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_caption(meta: dict, field: str) -> str:
    hf = meta.get("hf_row")
    if isinstance(hf, dict) and field in hf and isinstance(hf[field], str):
        return hf[field]
    return ""


def _extract_tag_count(meta: dict) -> int:
    dp = meta.get("danbooru_post")
    if isinstance(dp, dict):
        v = dp.get("tag_count")
        if isinstance(v, int):
            return v
    hf = meta.get("hf_row")
    if isinstance(hf, dict):
        v = hf.get("tag_count")
        if isinstance(v, int):
            return v
    return 0


def _split_is_val(md5: str, val_ratio: float) -> bool:
    h = hashlib.sha1(md5.encode("utf-8")).hexdigest()
    r = np.float64(np.uint32(int(h[:8], 16))) / np.float64(0xFFFFFFFF)
    return bool(r < np.float64(val_ratio))


def _load_failed_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _read_tags_file(path: Path) -> Optional[tuple[list[str], list[str]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip() or line == ""]
    if not lines:
        return None
    first = lines[0] if len(lines) > 0 else ""
    second = lines[1] if len(lines) > 1 else ""
    primary = [t.strip() for t in first.split(",") if t.strip()]
    gender = [t for t in second.split() if _GENDER_TAG_RE.match(t)]
    return primary, gender


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_token_cache_key(
    *,
    vocab_path: str | Path,
    merges_path: str | Path,
    caption_field: str,
    max_len: int,
    lowercase: bool,
    strip_punct: bool,
) -> str:
    h = hashlib.sha1()
    h.update(b"token_cache_v1")
    h.update(str(caption_field).encode("utf-8"))
    h.update(str(max_len).encode("utf-8"))
    h.update(str(int(lowercase)).encode("utf-8"))
    h.update(str(int(strip_punct)).encode("utf-8"))
    h.update(_hash_file(Path(vocab_path)).encode("utf-8"))
    h.update(_hash_file(Path(merges_path)).encode("utf-8"))
    return h.hexdigest()[:16]


def build_or_load_index(cfg: DanbooruConfig) -> Tuple[List[dict], List[dict]]:
    """
    Возвращает (train_entries, val_entries).
    entry: {"md5":..., "img":..., "caption":..., "tags_primary":..., "tags_gender":...}
    """
    root = Path(cfg.root)
    img_dir = root / cfg.image_dir
    meta_dir = root / cfg.meta_dir
    tags_dir = root / cfg.tags_dir
    cache_dir = root / cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed = _load_failed_list(root / cfg.failed_list)

    if cfg.images_only:
        cache_key = (
            f"danbooru_index_images_only_req512{int(cfg.require_512)}"
            f"_val{cfg.val_ratio}_imgdir{cfg.image_dir}.jsonl"
        )
        cache_path = cache_dir / cache_key

        if cache_path.exists():
            train_entries = []
            val_entries = []
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("split") == "val":
                        val_entries.append(obj["entry"])
                    else:
                        train_entries.append(obj["entry"])
            return train_entries, val_entries

        train_entries = []
        val_entries = []
        img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _ALLOWED_EXTS)
        for img_path in img_files:
            md5 = img_path.stem
            if not md5:
                continue
            if md5 in failed:
                continue
            if cfg.require_512:
                try:
                    with Image.open(img_path) as im:
                        if im.size != (512, 512):
                            continue
                except Exception:
                    continue

            entry = {
                "md5": md5,
                "img": str(img_path),
                "caption": "",
                "tags_primary": [],
                "tags_gender": [],
            }
            split = "val" if _split_is_val(md5, cfg.val_ratio) else "train"
            if split == "val":
                val_entries.append(entry)
            else:
                train_entries.append(entry)

            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"split": split, "entry": entry}, ensure_ascii=False) + "\n")

        return train_entries, val_entries

    cache_key = (
        f"danbooru_index_{cfg.caption_field}_tags{cfg.min_tag_count}"
        f"_req512{int(cfg.require_512)}_val{cfg.val_ratio}_tagsdir{cfg.tags_dir}"
        f"_text{int(cfg.use_text_conditioning)}.jsonl"
    )
    cache_path = cache_dir / cache_key

    if cache_path.exists():
        train_entries: List[dict] = []
        val_entries: List[dict] = []
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("split") == "val":
                    val_entries.append(obj["entry"])
                else:
                    train_entries.append(obj["entry"])
        return train_entries, val_entries

    train_entries = []
    val_entries = []

    meta_files = sorted(meta_dir.glob("*.json"))
    for mp in meta_files:
        meta = _read_json(mp)
        if not meta:
            continue

        md5 = meta.get("md5")
        if not isinstance(md5, str) or len(md5) < 6:
            continue

        if md5 in failed:
            continue

        if _extract_tag_count(meta) < int(cfg.min_tag_count):
            continue

        if cfg.use_text_conditioning:
            cap = _extract_caption(meta, cfg.caption_field).strip()
            if not cap:
                continue
        else:
            cap = ""

        candidates = [p for p in img_dir.glob(f"{md5}.*") if p.suffix.lower() in _ALLOWED_EXTS]
        if not candidates:
            continue
        img_path = candidates[0]

        if cfg.use_text_conditioning:
            tags_path = tags_dir / f"{md5}.txt"
            if not tags_path.exists():
                continue
            tag_data = _read_tags_file(tags_path)
            if not tag_data:
                continue
            tags_primary, tags_gender = tag_data
        else:
            tags_primary, tags_gender = [], []

        if cfg.require_512:
            try:
                with Image.open(img_path) as im:
                    if im.size != (512, 512):
                        continue
            except Exception:
                continue

        entry = {
            "md5": md5,
            "img": str(img_path),
            "caption": cap,
            "tags_primary": tags_primary,
            "tags_gender": tags_gender,
        }
        split = "val" if _split_is_val(md5, cfg.val_ratio) else "train"
        if split == "val":
            val_entries.append(entry)
        else:
            train_entries.append(entry)

        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"split": split, "entry": entry}, ensure_ascii=False) + "\n")

    return train_entries, val_entries


# ----------------------------
# Dataset
# ----------------------------

def load_image_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (512, 512):
            raise RuntimeError(f"Unexpected image size: {im.size}")
        # PIL -> torch float32 in [0,1], CHW
        import numpy as np
        arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        # [0,1] -> [-1,1]
        return x * 2.0 - 1.0


def latent_cache_path(cache_dir: str | Path, md5: str) -> Path:
    sub_a = md5[:2] if len(md5) >= 2 else "xx"
    sub_b = md5[2:4] if len(md5) >= 4 else "yy"
    return Path(cache_dir) / sub_a / sub_b / f"{md5}.pt"


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


class DanbooruDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        tokenizer: Optional[BPETokenizer],
        cond_drop_prob: float,
        token_drop_prob: float = 0.0,
        tag_drop_prob: float = 0.0,
        caption_drop_prob: float = 0.0,
        seed: int,
        cache_dir: Optional[str] = None,
        token_cache_key: Optional[str] = None,
        latent_cache_dir: Optional[str] = None,
        latent_dtype: Optional[torch.dtype] = None,
        return_latents: bool = False,
        latent_cache_strict: bool = True,
        latent_cache_fallback: bool = False,
        latent_expected_meta: Optional[LatentCacheMetadata] = None,
        include_is_latent: bool = False,
        latent_missing_log_path: Optional[Path] = None,
    ):
        self.entries = entries
        self.tokenizer = tokenizer
        self.cond_drop_prob = float(cond_drop_prob)
        self.token_drop_prob = float(token_drop_prob)
        self.tag_drop_prob = float(tag_drop_prob)
        self.caption_drop_prob = float(caption_drop_prob)
        self.rng = random.Random(int(seed))
        self.latent_cache_dir = str(latent_cache_dir) if latent_cache_dir else None
        self.latent_dtype = latent_dtype
        self.return_latents = bool(return_latents)
        self.latent_cache_strict = bool(latent_cache_strict)
        self.latent_cache_fallback = bool(latent_cache_fallback)
        self.latent_expected_meta = latent_expected_meta
        self.include_is_latent = bool(include_is_latent)
        self.latent_missing_log_path = latent_missing_log_path
        self.latent_cache_total = len(entries)
        self.latent_cache_hits = 0
        self.latent_cache_missing = 0
        self._token_cache_path: Optional[Path] = None
        self._token_cache_ids: Optional[torch.LongTensor] = None
        self._token_cache_mask: Optional[torch.BoolTensor] = None
        self._token_cache_md5: Optional[list[str]] = None
        self._empty_ids: Optional[torch.LongTensor] = None
        self._empty_mask: Optional[torch.BoolTensor] = None

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
            filtered = []
            for entry in self.entries:
                md5 = entry.get("md5", "")
                if not md5:
                    self._log_missing_latent(md5, "missing_md5")
                    self.latent_cache_missing += 1
                    continue
                cache_path = latent_cache_path(self.latent_cache_dir, md5)
                if cache_path.exists():
                    filtered.append(entry)
                    self.latent_cache_hits += 1
                else:
                    self._log_missing_latent(md5, f"missing_cache:{cache_path}")
                    self.latent_cache_missing += 1
            if self.latent_cache_strict:
                self.entries = filtered

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, path: str) -> torch.Tensor:
        return load_image_tensor(path)

    def _load_latent(self, md5: str) -> torch.Tensor:
        if not self.latent_cache_dir:
            raise RuntimeError("latent_cache_dir is required for latent mode.")
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

        # classifier-free guidance training: drop text sometimes
        drop_cond = self.cond_drop_prob > 0 and self.rng.random() < self.cond_drop_prob
        cap = e.get("caption", "")
        if drop_cond:
            cap = ""

        if self.tokenizer is None:
            return img, cap

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


def collate_with_tokenizer(
    batch,
    *,
    latent_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    # В режиме fallback элементы могут содержать флаг is_latent.
    if len(batch) == 0:
        raise RuntimeError("Empty batch.")
    item_len = len(batch[0])
    has_tokens = item_len >= 3
    has_flag = item_len >= 4 or (not has_tokens and item_len >= 2)

    if not has_tokens and not has_flag:
        imgs = torch.stack([b[0] for b in batch], dim=0)
        return imgs, None, None

    latents: list[Optional[torch.Tensor]] = [None] * len(batch)
    to_encode: list[torch.Tensor] = []
    encode_indices: list[int] = []

    for idx, item in enumerate(batch):
        if has_tokens:
            x, ids, mask, is_latent = item[:4]
        else:
            x, is_latent = item[:2]
        if is_latent:
            latents[idx] = x
        else:
            to_encode.append(x)
            encode_indices.append(idx)

    if to_encode:
        if latent_encoder is None:
            raise RuntimeError("latent_encoder is required for fallback encoding.")
        encoded = latent_encoder(torch.stack(to_encode, dim=0))
        for idx, z in zip(encode_indices, encoded):
            latents[idx] = z

    if any(x is None for x in latents):
        raise RuntimeError("Failed to assemble latent batch.")

    imgs = torch.stack([x for x in latents if x is not None], dim=0)
    if not has_tokens:
        return imgs, None, None
    ids = torch.stack([b[1] for b in batch], dim=0)
    mask = torch.stack([b[2] for b in batch], dim=0)
    return imgs, ids, mask
