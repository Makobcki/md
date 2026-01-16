from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import DataConfig


_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_GENDER_TAG_RE = re.compile(r"^\d+(?:boy|boys|girl|girls)$")


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


def build_or_load_index(cfg: DataConfig) -> Tuple[List[dict], List[dict]]:
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
            f"index_images_only_req512{int(cfg.require_512)}"
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
        f"index_{cfg.caption_field}_tags{cfg.min_tag_count}"
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
