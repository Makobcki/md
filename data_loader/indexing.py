from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .types import DataConfig


_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_GENDER_TAG_RE = re.compile(r"^\d+(?:boy|boys|girl|girls)$")
_INDEX_SCHEMA_VERSION = 5


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _first_str(meta: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _metadata_containers(meta: dict) -> list[dict]:
    containers = [meta]
    for key in ("hf_row", "data", "metadata"):
        value = meta.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def resolve_text_fields(*, caption_field: str, text_field: str = "", text_fields: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Return ordered metadata fields used as training text.

    ``caption_field`` is kept for backward compatibility. New configs should use
    ``text_field: prompt`` or ``text_fields: [prompt, caption]`` when training
    directly from prompt metadata instead of generated captions.
    """
    fields: list[str] = []
    if text_field and str(text_field).strip():
        fields.append(str(text_field).strip())
    for item in text_fields or []:
        name = str(item).strip()
        if name:
            fields.append(name)
    if caption_field and str(caption_field).strip():
        fields.append(str(caption_field).strip())
    fields.extend(["caption", "text", "prompt", "description", "title", "tag_string", "tags"])
    deduped: list[str] = []
    seen: set[str] = set()
    for field in fields:
        if field not in seen:
            deduped.append(field)
            seen.add(field)
    return deduped


def _extract_text(meta: dict, fields: list[str] | tuple[str, ...]) -> tuple[str, str]:
    for container in _metadata_containers(meta):
        for key in fields:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), str(key)
            if isinstance(value, list) and value:
                parts = [str(item).strip() for item in value if str(item).strip()]
                if parts:
                    return ", ".join(parts), str(key)
    return "", ""


def _extract_caption(meta: dict, field: str) -> str:
    # Backward-compatible wrapper for older callers/tests.
    text, _ = _extract_text(meta, resolve_text_fields(caption_field=field))
    return text


def _extract_tags_from_metadata(meta: dict) -> list[str]:
    for container in _metadata_containers(meta):
        for key in ("tags_primary", "tags", "tag_string", "tag_string_general", "keywords"):
            value = container.get(key)
            if isinstance(value, list):
                tags = [str(item).strip() for item in value if str(item).strip()]
                if tags:
                    return tags
            if isinstance(value, str) and value.strip():
                sep = "," if "," in value else " "
                tags = [part.strip() for part in value.split(sep) if part.strip()]
                if tags:
                    return tags
    return []


def _extract_tag_count(meta: dict) -> int:
    for container in _metadata_containers(meta):
        for key in ("tag_count", "tags_count"):
            value = container.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return len(_extract_tags_from_metadata(meta))


def _read_metadata_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except Exception:
        return []
    return rows


def _metadata_md5(meta: dict) -> str:
    for container in _metadata_containers(meta):
        value = _first_str(container, ("md5", "sha1", "hash", "id", "image_id"))
        if value:
            return Path(value).stem
        file_value = _first_str(container, ("file_name", "filename", "image", "img", "path", "image_path"))
        if file_value:
            return Path(file_value).stem
    return ""


def _metadata_image_path(root: Path, img_dir: Path, meta: dict, md5: str) -> Optional[Path]:
    candidates: list[Path] = []
    for container in _metadata_containers(meta):
        file_value = _first_str(container, ("file_name", "filename", "image", "img", "path", "image_path"))
        if not file_value:
            continue
        raw = Path(file_value)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(root / raw)
            candidates.append(img_dir / raw.name)
    if md5:
        candidates.extend(sorted(p for p in img_dir.glob(f"{md5}.*") if p.suffix.lower() in _ALLOWED_EXTS))
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.suffix.lower() in _ALLOWED_EXTS and candidate.exists():
            return candidate
    return None


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
    h.update(b"token_cache_v2")
    h.update(b"text_builder_v2")
    h.update(str(caption_field).encode("utf-8"))
    h.update(str(max_len).encode("utf-8"))
    h.update(str(int(lowercase)).encode("utf-8"))
    h.update(str(int(strip_punct)).encode("utf-8"))
    h.update(_hash_file(Path(vocab_path)).encode("utf-8"))
    h.update(_hash_file(Path(merges_path)).encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_metadata(cfg: DataConfig) -> dict:
    return {
        "root": str(Path(cfg.root)),
        "image_dir": str(cfg.image_dir),
        "meta_dir": str(cfg.meta_dir),
        "tags_dir": str(cfg.tags_dir),
        "caption_field": str(cfg.caption_field),
        "text_field": str(cfg.text_field or ""),
        "text_fields": list(cfg.text_fields or []),
        "resolved_text_fields": resolve_text_fields(
            caption_field=str(cfg.caption_field),
            text_field=str(cfg.text_field or ""),
            text_fields=list(cfg.text_fields or []),
        ),
        "images_only": bool(cfg.images_only),
        "use_text_conditioning": bool(cfg.use_text_conditioning),
        "min_tag_count": int(cfg.min_tag_count),
        "require_512": bool(cfg.require_512),
        "val_ratio": float(cfg.val_ratio),
    }


def _cached_entry_paths_exist(entries: list[dict], *, max_checks: int = 16) -> bool:
    if not entries:
        return True
    checked = 0
    for entry in entries:
        img = entry.get("img")
        if not isinstance(img, str) or not img:
            return False
        if not Path(img).exists():
            return False
        checked += 1
        if checked >= max_checks:
            return True
    return True


def _load_index_cache(cache_path: Path, expected_meta: Optional[dict] = None) -> Optional[Tuple[List[dict], List[dict]]]:
    if not cache_path.exists():
        return None
    train_entries: List[dict] = []
    val_entries: List[dict] = []
    saw_meta = False
    saw_done = False
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "meta":
                    if int(obj.get("schema_version", 0)) != _INDEX_SCHEMA_VERSION:
                        return None
                    if expected_meta is not None and obj.get("config") != expected_meta:
                        return None
                    saw_meta = True
                    continue
                if obj.get("type") == "done":
                    saw_done = True
                    continue
                if obj.get("split") == "val":
                    val_entries.append(obj["entry"])
                else:
                    train_entries.append(obj["entry"])
    except Exception:
        return None
    if not saw_meta or not saw_done:
        return None
    if not _cached_entry_paths_exist(train_entries) or not _cached_entry_paths_exist(val_entries):
        return None
    return train_entries, val_entries


def _write_index_cache_atomic(cache_path: Path, rows: list[dict], *, metadata: dict) -> None:
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "schema_version": _INDEX_SCHEMA_VERSION, "config": metadata}, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps({"type": "done", "schema_version": _INDEX_SCHEMA_VERSION}, ensure_ascii=False) + "\n")
    tmp_path.replace(cache_path)


def build_or_load_index(cfg: DataConfig) -> Tuple[List[dict], List[dict]]:
    """
    Возвращает (train_entries, val_entries).
    entry: {"md5":..., "img":..., "text":..., "text_source":..., "caption":..., "tags_primary":..., "tags_gender":...}
    """
    root = Path(cfg.root)
    img_dir = root / cfg.image_dir
    meta_dir = root / cfg.meta_dir
    tags_dir = root / cfg.tags_dir
    cache_dir = root / cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed = _load_failed_list(root / cfg.failed_list)
    metadata = _cache_metadata(cfg)

    if cfg.images_only:
        cache_key = (
            f"index_images_only_req512{int(cfg.require_512)}"
            f"_val{cfg.val_ratio}_imgdir{cfg.image_dir}.jsonl"
        )
        cache_path = cache_dir / cache_key

        cached = _load_index_cache(cache_path, metadata)
        if cached is not None:
            train_entries, val_entries = cached
            return train_entries, val_entries

        train_entries = []
        val_entries = []
        rows = []
        img_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _ALLOWED_EXTS)
        for img_path in img_files:
            md5 = img_path.stem
            if not md5:
                continue
            if md5 in failed:
                continue
            try:
                with Image.open(img_path) as im:
                    width, height = im.size
                    if cfg.require_512 and (width, height) != (512, 512):
                        continue
            except Exception:
                continue

            entry = {
                "md5": md5,
                "img": str(img_path),
                "text": "",
                "text_source": "",
                "caption": "",
                "tags_primary": [],
                "tags_gender": [],
                "width": int(width),
                "height": int(height),
            }
            split = "val" if _split_is_val(md5, cfg.val_ratio) else "train"
            if split == "val":
                val_entries.append(entry)
            else:
                train_entries.append(entry)
            rows.append({"split": split, "entry": entry})

        _write_index_cache_atomic(cache_path, rows, metadata=metadata)
        return train_entries, val_entries

    resolved_text_fields = resolve_text_fields(
        caption_field=str(cfg.caption_field),
        text_field=str(cfg.text_field or ""),
        text_fields=list(cfg.text_fields or []),
    )
    text_key_hash = hashlib.sha1("\0".join(resolved_text_fields).encode("utf-8")).hexdigest()[:10]
    cache_key = (
        f"index_textfields{text_key_hash}_tags{cfg.min_tag_count}"
        f"_req512{int(cfg.require_512)}_val{cfg.val_ratio}_tagsdir{cfg.tags_dir}"
        f"_text{int(cfg.use_text_conditioning)}.jsonl"
    )
    cache_path = cache_dir / cache_key

    cached = _load_index_cache(cache_path, metadata)
    if cached is not None:
        train_entries, val_entries = cached
        return train_entries, val_entries

    train_entries = []
    val_entries = []
    rows = []

    meta_files = sorted(meta_dir.glob("*.json")) if meta_dir.exists() else []
    metadata_rows = _read_metadata_jsonl(root / "metadata.jsonl") if not meta_files else []

    if meta_files:
        metadata_iter = (_read_json(mp) for mp in meta_files)
    else:
        metadata_iter = iter(metadata_rows)

    for meta in metadata_iter:
        if not meta:
            continue

        md5 = _metadata_md5(meta)
        if not isinstance(md5, str) or len(md5) < 1:
            continue

        if md5 in failed:
            continue

        if _extract_tag_count(meta) < int(cfg.min_tag_count):
            continue

        text_value, text_source = (
            _extract_text(meta, resolved_text_fields) if cfg.use_text_conditioning else ("", "")
        )
        cap = text_value.strip()
        metadata_tags = _extract_tags_from_metadata(meta)
        tags_primary: list[str] = []
        tags_gender: list[str] = []
        if cfg.use_text_conditioning:
            tags_path = tags_dir / f"{md5}.txt"
            if tags_dir.exists() and tags_path.exists():
                tag_data = _read_tags_file(tags_path)
                if tag_data:
                    tags_primary, tags_gender = tag_data
            elif metadata_tags:
                tags_primary = metadata_tags
            if not cap and not tags_primary and not tags_gender:
                continue

        img_path = _metadata_image_path(root, img_dir, meta, md5)
        if img_path is None:
            continue

        try:
            with Image.open(img_path) as im:
                width, height = im.size
                if cfg.require_512 and (width, height) != (512, 512):
                    continue
        except Exception:
            continue

        entry = {
            "md5": md5,
            "img": str(img_path),
            "text": cap,
            "text_source": text_source,
            "caption": cap,
            "tags_primary": tags_primary,
            "tags_gender": tags_gender,
            "width": int(width),
            "height": int(height),
        }
        split = "val" if _split_is_val(md5, cfg.val_ratio) else "train"
        if split == "val":
            val_entries.append(entry)
        else:
            train_entries.append(entry)
        rows.append({"split": split, "entry": entry})

    _write_index_cache_atomic(cache_path, rows, metadata=metadata)
    return train_entries, val_entries
