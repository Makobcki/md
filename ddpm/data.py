from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from ddpm.text import BPETokenizer
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
    r = int(h[:8], 16) / 0xFFFFFFFF
    return r < float(val_ratio)


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
    meta_dir = root / cfg.meta_dir
    img_dir = root / cfg.image_dir
    tags_dir = root / cfg.tags_dir
    cache_dir = root / cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed = _load_failed_list(root / cfg.failed_list)

    cache_key = (
        f"danbooru_index_{cfg.caption_field}_tags{cfg.min_tag_count}"
        f"_req512{int(cfg.require_512)}_val{cfg.val_ratio}_tagsdir{cfg.tags_dir}.jsonl"
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

        cap = _extract_caption(meta, cfg.caption_field).strip()
        if not cap:
            continue

        candidates = [p for p in img_dir.glob(f"{md5}.*") if p.suffix.lower() in _ALLOWED_EXTS]
        if not candidates:
            continue
        img_path = candidates[0]

        tags_path = tags_dir / f"{md5}.txt"
        if not tags_path.exists():
            continue
        tag_data = _read_tags_file(tags_path)
        if not tag_data:
            continue
        tags_primary, tags_gender = tag_data

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

class DanbooruDataset(Dataset):
    def __init__(
        self,
        entries: List[dict],
        tokenizer: Optional[BPETokenizer],
        cond_drop_prob: float,
        seed: int,
        cache_dir: Optional[str] = None,
        token_cache_key: Optional[str] = None,
    ):
        self.entries = entries
        self.tokenizer = tokenizer
        self.cond_drop_prob = float(cond_drop_prob)
        self.rng = random.Random(int(seed))
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

        if self.tokenizer is not None and cache_dir and token_cache_key:
            cache_path = Path(cache_dir) / f"token_cache_{token_cache_key}.pt"
            self._token_cache_path = cache_path
            self._load_or_build_token_cache()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, path: str) -> torch.Tensor:
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

    def _build_text(self, entry: dict) -> str:
        cap = entry.get("caption", "") or ""
        tags_primary = list(entry.get("tags_primary", []))
        tags_gender = list(entry.get("tags_gender", []))

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
        img = self._load_image(e["img"])
        cap = e["caption"]

        # classifier-free guidance training: drop text sometimes
        drop_cond = self.cond_drop_prob > 0 and self.rng.random() < self.cond_drop_prob
        if drop_cond:
            cap = ""

        if self.tokenizer is None:
            return img, cap

        if drop_cond:
            if self._empty_ids is None or self._empty_mask is None:
                empty_ids, empty_mask = self.tokenizer.encode("")
                self._empty_ids = empty_ids
                self._empty_mask = empty_mask
            return img, self._empty_ids, self._empty_mask

        if self._token_cache_ids is not None and self._token_cache_mask is not None:
            return img, self._token_cache_ids[idx], self._token_cache_mask[idx]

        text = self._build_text(e)
        ids, mask = self.tokenizer.encode(text)
        return img, ids, mask


def collate_with_tokenizer(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    ids = torch.stack([b[1] for b in batch], dim=0)
    mask = torch.stack([b[2] for b in batch], dim=0)
    return imgs, ids, mask
