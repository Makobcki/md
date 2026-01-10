from __future__ import annotations

import hashlib
import json
import random
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
    caption_field: str = "caption_llava_34b_no_tags_short"
    min_tag_count: int = 8          # danbooru_post.tag_count >= min_tag_count
    require_512: bool = True        # пропускать всё, что не 512x512
    val_ratio: float = 0.01         # 99/1
    seed: int = 42
    cache_dir: str = ".cache"       # внутри root
    failed_list: str = "failed/md5.txt"


_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


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


def build_or_load_index(cfg: DanbooruConfig) -> Tuple[List[dict], List[dict]]:
    """
    Возвращает (train_entries, val_entries).
    entry: {"md5":..., "img":..., "caption":...}
    """
    root = Path(cfg.root)
    meta_dir = root / cfg.meta_dir
    img_dir = root / cfg.image_dir
    cache_dir = root / cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed = _load_failed_list(root / cfg.failed_list)

    cache_key = (
        f"danbooru_index_{cfg.caption_field}_tags{cfg.min_tag_count}"
        f"_req512{int(cfg.require_512)}_val{cfg.val_ratio}.jsonl"
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

        if cfg.require_512:
            try:
                with Image.open(img_path) as im:
                    if im.size != (512, 512):
                        continue
            except Exception:
                continue

        entry = {"md5": md5, "img": str(img_path), "caption": cap}
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
    ):
        self.entries = entries
        self.tokenizer = tokenizer
        self.cond_drop_prob = float(cond_drop_prob)
        self.rng = random.Random(int(seed))

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

    def __getitem__(self, idx: int):
        e = self.entries[idx]
        img = self._load_image(e["img"])
        cap = e["caption"]

        # classifier-free guidance training: drop text sometimes
        if self.cond_drop_prob > 0 and self.rng.random() < self.cond_drop_prob:
            cap = ""

        if self.tokenizer is None:
            return img, cap

        ids, mask = self.tokenizer.encode(cap)
        return img, ids, mask


def collate_with_tokenizer(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    ids = torch.stack([b[1] for b in batch], dim=0)
    mask = torch.stack([b[2] for b in batch], dim=0)
    return imgs, ids, mask
