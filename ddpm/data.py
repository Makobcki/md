from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

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


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int = 50_000
    max_len: int = 64
    lowercase: bool = True
    strip_punct: bool = True


# ----------------------------
# Tokenizer (simple & offline)
# ----------------------------

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


def _normalize_text(s: str, lowercase: bool, strip_punct: bool) -> str:
    s = s.strip()
    if lowercase:
        s = s.lower()
    if strip_punct:
        s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


class SimpleTokenizer:
    """
    Простой оффлайн-токенизатор: whitespace + vocab по частотам.
    """
    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, vocab: Dict[str, int], text_cfg: TextConfig):
        self.vocab = vocab
        self.text_cfg = text_cfg

        self.pad_id = self.vocab[self.PAD]
        self.unk_id = self.vocab[self.UNK]
        self.bos_id = self.vocab[self.BOS]
        self.eos_id = self.vocab[self.EOS]

    @staticmethod
    def build(captions: List[str], text_cfg: TextConfig) -> "SimpleTokenizer":
        freq: Dict[str, int] = {}
        for cap in captions:
            cap = _normalize_text(cap, text_cfg.lowercase, text_cfg.strip_punct)
            if not cap:
                continue
            for tok in cap.split(" "):
                if not tok:
                    continue
                freq[tok] = freq.get(tok, 0) + 1

        vocab_tokens = [SimpleTokenizer.PAD, SimpleTokenizer.UNK, SimpleTokenizer.BOS, SimpleTokenizer.EOS]
        items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        for tok, _ in items:
            vocab_tokens.append(tok)
            if len(vocab_tokens) >= int(text_cfg.vocab_size):
                break

        vocab = {t: i for i, t in enumerate(vocab_tokens)}
        return SimpleTokenizer(vocab, text_cfg)

    def encode(self, caption: str) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        cap = _normalize_text(caption, self.text_cfg.lowercase, self.text_cfg.strip_punct)
        toks = cap.split(" ") if cap else []
        ids = [self.bos_id]
        for t in toks:
            ids.append(self.vocab.get(t, self.unk_id))
        ids.append(self.eos_id)

        max_len = int(self.text_cfg.max_len)
        ids = ids[:max_len]
        attn_mask = [True] * len(ids)

        while len(ids) < max_len:
            ids.append(self.pad_id)
            attn_mask.append(False)

        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.bool)


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
    if field in meta and isinstance(meta[field], str):
        return meta[field]
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

        if _extract_tag_count(meta) < int(cfg.min_tag_count):
            continue

        cap = _extract_caption(meta, cfg.caption_field).strip()
        if not cap:
            continue

        candidates = list(img_dir.glob(f"{md5}.*"))
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
        text_cfg: TextConfig,
        tokenizer: Optional[SimpleTokenizer],
        cond_drop_prob: float,
        seed: int,
    ):
        self.entries = entries
        self.text_cfg = text_cfg
        self.tokenizer = tokenizer
        self.cond_drop_prob = float(cond_drop_prob)
        self.rng = random.Random(int(seed))

    def __len__(self) -> int:
        return len(self.entries)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != (512, 512):
                im = im.resize((512, 512), resample=Image.BICUBIC)
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
