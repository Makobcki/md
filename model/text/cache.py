from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .conditioning import TextConditioning


@dataclass(frozen=True)
class TextCacheEntry:
    key: str
    shard: str
    idx: int


class TextCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.jsonl"
        self.entries: dict[str, TextCacheEntry] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = str(obj["key"])
                self.entries[key] = TextCacheEntry(key=key, shard=str(obj["shard"]), idx=int(obj["idx"]))

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def load(self, key: str) -> TextConditioning:
        entry = self.entries[key]
        path = self.root / "shards" / entry.shard
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("TextCache requires safetensors to be installed.") from exc
        payload = load_file(str(path), device="cpu")
        i = entry.idx
        tokens = payload["tokens"][i]
        mask = payload["mask"][i].to(torch.bool)
        pooled = payload["pooled"][i]
        is_uncond = payload.get("is_uncond")
        return TextConditioning(
            tokens=tokens,
            mask=mask,
            pooled=pooled,
            is_uncond=is_uncond[i].to(torch.bool).view(()) if is_uncond is not None else None,
        )

    def load_empty(self) -> TextConditioning:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("TextCache requires safetensors to be installed.") from exc
        payload = load_file(str(self.root / "empty_prompt.safetensors"), device="cpu")
        return TextConditioning(
            tokens=payload["tokens"],
            mask=payload["mask"].to(torch.bool),
            pooled=payload["pooled"],
            is_uncond=torch.ones(payload["tokens"].shape[0], dtype=torch.bool),
        )

