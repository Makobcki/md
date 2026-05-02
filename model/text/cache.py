from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from .conditioning import TextConditioning


@dataclass(frozen=True)
class TextCacheEntry:
    key: str
    shard: str
    idx: int


class TextCache:
    def __init__(self, root: str | Path, *, shard_cache_size: int = 2) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.jsonl"
        self.metadata_path = self.root / "metadata.json"
        self.shard_cache_size = int(shard_cache_size)
        if self.shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive.")
        self._shard_cache: "OrderedDict[Path, dict[str, torch.Tensor]]" = OrderedDict()
        self.entries: dict[str, TextCacheEntry] = {}
        self.metadata: dict = {}
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = str(obj["key"])
                self.entries[key] = TextCacheEntry(key=key, shard=str(obj["shard"]), idx=int(obj["idx"]))

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def _load_safetensors(self, path: Path) -> dict[str, torch.Tensor]:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("TextCache requires safetensors to be installed.") from exc
        return load_file(str(path), device="cpu")

    def _load_shard(self, shard: str) -> dict[str, torch.Tensor]:
        path = self.root / "shards" / shard
        cached = self._shard_cache.get(path)
        if cached is not None:
            self._shard_cache.move_to_end(path)
            return cached
        payload = self._load_safetensors(path)
        if len(self._shard_cache) >= self.shard_cache_size:
            self._shard_cache.popitem(last=False)
        self._shard_cache[path] = payload
        return payload

    def load(self, key: str) -> TextConditioning:
        entry = self.entries[key]
        payload = self._load_shard(entry.shard)
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
        payload = self._load_safetensors(self.root / "empty_prompt.safetensors")
        return TextConditioning(
            tokens=payload["tokens"],
            mask=payload["mask"].to(torch.bool),
            pooled=payload["pooled"],
            is_uncond=torch.ones(payload["tokens"].shape[0], dtype=torch.bool),
        )
