from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .conditioning import TextConditioning


@dataclass(frozen=True)
class TextCacheEntry:
    key: str
    shard: str
    idx: int
    text_hash: str = ""


class TextCache:
    def __init__(self, root: str | Path, *, shard_cache_size: int = 2) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.jsonl"
        self.metadata_path = self.root / "metadata.json"
        self.manifest_path = self.root / "manifest.json"
        self.empty_prompt_path = self.root / "empty_prompt.safetensors"
        self.shard_cache_size = int(shard_cache_size)
        if self.shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive.")
        self._shard_cache: "OrderedDict[Path, dict[str, torch.Tensor]]" = OrderedDict()
        self.entries: dict[str, TextCacheEntry] = {}
        self.metadata: dict[str, Any] = {}
        self.manifest: dict[str, Any] = {}
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.index_path.exists():
            for line_no, line in enumerate(self.index_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = str(obj["key"])
                if key in self.entries:
                    raise ValueError(f"Duplicate text cache key at line {line_no}: {key}")
                idx = int(obj["idx"])
                if idx < 0:
                    raise ValueError(f"Negative text cache shard index at line {line_no}: {line}")
                self.entries[key] = TextCacheEntry(
                    key=key,
                    shard=str(obj["shard"]),
                    idx=idx,
                    text_hash=str(obj.get("text_hash", "") or ""),
                )

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def _load_safetensors(self, path: Path) -> dict[str, torch.Tensor]:
        if not path.exists():
            raise FileNotFoundError(f"Missing text cache tensor file: {path}")

        safetensors_error: Exception | None = None
        try:
            from safetensors.torch import load_file

            return load_file(str(path), device="cpu")
        except ImportError:
            pass
        except Exception as exc:
            # Some tests and older local caches used torch.save(...) with a
            # .safetensors extension.  Keep the reader backward-compatible while
            # still preferring real safetensors files whenever possible.
            safetensors_error = exc

        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            if safetensors_error is not None:
                raise RuntimeError(
                    f"Failed to load text cache tensor file as safetensors or torch payload: {path}. "
                    f"safetensors error: {safetensors_error}; torch error: {exc}"
                ) from exc
            raise
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid torch-serialized text cache payload: {path}")
        for name, value in payload.items():
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Invalid text cache tensor {name!r} in {path}: expected torch.Tensor.")
        return payload

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

    @staticmethod
    def _validate_payload_shapes(payload: dict[str, torch.Tensor], *, path: Path | None = None) -> None:
        where = f" in {path}" if path is not None else ""
        for name in ("tokens", "mask", "pooled"):
            if name not in payload:
                raise RuntimeError(f"Text cache shard missing tensor {name!r}{where}.")
        tokens = payload["tokens"]
        mask = payload["mask"]
        pooled = payload["pooled"]
        if tokens.dim() != 3:
            raise RuntimeError(f"Text cache tokens must be [B,N,D]{where}; got {tuple(tokens.shape)}")
        if mask.dim() != 2:
            raise RuntimeError(f"Text cache mask must be [B,N]{where}; got {tuple(mask.shape)}")
        if pooled.dim() != 2:
            raise RuntimeError(f"Text cache pooled must be [B,D]{where}; got {tuple(pooled.shape)}")
        if tokens.shape[:2] != mask.shape:
            raise RuntimeError(f"Text cache token/mask shape mismatch{where}: {tuple(tokens.shape)} vs {tuple(mask.shape)}")
        if tokens.shape[0] != pooled.shape[0]:
            raise RuntimeError(f"Text cache token/pooled batch mismatch{where}: {tuple(tokens.shape)} vs {tuple(pooled.shape)}")
        if "is_uncond" in payload and payload["is_uncond"].shape[0] != tokens.shape[0]:
            raise RuntimeError(f"Text cache is_uncond batch mismatch{where}.")
        if "token_types" in payload:
            token_types = payload["token_types"]
            if token_types.shape != mask.shape:
                raise RuntimeError(f"Text cache token_types shape mismatch{where}: {tuple(token_types.shape)} vs {tuple(mask.shape)}")

    def load(self, key: str) -> TextConditioning:
        if key not in self.entries:
            raise KeyError(f"Text cache missing key: {key}")
        entry = self.entries[key]
        payload = self._load_shard(entry.shard)
        self._validate_payload_shapes(payload, path=self.root / "shards" / entry.shard)
        i = entry.idx
        if i >= payload["tokens"].shape[0]:
            raise RuntimeError(
                f"Text cache index out of range for key={key!r}: idx={i}, shard size={payload['tokens'].shape[0]}"
            )
        tokens = payload["tokens"][i]
        mask = payload["mask"][i].to(torch.bool)
        pooled = payload["pooled"][i]
        is_uncond = payload.get("is_uncond")
        token_types = payload.get("token_types")
        return TextConditioning(
            tokens=tokens,
            mask=mask,
            pooled=pooled,
            is_uncond=is_uncond[i].to(torch.bool).view(()) if is_uncond is not None else None,
            token_types=token_types[i].to(torch.long) if token_types is not None else None,
        )

    def load_empty(self) -> TextConditioning:
        payload = self._load_safetensors(self.empty_prompt_path)
        self._validate_payload_shapes(payload, path=self.empty_prompt_path)
        return TextConditioning(
            tokens=payload["tokens"],
            mask=payload["mask"].to(torch.bool),
            pooled=payload["pooled"],
            is_uncond=payload.get("is_uncond", torch.ones(payload["tokens"].shape[0], dtype=torch.uint8)).to(torch.bool),
            token_types=payload.get("token_types", None).to(torch.long) if payload.get("token_types", None) is not None else None,
        )

    def validate_files_readable(self) -> None:
        if not self.index_path.exists():
            raise RuntimeError(f"Missing text cache index: {self.index_path}")
        if not self.metadata_path.exists():
            raise RuntimeError(f"Missing text cache metadata: {self.metadata_path}")
        if not self.empty_prompt_path.exists():
            raise RuntimeError(f"Missing empty prompt text cache: {self.empty_prompt_path}")
        self._validate_payload_shapes(self._load_safetensors(self.empty_prompt_path), path=self.empty_prompt_path)
        for shard in sorted({entry.shard for entry in self.entries.values()}):
            path = self.root / "shards" / shard
            payload = self._load_safetensors(path)
            self._validate_payload_shapes(payload, path=path)

    def shard_names(self) -> list[str]:
        return sorted({entry.shard for entry in self.entries.values()})
