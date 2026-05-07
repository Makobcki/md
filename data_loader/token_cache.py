from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TokenCacheMetadata:
    kind: str
    codebook_size: int
    codebook_dim: int
    scale_schedule: tuple[int, ...]
    max_token_length: int
    format_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TokenCacheMetadata:
        return cls(
            kind=str(data.get("kind", "vq")),
            codebook_size=int(data["codebook_size"]),
            codebook_dim=int(data.get("codebook_dim", 0)),
            scale_schedule=tuple(int(v) for v in data["scale_schedule"]),  # type: ignore[index]
            max_token_length=int(data["max_token_length"]),
            format_version=int(data.get("format_version", 1)),
        )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["scale_schedule"] = list(self.scale_schedule)
        return data


def save_token_cache_metadata(path: str | Path, metadata: TokenCacheMetadata) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metadata.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_token_cache_metadata(path: str | Path) -> TokenCacheMetadata:
    return TokenCacheMetadata.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_tokenizer_metadata(actual: TokenCacheMetadata, expected: TokenCacheMetadata) -> None:
    for field_name in ("kind", "codebook_size", "codebook_dim", "scale_schedule", "max_token_length"):
        if getattr(actual, field_name) != getattr(expected, field_name):
            raise ValueError(f"Tokenizer metadata mismatch for {field_name}: {getattr(actual, field_name)!r} != {getattr(expected, field_name)!r}.")
    if sum(scale * scale for scale in actual.scale_schedule) > int(actual.max_token_length):
        raise ValueError("scale_schedule exceeds max_token_length.")


def build_synthetic_token_entries(
    metadata: TokenCacheMetadata, *, count: int, seed: int = 0
) -> list[dict[str, object]]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    entries: list[dict[str, object]] = []
    for idx in range(int(count)):
        entries.append(
            {
                "id": f"synthetic-{idx}",
                "tokens": [
                    torch.randint(
                        0,
                        int(metadata.codebook_size),
                        (int(scale) * int(scale),),
                        generator=generator,
                        dtype=torch.long,
                    )
                    for scale in metadata.scale_schedule
                ],
            }
        )
    return entries


class MultiscaleTokenDataset(Dataset):
    def __init__(self, entries: list[dict[str, object]], metadata: TokenCacheMetadata) -> None:
        self.entries = list(entries)
        self.metadata = metadata
        if sum(scale * scale for scale in metadata.scale_schedule) > int(metadata.max_token_length):
            raise ValueError("scale_schedule exceeds max_token_length.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[int(index)]
        tokens = entry.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != len(self.metadata.scale_schedule):
            raise ValueError("Token cache entry has invalid scale count.")
        out = [torch.as_tensor(item, dtype=torch.long) for item in tokens]
        for scale, item in zip(self.metadata.scale_schedule, out, strict=True):
            expected = int(scale) * int(scale)
            if item.numel() != expected:
                raise ValueError(f"Token scale {scale} expected {expected} tokens, got {item.numel()}.")
            if int(item.max().item()) >= int(self.metadata.codebook_size) or int(item.min().item()) < 0:
                raise ValueError("Token value outside tokenizer codebook.")
        return {"id": entry.get("id", str(index)), "tokens": out, "metadata": self.metadata}
