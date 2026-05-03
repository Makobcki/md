from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class AspectRatioBucket:
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return float(self.width) / float(self.height)

    @property
    def latent_size(self) -> tuple[int, int]:
        return self.height, self.width


def validate_buckets(
    buckets: Sequence[AspectRatioBucket],
    *,
    latent_downsample_factor: int = 8,
    latent_patch_size: int = 2,
) -> list[AspectRatioBucket]:
    if not buckets:
        raise ValueError("at least one aspect ratio bucket is required")
    out: list[AspectRatioBucket] = []
    seen: set[tuple[int, int]] = set()
    for bucket in buckets:
        if bucket.width <= 0 or bucket.height <= 0:
            raise ValueError("bucket width/height must be positive")
        if bucket.width % latent_downsample_factor != 0 or bucket.height % latent_downsample_factor != 0:
            raise ValueError("bucket dimensions must be divisible by latent_downsample_factor")
        latent_w = bucket.width // latent_downsample_factor
        latent_h = bucket.height // latent_downsample_factor
        if latent_w % latent_patch_size != 0 or latent_h % latent_patch_size != 0:
            raise ValueError("bucket latent grid must be divisible by latent_patch_size")
        key = (bucket.width, bucket.height)
        if key in seen:
            continue
        seen.add(key)
        out.append(bucket)
    return out


def parse_buckets(items: Iterable[Sequence[int] | dict | str]) -> list[AspectRatioBucket]:
    buckets: list[AspectRatioBucket] = []
    for item in items:
        if isinstance(item, str):
            sep = "x" if "x" in item else "×"
            left, right = item.lower().replace("×", "x").split("x", 1)
            buckets.append(AspectRatioBucket(width=int(left), height=int(right)))
        elif isinstance(item, dict):
            buckets.append(AspectRatioBucket(width=int(item["width"]), height=int(item["height"])))
        else:
            width, height = item
            buckets.append(AspectRatioBucket(width=int(width), height=int(height)))
    return buckets


def assign_bucket(width: int, height: int, buckets: Sequence[AspectRatioBucket]) -> AspectRatioBucket:
    if width <= 0 or height <= 0:
        raise ValueError("sample width/height must be positive")
    aspect = float(width) / float(height)
    return min(buckets, key=lambda b: (abs(b.aspect - aspect), abs(b.width * b.height - width * height)))


def group_entries_by_bucket(entries: Sequence[dict], buckets: Sequence[AspectRatioBucket]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {(b.width, b.height): [] for b in buckets}
    for idx, entry in enumerate(entries):
        width = int(entry.get("width", entry.get("image_width", 0)) or 0)
        height = int(entry.get("height", entry.get("image_height", 0)) or 0)
        if width <= 0 or height <= 0:
            path_size = entry.get("size")
            if isinstance(path_size, (list, tuple)) and len(path_size) == 2:
                width, height = int(path_size[0]), int(path_size[1])
        bucket = assign_bucket(width, height, buckets)
        groups[(bucket.width, bucket.height)].append(idx)
    return groups


class AspectBucketBatchSampler:
    """Yield batches whose samples all belong to the same aspect-ratio bucket."""

    def __init__(
        self,
        entries: Sequence[dict],
        buckets: Sequence[AspectRatioBucket],
        *,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.entries = entries
        self.buckets = list(buckets)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.groups = group_entries_by_bucket(entries, self.buckets)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        bucket_items = [list(indices) for indices in self.groups.values() if indices]
        if self.shuffle:
            for indices in bucket_items:
                rng.shuffle(indices)
            rng.shuffle(bucket_items)
        for indices in bucket_items:
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.groups.values():
            full, rem = divmod(len(indices), self.batch_size)
            total += full
            if rem and not self.drop_last:
                total += 1
        return total
