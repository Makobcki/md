from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterator

from torch.utils.data import Sampler


class ShardAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        shard_to_entry_indices: dict[Path, list[int]],
        batch_size: int,
        drop_last: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not shard_to_entry_indices:
            raise ValueError("shard_to_entry_indices must be non-empty.")
        self._shard_to_entry_indices = {
            shard: list(indices) for shard, indices in shard_to_entry_indices.items()
        }
        self._batch_size = int(batch_size)
        self._drop_last = bool(drop_last)
        self._seed = int(seed)
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self._epoch)
        shard_items = list(self._shard_to_entry_indices.items())
        rng.shuffle(shard_items)
        for _, indices in shard_items:
            local_indices = list(indices)
            rng.shuffle(local_indices)
            for i in range(0, len(local_indices), self._batch_size):
                batch = local_indices[i : i + self._batch_size]
                if len(batch) < self._batch_size and self._drop_last:
                    continue
                yield batch
        self._epoch += 1

    def __len__(self) -> int:
        total = 0
        for indices in self._shard_to_entry_indices.values():
            if self._drop_last:
                total += len(indices) // self._batch_size
            else:
                total += int(math.ceil(len(indices) / self._batch_size))
        return total
