from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    image_dir: str
    meta_dir: str
    tags_dir: str
    caption_field: str
    images_only: bool
    min_tag_count: int
    require_512: bool
    val_ratio: float
    cache_dir: str
    failed_list: str
    seed: int
