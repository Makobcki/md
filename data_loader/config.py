from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    root: str  # ./data_loader/raw/<dataset>
    image_dir: str = "image_512"
    meta_dir: str = "meta"
    tags_dir: str = "tags"
    caption_field: str = "caption_llava_34b_no_tags_short"
    images_only: bool = False
    use_text_conditioning: bool = True
    min_tag_count: int = 8          # post.tag_count >= min_tag_count
    require_512: bool = True        # пропускать всё, что не 512x512
    val_ratio: float = 0.01         # 99/1
    seed: int = 42
    cache_dir: str = ".cache"       # внутри root
    failed_list: str = "failed/md5.txt"
