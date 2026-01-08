from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import random
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class DataConfig:
    data_root: str
    split: str  # "train" / "val" / "test"
    image_size: int


def _list_images(data_root: Path, split: str) -> List[Path]:
    """
    Ожидаем структуру:
    ./data/raw/<dataset_name>/<split>/{1,2}/**/*.(jpg|png|webp|...)
    """
    paths: List[Path] = []
    for dataset_dir in sorted(data_root.glob("*")):
        if not dataset_dir.is_dir():
            continue

        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue

        for cls in ("1", "2"):
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue

            for p in cls_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    paths.append(p)

    return paths


class ResizePadCrop:
    """
    Сохраняет aspect ratio, затем делает pad до квадрата, затем crop в target size.

    - mode="random": RandomCrop (для train)
    - mode="center": CenterCrop (для val/test)
    """

    def __init__(self, size: int, mode: str = "random", pad_fill: int = 0) -> None:
        if mode not in {"random", "center"}:
            raise ValueError("mode must be 'random' or 'center'")
        self.size = int(size)
        self.mode = mode
        self.pad_fill = int(pad_fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        s = self.size
        w, h = img.size

        # 1) Scale so that the MIN side >= s
        scale = s / min(w, h)
        nw = max(s, int(round(w * scale)))
        nh = max(s, int(round(h * scale)))
        img = img.resize((nw, nh), resample=Image.BICUBIC)

        # 2) Pad to square >= s (letterbox)
        w2, h2 = img.size
        side = max(w2, h2, s)
        pad_w = side - w2
        pad_h = side - h2

        if pad_w > 0 or pad_h > 0:
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            img = ImageOps.expand(img, border=(left, top, right, bottom), fill=self.pad_fill)

        # 3) Crop to s x s
        w3, h3 = img.size
        if w3 == s and h3 == s:
            return img

        if self.mode == "random":
            w2, h2 = img.size
            x0 = random.randint(0, w2 - s)
            y0 = random.randint(0, h2 - s)
            return img.crop((x0, y0, x0 + s, y0 + s))


class ImageFolderRecursive(Dataset):
    """
    Unconditional dataset: возвращает только изображение (тензор [-1, 1]).
    """

    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.data_root)
        self.paths = _list_images(self.root, cfg.split)

        if not self.paths:
            raise RuntimeError(
                f"No images found in {self.root} with split='{cfg.split}'. "
                f"Expected ./data/raw/*/{cfg.split}/{{1,2}}/**"
            )

        # Train: random crop, Val/Test: center crop
        crop_mode = "random" if cfg.split == "train" else "center"

        self.tf = transforms.Compose([
            ResizePadCrop(cfg.image_size, mode=crop_mode, pad_fill=0),
            transforms.RandomHorizontalFlip(p=0.5 if cfg.split == "train" else 0.0),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.tf(img)
        return x
