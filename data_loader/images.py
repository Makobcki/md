from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def load_image_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (512, 512):
            raise RuntimeError(f"Unexpected image size: {im.size}")
        # PIL -> torch float32 in [0,1], CHW
        arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        # [0,1] -> [-1,1]
        return x * 2.0 - 1.0
