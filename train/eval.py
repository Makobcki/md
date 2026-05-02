from __future__ import annotations

from pathlib import Path

import torch


def save_image_grid(x: torch.Tensor, path: str | Path, nrow: int) -> None:
    try:
        from torchvision.utils import save_image
    except Exception as exc:
        raise RuntimeError("Saving eval images requires a working torchvision install.") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(x, path, nrow=nrow)


def _resolve_eval_prompts(path: str, *, count: int, use_text_conditioning: bool) -> list[str]:
    if not use_text_conditioning:
        return [""] * int(count)
    p = Path(path)
    if not str(path).strip() or not p.exists():
        raise RuntimeError(f"Eval prompts file not found: {path}")
    prompts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return prompts[:count] if prompts else ["1girl"]
