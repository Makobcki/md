from __future__ import annotations

from pathlib import Path

import torch


DEFAULT_EVAL_PROMPT_SETS: tuple[str, ...] = (
    "core",
    "composition",
    "style",
    "text_rendering",
    "characters",
    "img2img",
    "inpaint",
)


def save_image_grid(x: torch.Tensor, path: str | Path, nrow: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = x.detach().cpu().float().clamp(0.0, 1.0)
    try:
        from torchvision.utils import save_image

        save_image(x, path, nrow=max(int(nrow), 1))
        return
    except Exception:
        pass

    from PIL import Image
    import math
    import numpy as np

    b, c, h, w = x.shape
    if c == 1:
        x = x.repeat(1, 3, 1, 1)
    elif c > 3:
        x = x[:, :3]
    nrow = max(min(int(nrow), b), 1)
    ncol = int(math.ceil(b / nrow))
    grid = torch.zeros(3, ncol * h, nrow * w, dtype=x.dtype)
    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = x[idx]
    arr = (grid.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def load_prompt_file(path: str | Path, *, count: int | None = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"Eval prompts file not found: {p}")
    prompts = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if count is not None:
        prompts = prompts[: int(count)]
    return prompts


def load_eval_prompt_set(
    name: str,
    *,
    root: str | Path = "data/eval_prompts",
    count: int | None = None,
) -> list[str]:
    prompt_set = str(name).strip()
    if not prompt_set or any(part in prompt_set for part in ("/", "\\")) or prompt_set in {".", ".."}:
        raise RuntimeError(f"Invalid eval prompt set name: {name!r}")
    return load_prompt_file(Path(root) / f"{prompt_set}.txt", count=count)


def load_eval_prompt_bank(
    prompt_sets: list[str] | tuple[str, ...] | None = None,
    *,
    root: str | Path = "data/eval_prompts",
    count_per_set: int | None = None,
) -> dict[str, list[str]]:
    names = tuple(prompt_sets or DEFAULT_EVAL_PROMPT_SETS)
    bank: dict[str, list[str]] = {}
    for name in names:
        bank[str(name)] = load_eval_prompt_set(str(name), root=root, count=count_per_set)
    return bank


def _resolve_eval_prompts(path: str, *, count: int, use_text_conditioning: bool) -> list[str]:
    if not use_text_conditioning:
        return [""] * int(count)
    if not str(path).strip():
        raise RuntimeError(f"Eval prompts file not found: {path}")
    prompts = load_prompt_file(path, count=int(count))
    return prompts if prompts else ["1girl"]
