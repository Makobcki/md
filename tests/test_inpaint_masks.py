from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.inpaint_masks import InpaintMaskConfig, generate_inpaint_mask


def _area(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item())


@pytest.mark.parametrize("mode", ["rectangle", "brush", "center_rectangle", "small", "large", "random_blocks"])
def test_inpaint_mask_modes_shape_area_and_binary(mode: str) -> None:
    cfg = InpaintMaskConfig(mask_min_area=0.10, mask_max_area=0.40, mask_modes={mode: 1.0})
    mask = generate_inpaint_mask((4, 32, 32), config=cfg, seed=123, dtype=torch.float32)

    assert mask.shape == (1, 32, 32)
    assert mask.dtype == torch.float32
    assert torch.isfinite(mask).all()
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    assert 0.10 <= _area(mask) <= 0.40


def test_full_inpaint_mask_requires_and_respects_full_area() -> None:
    cfg = InpaintMaskConfig(mask_min_area=1.0, mask_max_area=1.0, mask_modes={"full": 1.0})
    mask = generate_inpaint_mask((16, 16), config=cfg, seed=123)
    assert mask.shape == (1, 16, 16)
    assert torch.equal(mask, torch.ones_like(mask))


def test_inpaint_mask_seed_is_deterministic() -> None:
    cfg = InpaintMaskConfig(mask_min_area=0.05, mask_max_area=0.25, mask_modes={"random_blocks": 1.0})
    a = generate_inpaint_mask((4, 24, 24), config=cfg, seed=42)
    b = generate_inpaint_mask((4, 24, 24), config=cfg, seed=42)
    c = generate_inpaint_mask((4, 24, 24), config=cfg, seed=43)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mask_min_area": 0.6, "mask_max_area": 0.1}, "area"),
        ({"mask_modes": {"unknown": 1.0}}, "unsupported"),
        ({"mask_modes": {"rectangle": -1.0}}, "non-negative"),
        ({"mask_modes": {"rectangle": 0.0}}, "positive"),
        ({"mask_max_area": 0.6, "mask_modes": {"full": 1.0}}, "requires"),
    ],
)
def test_inpaint_mask_config_rejects_invalid_settings(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        InpaintMaskConfig(**kwargs)
