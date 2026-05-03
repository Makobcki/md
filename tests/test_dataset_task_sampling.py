from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config.train import TrainConfig
from model.text.conditioning import TextConditioning
from train.inpaint_masks import InpaintMaskConfig
from train.runner import _MMDiTCachedDataset


class _FakeLatentDataset:
    entries = [{"md5": f"sample{i}"} for i in range(4)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return (torch.full((4, 8, 8), float(idx)),)


class _FakeTextCache:
    def load(self, key):
        return TextConditioning(torch.zeros(2, 16), torch.ones(2, dtype=torch.bool), torch.zeros(16))


def test_txt2img_weight_one_yields_only_txt2img_batches() -> None:
    ds = _MMDiTCachedDataset(
        _FakeLatentDataset(),
        _FakeTextCache(),
        dataset_tasks={"txt2img": 1.0, "img2img": 0.0, "inpaint": 0.0, "control": 0.0},
    )
    assert {ds[i].task for i in range(len(ds))} == {"txt2img"}


def test_img2img_batch_uses_x0_as_source_latent() -> None:
    ds = _MMDiTCachedDataset(
        _FakeLatentDataset(),
        _FakeTextCache(),
        dataset_tasks={"txt2img": 0.0, "img2img": 1.0, "inpaint": 0.0},
    )
    item = ds[2]
    assert item.task == "img2img"
    assert item.source_latent is not None
    assert torch.equal(item.source_latent, item.x0)


def test_inpaint_batch_uses_reproducible_mask_and_source_latent() -> None:
    mask_cfg = InpaintMaskConfig(mask_min_area=0.25, mask_max_area=0.25, mask_modes={"rectangle": 1.0})
    ds1 = _MMDiTCachedDataset(
        _FakeLatentDataset(),
        _FakeTextCache(),
        dataset_tasks={"inpaint": 1.0},
        seed=99,
        inpaint_config=mask_cfg,
    )
    ds2 = _MMDiTCachedDataset(
        _FakeLatentDataset(),
        _FakeTextCache(),
        dataset_tasks={"inpaint": 1.0},
        seed=99,
        inpaint_config=mask_cfg,
    )
    a = ds1[1]
    b = ds2[1]
    assert a.task == "inpaint"
    assert a.source_latent is not None
    assert torch.equal(a.source_latent, a.x0)
    assert a.mask is not None and b.mask is not None
    assert torch.equal(a.mask, b.mask)
    assert a.mask.shape == (1, 8, 8)


@pytest.mark.parametrize(
    "data,match",
    [
        ({"dataset_tasks": {"txt2img": 1.0, "control": 0.1}}, "reserved"),
        ({"dataset_tasks": {"txt2img": 1.0, "bad": 0.1}}, "unsupported"),
        ({"dataset_tasks": {"txt2img": 0.0}}, "positive"),
        ({"inpaint": {"mask_modes": {"bad": 1.0}}}, "unsupported"),
        ({"inpaint": {"mask_min_area": 0.7, "mask_max_area": 0.2}}, "area"),
    ],
)
def test_train_config_rejects_invalid_task_and_inpaint_config(data: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TrainConfig.from_dict(data)
