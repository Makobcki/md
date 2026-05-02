from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning
from train.runner import _MMDiTCachedDataset, _collate_mmdit


def test_mmdit_accepts_img2img_source_latent() -> None:
    cfg = MMDiTConfig(hidden_dim=32, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0, text_dim=16, pooled_dim=16, gradient_checkpointing=False)
    model = MMDiTFlowModel(cfg)
    x = torch.randn(1, 4, 8, 8)
    text = TextConditioning(torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 16))
    out = model(x, torch.rand(1), text, source_latent=torch.randn_like(x), task="img2img")
    assert out.shape == x.shape


class _FakeLatentDataset:
    entries = [{"md5": "a"}]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (torch.ones(4, 8, 8),)


class _FakeTextCache:
    def load(self, key):
        return TextConditioning(torch.randn(2, 16), torch.ones(2, dtype=torch.bool), torch.randn(16))


def test_mmdit_dataset_builds_img2img_train_batch() -> None:
    ds = _MMDiTCachedDataset(
        _FakeLatentDataset(),
        _FakeTextCache(),
        dataset_tasks={"txt2img": 0.0, "img2img": 1.0, "inpaint": 0.0},
    )
    item = ds[0]
    assert item.task == "img2img"
    assert item.source_latent is not None
    assert torch.equal(item.source_latent, item.x0)

    batch = _collate_mmdit([item])
    assert batch.source_latent is not None
    assert batch.source_latent.shape == (1, 4, 8, 8)
    assert batch.mask is None
