from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config.train import TrainConfig
from model.text.conditioning import TextConditioning, TrainBatch
from train.runner import _MMDiTCachedDataset, _collate_mmdit


class _LatentDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.entries = [{"md5": "a"}, {"md5": "b"}]
        self.values = [torch.ones(4, 8, 8), torch.full((4, 8, 8), 2.0)]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int):
        return self.values[idx]


class _TextCache:
    def load(self, key: str) -> TextConditioning:
        return TextConditioning(
            tokens=torch.zeros(1, 3, 8),
            mask=torch.ones(1, 3, dtype=torch.bool),
            pooled=torch.zeros(1, 8),
        )


def test_control_task_requires_control_enabled() -> None:
    with pytest.raises(ValueError, match="control.enabled=true"):
        TrainConfig.from_dict({"dataset_tasks": {"txt2img": 0.0, "control": 1.0}})


def test_control_task_builds_control_latents_without_changing_target() -> None:
    ds = _MMDiTCachedDataset(
        _LatentDataset(),
        _TextCache(),
        dataset_tasks={"txt2img": 0.0, "control": 1.0},
        control_enabled=True,
        control_strength=0.5,
        control_num_streams=2,
    )

    item = ds[0]

    assert item.task == "control"
    assert item.source_latent is None
    assert item.mask is None
    assert item.control_latents is not None
    assert item.control_latents.shape == (2, 4, 8, 8)
    assert torch.equal(item.x0, torch.ones(4, 8, 8))
    assert torch.allclose(item.control_latents, torch.full((2, 4, 8, 8), 0.5))
    assert item.metadata["control_preprocessing"] == "latent_identity"


def test_control_collate_keeps_control_streams() -> None:
    text = TextConditioning(
        tokens=torch.zeros(1, 3, 8),
        mask=torch.ones(1, 3, dtype=torch.bool),
        pooled=torch.zeros(1, 8),
    )
    batch = [
        TrainBatch(x0=torch.ones(4, 8, 8), text=text, control_latents=torch.ones(2, 4, 8, 8), task="control"),
        TrainBatch(x0=torch.ones(4, 8, 8), text=text, task="txt2img"),
    ]

    out = _collate_mmdit(batch)

    assert out.control_latents is not None
    assert out.control_latents.shape == (2, 2, 4, 8, 8)
    assert out.task == ["control", "txt2img"]
