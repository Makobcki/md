from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.conditioning import TextConditioning
from train.runner import _MMDiTCachedDataset, _collate_mmdit


def test_mmdit_accepts_inpaint_mask() -> None:
    cfg = MMDiTConfig(hidden_dim=32, depth=1, num_heads=4, double_stream_blocks=1, single_stream_blocks=0, text_dim=16, pooled_dim=16, gradient_checkpointing=False)
    model = MMDiTFlowModel(cfg)
    x = torch.randn(1, 4, 8, 8)
    text = TextConditioning(torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 16))
    out = model(x, torch.rand(1), text, source_latent=torch.randn_like(x), mask=torch.ones(1, 1, 8, 8), task="inpaint")
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


def test_mmdit_dataset_builds_inpaint_train_batch() -> None:
    ds = _MMDiTCachedDataset(
        _FakeLatentDataset(),
        _FakeTextCache(),
        dataset_tasks={"txt2img": 0.0, "img2img": 0.0, "inpaint": 1.0},
    )
    item = ds[0]
    assert item.task == "inpaint"
    assert item.source_latent is not None
    assert item.mask is not None
    assert item.mask.shape == (1, 8, 8)
    assert float(item.mask.max()) == 1.0

    batch = _collate_mmdit([item])
    assert batch.source_latent is not None
    assert batch.mask is not None
    assert batch.source_latent.shape == (1, 4, 8, 8)
    assert batch.mask.shape == (1, 1, 8, 8)


def test_flow_sampler_preserves_unmasked_inpaint_region() -> None:
    from samplers.flow_euler import sample_flow_euler

    class _ConstantFlow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))

        def forward(self, x, t, text, **kwargs):
            del t, text, kwargs
            return torch.ones_like(x)

    model = _ConstantFlow()
    text = TextConditioning(torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 4))
    source = torch.full((1, 4, 4, 4), 7.0)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, 1:3, 1:3] = 1.0
    out = sample_flow_euler(
        model,
        (1, 4, 4, 4),
        text,
        steps=2,
        noise=torch.zeros(1, 4, 4, 4),
        source_latent=source,
        mask=mask,
        task="inpaint",
    )
    assert torch.equal(out * (1.0 - mask), source * (1.0 - mask))


def test_preserve_inpaint_region_uses_noised_source_at_intermediate_timestep() -> None:
    from samplers.cfg import preserve_inpaint_region

    x = torch.full((1, 1, 2, 2), 9.0)
    source = torch.full_like(x, 2.0)
    eps = torch.full_like(x, -2.0)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    out = preserve_inpaint_region(
        x,
        source_latent=source,
        mask=mask,
        task="inpaint",
        reference_noise=eps,
        t=torch.tensor(0.25),
    )

    expected_known = (1.0 - 0.25) * source + 0.25 * eps
    assert torch.equal(out * (1.0 - mask), expected_known * (1.0 - mask))
    assert torch.equal(out * mask, x * mask)
