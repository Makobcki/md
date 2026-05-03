from __future__ import annotations

import torch

from diffusion.objectives.flow_matching import rectified_flow_loss
from model.mmdit.norms import FP32LayerNorm, RMSNorm
from model.mmdit.timestep import TimestepEmbedder
from train.loop_mmdit_full import _per_sample_flow_mse


def test_flow_losses_accumulate_in_fp32_for_bf16_predictions() -> None:
    pred = torch.randn(2, 4, 4, 4, dtype=torch.bfloat16)
    target = torch.randn(2, 4, 4, 4, dtype=torch.bfloat16)
    mask = torch.ones(2, 1, 4, 4, dtype=torch.bfloat16)
    per = _per_sample_flow_mse(pred, target, mask, mask_weight=1.0, unmask_weight=0.1)
    loss = rectified_flow_loss(pred, target, torch.ones(2), mask=mask)
    assert per.dtype == torch.float32
    assert loss.dtype == torch.float32
    assert torch.isfinite(per).all()
    assert torch.isfinite(loss)


def test_norms_and_timestep_embedding_keep_output_dtype_but_compute_safely() -> None:
    x = (torch.randn(2, 3, 8) * 1000).to(torch.bfloat16)
    for norm in (RMSNorm(8), FP32LayerNorm(8)):
        y = norm.to(torch.bfloat16)(x)
        assert y.dtype == torch.bfloat16
        assert torch.isfinite(y.float()).all()

    embedder = TimestepEmbedder(16).to(torch.bfloat16)
    out = embedder(torch.tensor([0.0, 1.0], dtype=torch.float32))
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
