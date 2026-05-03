from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diffusion.objectives import RectifiedFlowObjective


def test_rectified_flow_training_tuple_shapes() -> None:
    x0 = torch.randn(4, 4, 8, 8)
    tup = RectifiedFlowObjective().sample_training_tuple(x0)
    assert tup.xt.shape == x0.shape
    assert tup.target.shape == x0.shape
    assert tup.t.shape == (4,)
    assert tup.weight.shape == (4,)
    assert float(tup.t.min()) >= 0.0
    assert float(tup.t.max()) <= 1.0

