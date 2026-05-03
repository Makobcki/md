from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.metrics import loss_by_t_bins, required_t_bin_keys


def test_loss_by_t_bins_places_boundaries_and_includes_one() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t = torch.tensor([0.0, 0.0999, 0.1, 1.0])

    bins = loss_by_t_bins(losses, t, bins=10)

    assert bins["loss_t_bin_00_01"] == pytest.approx(1.5)
    assert bins["loss_t_bin_01_02"] == pytest.approx(3.0)
    assert bins["loss_t_bin_09_10"] == pytest.approx(4.0)


def test_loss_by_t_bins_omits_empty_bins_safely() -> None:
    bins = loss_by_t_bins(torch.tensor([5.0]), torch.tensor([0.55]), bins=10)

    assert bins == {"loss_t_bin_05_06": 5.0}
    assert "loss_t_bin_00_01" not in bins


def test_loss_by_t_bins_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same number of elements"):
        loss_by_t_bins(torch.ones(2), torch.ones(3), bins=10)


def test_required_t_bin_keys_uses_stable_names() -> None:
    assert required_t_bin_keys(bins=10) == [
        "loss_t_bin_00_01",
        "loss_t_bin_01_02",
        "loss_t_bin_02_03",
        "loss_t_bin_03_04",
        "loss_t_bin_04_05",
        "loss_t_bin_05_06",
        "loss_t_bin_06_07",
        "loss_t_bin_07_08",
        "loss_t_bin_08_09",
        "loss_t_bin_09_10",
    ]
