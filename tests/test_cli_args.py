from __future__ import annotations

from pathlib import Path

from config.train import TrainConfig
from webui.backend.argparse_reader import parse_argparse_args


def test_sample_n_uses_positive_int_validator() -> None:
    args = parse_argparse_args(Path(__file__).resolve().parents[1] / "sample" / "cli.py")
    n_arg = next(item for item in args if item["name"] == "n")

    assert n_arg["type"] == "_positive_int"


def test_train_yaml_defaults_disable_compile_and_nonfinite_grad_fail() -> None:
    cfg = TrainConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "config" / "train.yaml"))

    assert cfg.compile is False
    assert cfg.fail_on_nonfinite_grad is False
