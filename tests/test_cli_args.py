from __future__ import annotations

from pathlib import Path

from config.loader import load_train_config
from webui.backend.argparse_reader import parse_argparse_args


def test_sample_n_uses_positive_int_validator() -> None:
    args = parse_argparse_args(Path(__file__).resolve().parents[1] / "sample" / "cli.py")
    n_arg = next(item for item in args if item["name"] == "n")

    assert n_arg["type"] == "_positive_int"


def test_train_kdl_defaults_disable_compile_and_nonfinite_grad_fail() -> None:
    cfg = load_train_config()

    assert cfg.compile is False
    assert cfg.fail_on_nonfinite_grad is False


def test_train_config_accepts_default_kdl() -> None:
    cfg = load_train_config()

    assert cfg.architecture == "mmdit_rf"
    assert cfg.objective == "rectified_flow"
    assert cfg.prediction_type == "flow_velocity"
    assert cfg.hidden_dim == 1152
    assert cfg.pos_embed == "rope_2d"
    assert cfg.eval_sampler == "flow_heun"


def test_train_config_accepts_kdl_preset_overrides() -> None:
    cfg = load_train_config(
        overrides={
            "model": {"preset": "mmdit_1024"},
            "training": {"preset": "single_gpu_debug", "batch_size": 1},
        }
    )

    assert cfg.architecture == "mmdit_rf"
    assert cfg.objective == "rectified_flow"
    assert cfg.hidden_dim == 1536
    assert cfg.depth == 28
    assert cfg.batch_size == 1
    assert cfg.dataset_limit == 8


def test_prepare_text_cache_cpu_defaults_to_fp32() -> None:
    import torch

    from scripts.prepare_text_cache import _resolve_prepare_dtype

    assert _resolve_prepare_dtype(None, "bf16", torch.device("cpu")) is torch.float32
    assert _resolve_prepare_dtype("bf16", "fp32", torch.device("cpu")) is torch.bfloat16
    assert _resolve_prepare_dtype(None, "bf16", torch.device("cuda")) is torch.bfloat16
