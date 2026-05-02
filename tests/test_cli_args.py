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


def test_train_config_accepts_legacy_model_fields() -> None:
    cfg = TrainConfig.from_dict({
        "base_channels": 16,
        "channel_mults": [1],
        "attn_resolutions": [8],
    })

    assert cfg.self_attn_type == "global"
    assert cfg.attention_placement == "all"
    assert cfg.cross_attn_resolutions == ()
    assert cfg.mid_blocks == 1
    assert cfg.checkpoint_attention is False
    assert cfg.checkpoint_downsample is False
    assert cfg.optimizer == "adamw"
    assert cfg.self_cond_interval == 1


def test_train_config_accepts_image_only_alias() -> None:
    cfg = TrainConfig.from_dict({"image_only": True})

    assert cfg.images_only is True


def test_train_config_accepts_mmdit_rf_yaml() -> None:
    cfg = TrainConfig.from_yaml(str(Path(__file__).resolve().parents[1] / "config" / "train_mmdit_rf.yaml"))

    assert cfg.architecture == "mmdit_rf"
    assert cfg.objective == "rectified_flow"
    assert cfg.prediction_type == "flow_velocity"
    assert cfg.hidden_dim == 1024
    assert cfg.eval_sampler == "flow_heun"
