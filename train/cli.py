from __future__ import annotations

import argparse
from dataclasses import replace

from config.loader import load_train_config
from config.train import TrainConfig


def _linear_params(in_dim: int, out_dim: int, bias: bool = True) -> int:
    return int(in_dim) * int(out_dim) + (int(out_dim) if bias else 0)


def _norm_params(hidden_dim: int, *, rms_norm: bool) -> int:
    return int(hidden_dim) if rms_norm else int(hidden_dim) * 2


def _ff_params(hidden_dim: int, mlp_ratio: float, *, swiglu: bool) -> int:
    d = int(hidden_dim)
    inner = int(d * float(mlp_ratio))
    if swiglu:
        return _linear_params(d, inner * 2) + _linear_params(inner, d)
    return _linear_params(d, inner) + _linear_params(inner, d)


def _estimate_mmdit_params_from_config(cfg: TrainConfig) -> int:
    d = int(cfg.hidden_dim)
    c = int(cfg.latent_channels)
    p = int(cfg.latent_patch_size)
    out_patch = c * p * p
    head_dim = d // int(cfg.num_heads)
    total = 0
    total += 3 * (c * p * p * d + d)
    total += 1 * (1 * p * p * d + d)
    total += 3 * _linear_params(int(cfg.text_dim), d)
    total += _linear_params(int(cfg.pooled_dim), d)
    total += _linear_params(d, d * 4) + _linear_params(d * 4, d)
    total += 8 * d
    total += 5 * d
    total += _norm_params(d, rms_norm=bool(cfg.rms_norm))

    qk_norm_params = 2 * head_dim if bool(cfg.qk_norm) else 0
    norm = _norm_params(d, rms_norm=bool(cfg.rms_norm))
    double = 0
    double += 4 * norm
    double += 2 * _linear_params(d, 6 * d)
    double += 2 * _linear_params(d, 3 * d)
    double += qk_norm_params
    double += 2 * _linear_params(d, d)
    double += 2 * _ff_params(d, float(cfg.mlp_ratio), swiglu=bool(cfg.swiglu))
    total += int(cfg.double_stream_blocks) * double

    single = 0
    single += 2 * norm
    single += _linear_params(d, 6 * d)
    single += _linear_params(d, 3 * d)
    single += qk_norm_params
    single += _linear_params(d, d)
    single += _ff_params(d, float(cfg.mlp_ratio), swiglu=bool(cfg.swiglu))
    total += int(cfg.single_stream_blocks) * single

    total += 2 * d
    total += _linear_params(d, 2 * d)
    total += _linear_params(d, out_patch)
    return int(total)


def _dry_run_light(cfg: TrainConfig) -> None:
    if cfg.architecture != "mmdit_rf":
        raise RuntimeError("Only architecture=mmdit_rf is supported.")
    if str(cfg.mode) != "latent":
        raise RuntimeError("architecture=mmdit_rf requires mode=latent.")
    latent_side = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    print(
        "[DRY-RUN] "
        f"architecture={cfg.architecture} objective={cfg.objective} mode={cfg.mode} "
        f"image_size={cfg.image_size} latent_shape=({cfg.latent_channels}, {latent_side}, {latent_side}) "
        f"hidden_dim={cfg.hidden_dim} depth={cfg.depth} num_heads={cfg.num_heads} "
        f"params={_estimate_mmdit_params_from_config(cfg)}",
        flush=True,
    )


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if args.resume:
        cfg = replace(cfg, resume_ckpt=str(args.resume))
    if args.ckpt_keep_last is not None:
        cfg = replace(cfg, ckpt_keep_last=int(args.ckpt_keep_last))
    return cfg


def _main_impl() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument(
        "--profile",
        default="",
        choices=("", "smoke", "overfit", "dev", "base", "full", "milestone_a", "milestone_b", "milestone_c", "distributed_smoke", "fsdp_template"),
        help="Convenience profile; ignored when --config is set.",
    )
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt-keep-last", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile_to_config = {
        "smoke": "config/train_smoke.yaml",
        "overfit": "config/train_overfit.yaml",
        "dev": "config/train_dev.yaml",
        "base": "config/train_base.yaml",
        "full": "config/train.yaml",
        "milestone_a": "config/train_milestone_a.yaml",
        "milestone_b": "config/train_milestone_b.yaml",
        "milestone_c": "config/train_milestone_c.yaml",
        "distributed_smoke": "config/train_distributed_smoke.yaml",
        "fsdp_template": "config/train_fsdp_template.yaml",
    }
    config_path = args.config or profile_to_config.get(args.profile, "config/train.yaml")
    cfg = load_train_config(config_path)
    cfg = _apply_overrides(cfg, args)
    if args.dry_run:
        _dry_run_light(cfg)
    else:
        from train.runner import run

        run(cfg)


def main() -> None:
    try:
        _main_impl()
    except Exception as exc:
        from diffusion.utils.oom import is_torch_oom_error, print_torch_oom

        if is_torch_oom_error(exc):
            print_torch_oom(exc, context="training")
            raise SystemExit(2) from None
        raise


if __name__ == "__main__":
    main()
