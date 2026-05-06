from __future__ import annotations

import argparse
import json
from dataclasses import replace

from config.loader import load_sample_config, parse_cli_overrides

from .api import (
    SampleOptions,
    _metadata_sidecar_path,
    _sample_metadata,
    _write_sample_metadata,
    run_sample,
)

__all__ = [
    "SampleOptions",
    "_metadata_sidecar_path",
    "_sample_metadata",
    "_write_sample_metadata",
    "main",
    "run_sample",
]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _bounded_strength(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _apply_cli_args(options: SampleOptions, args: argparse.Namespace) -> SampleOptions:
    updates: dict[str, object] = {}
    for name in (
        "ckpt",
        "out",
        "n",
        "steps",
        "prompt",
        "neg_prompt",
        "cfg",
        "shift",
        "sampler",
        "seed",
        "device",
        "width",
        "height",
        "init_image",
        "strength",
        "mask",
        "control_image",
        "control_strength",
        "control_type",
        "task",
    ):
        value = getattr(args, name)
        if value is not None:
            updates[name] = value
    if bool(args.latent_only):
        updates["latent_only"] = True
    if bool(args.fake_vae):
        updates["fake_vae"] = True
    if args.use_ema is not None:
        updates["use_ema"] = bool(args.use_ema)
    if updates:
        options = replace(options, **updates)
        options.validate()
    return options


def _main_impl() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="Config path. Defaults to configs/sample.kdl.")
    ap.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value, e.g. --set prompt.text='a landscape'.",
    )
    ap.add_argument(
        "--print-config", action="store_true", help="Print resolved sample options and exit."
    )
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=_positive_int, default=None)
    ap.add_argument("--steps", type=_positive_int, default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--neg_prompt", "--negative-prompt", dest="neg_prompt", default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument(
        "--shift",
        type=_positive_float,
        default=None,
        help="Positive inference timestep shift override. Defaults to checkpoint/config sampling shift.",
    )
    ap.add_argument("--sampler", default=None, choices=("flow_euler", "flow_heun"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=_positive_int, default=None)
    ap.add_argument("--height", type=_positive_int, default=None)
    ap.add_argument("--init-image", dest="init_image", default=None)
    ap.add_argument("--strength", type=_bounded_strength, default=None)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--control-image", dest="control_image", default=None)
    ap.add_argument(
        "--control-strength", dest="control_strength", type=_nonnegative_float, default=None
    )
    ap.add_argument(
        "--control-type",
        dest="control_type",
        default=None,
        choices=("none", "latent_identity", "image", "canny", "depth", "pose", "lineart", "normal"),
    )
    ap.add_argument("--task", default=None, choices=("txt2img", "img2img", "inpaint", "control"))
    ap.add_argument(
        "--latent-only",
        dest="latent_only",
        action="store_true",
        help="Write final latent tensor instead of decoding through VAE.",
    )
    ap.add_argument(
        "--fake-vae",
        dest="fake_vae",
        action="store_true",
        help="Use deterministic fake VAE decoder for smoke tests.",
    )
    ap.add_argument("--use-ema", dest="use_ema", action="store_true", default=None)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    args = ap.parse_args()

    config = load_sample_config(
        args.config or None,
        overrides=parse_cli_overrides(args.set_values),
    )
    options = _apply_cli_args(config.options, args)
    if args.shift is None and not args.config and not args.set_values:
        options = replace(options, shift=None)
    if args.print_config:
        print(json.dumps(options.__dict__, indent=2, ensure_ascii=False), flush=True)
        return
    run_sample(options)


def main() -> None:
    try:
        _main_impl()
    except Exception as exc:
        from diffusion.utils.oom import is_torch_oom_error, print_torch_oom

        if is_torch_oom_error(exc):
            print_torch_oom(exc, context="sampling")
            raise SystemExit(2) from None
        raise


if __name__ == "__main__":
    main()
