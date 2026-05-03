from __future__ import annotations

import argparse

from .api import (
    SampleOptions,
    _metadata_sidecar_path,
    _sample_metadata,
    _save_image_grid,
    _write_sample_metadata,
    run_sample,
)


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


def _main_impl() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=_positive_int, default=8)
    ap.add_argument("--steps", type=_positive_int, default=30)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--neg_prompt", "--negative-prompt", dest="neg_prompt", default="")
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--shift", type=_nonnegative_float, default=None, help="Positive inference timestep shift override. Defaults to checkpoint/config sampling shift.")
    ap.add_argument("--sampler", default="flow_heun", choices=("flow_euler", "flow_heun"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init-image", dest="init_image", default="")
    ap.add_argument("--strength", type=_bounded_strength, default=1.0)
    ap.add_argument("--mask", default="")
    ap.add_argument("--control-image", dest="control_image", default="")
    ap.add_argument("--control-strength", dest="control_strength", type=_nonnegative_float, default=1.0)
    ap.add_argument("--task", default="txt2img", choices=("txt2img", "img2img", "inpaint", "control"))
    ap.add_argument("--latent-only", dest="latent_only", action="store_true", help="Write final latent tensor instead of decoding through VAE.")
    ap.add_argument("--fake-vae", dest="fake_vae", action="store_true", help="Use deterministic fake VAE decoder for smoke tests.")
    ap.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    args = ap.parse_args()
    run_sample(SampleOptions(**vars(args)))


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
