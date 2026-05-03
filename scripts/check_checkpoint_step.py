from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusion.utils import load_ckpt


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate checkpoint step metadata.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file.")
    parser.add_argument("--step", required=True, type=int, help="Expected checkpoint step.")
    args = parser.parse_args()

    path = Path(args.ckpt)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run the matching train target before this check."
        )
    ckpt = load_ckpt(str(path), torch.device("cpu"))
    actual = int(ckpt.get("step", -1))
    if actual != int(args.step):
        raise RuntimeError(f"Expected {path} to contain step={args.step}, got step={actual}.")
    print(f"[OK] {path} contains step={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
