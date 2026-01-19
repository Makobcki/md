from __future__ import annotations

import argparse
from dataclasses import replace

from config.loader import load_train_config
from config.train import TrainConfig
from train.runner import run


def _apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if args.resume:
        cfg = replace(cfg, resume_ckpt=str(args.resume))
    if args.ckpt_keep_last is not None:
        cfg = replace(cfg, ckpt_keep_last=int(args.ckpt_keep_last))
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/train.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt-keep-last", type=int, default=None)
    args = ap.parse_args()

    cfg = load_train_config(args.config)
    cfg = _apply_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
