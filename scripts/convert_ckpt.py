#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch


def strip_prefix(sd: dict, prefix: str) -> dict:
    if not sd:
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--prefix", default="_orig_mod.")
    args = ap.parse_args()

    ck = torch.load(args.inp, map_location="cpu")

    if "model" in ck:
        ck["model"] = strip_prefix(ck["model"], args.prefix)
    if "ema" in ck and isinstance(ck["ema"], dict):
        ck["ema"] = strip_prefix(ck["ema"], args.prefix)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    print("[OK] saved:", args.out)


if __name__ == "__main__":
    main()
