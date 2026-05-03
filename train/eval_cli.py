from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train.eval import DEFAULT_EVAL_PROMPT_SETS, load_eval_prompt_bank
from train.eval_grid import DEFAULT_CFG_SWEEP, DEFAULT_SAMPLER_SWEEP, DEFAULT_SHIFT_SWEEP, DEFAULT_STEP_SWEEP, run_fixed_seed_eval_grids


def _parse_float_list(items: list[str] | None, default: tuple[float, ...]) -> list[float] | None:
    if items is None:
        return None
    if not items:
        return list(default)
    out: list[float] = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(float(part))
    return out


def _parse_int_list(items: list[str] | None, default: tuple[int, ...]) -> list[int] | None:
    if items is None:
        return None
    if not items:
        return list(default)
    out: list[int] = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def _parse_sampler_list(items: list[str] | None, default: tuple[str, ...]) -> list[str] | None:
    if items is None:
        return None
    if not items:
        return list(default)
    out: list[str] = []
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                if part not in {"flow_euler", "flow_heun"}:
                    raise argparse.ArgumentTypeError(f"unsupported sampler: {part}")
                out.append(part)
    return out


def build_prompt_bank_metadata(
    *,
    prompt_root: str | Path,
    prompt_sets: list[str] | None,
    count_per_set: int | None,
) -> dict[str, Any]:
    selected = prompt_sets or list(DEFAULT_EVAL_PROMPT_SETS)
    bank = load_eval_prompt_bank(selected, root=prompt_root, count_per_set=count_per_set)
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_root": str(prompt_root),
        "prompt_sets": [
            {"name": name, "count": len(prompts), "prompts": prompts}
            for name, prompts in bank.items()
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Read eval prompt banks or run fixed-seed MMDiT eval grids.")
    ap.add_argument("--prompt-root", default="data/eval_prompts")
    ap.add_argument("--prompt-set", action="append", default=None, help="Prompt set name without .txt. Can be repeated.")
    ap.add_argument("--count-per-set", type=int, default=None)
    ap.add_argument("--metadata-out", default="")
    ap.add_argument("--print", action="store_true", dest="print_json")

    ap.add_argument("--ckpt", default="", help="Checkpoint path. When provided, eval grids are generated.")
    ap.add_argument("--out-dir", default="runs/eval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sampler", choices=("flow_euler", "flow_heun"), default="flow_heun")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.5)
    ap.add_argument("--shift", type=float, default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--n-per-prompt", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fake-vae", action="store_true")
    ap.add_argument("--latent-only", action="store_true")
    ap.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    ap.add_argument("--cfg-sweep", nargs="*", default=None, metavar="CFG")
    ap.add_argument("--step-sweep", nargs="*", default=None, metavar="STEPS")
    ap.add_argument("--sampler-sweep", nargs="*", default=None, metavar="SAMPLER")
    ap.add_argument("--shift-sweep", nargs="*", default=None, metavar="SHIFT")
    args = ap.parse_args()

    if args.ckpt:
        result = run_fixed_seed_eval_grids(
            ckpt=args.ckpt,
            out_dir=args.out_dir,
            prompt_root=args.prompt_root,
            prompt_sets=args.prompt_set,
            count_per_set=args.count_per_set,
            seed=int(args.seed),
            sampler=str(args.sampler),
            steps=int(args.steps),
            cfg=float(args.cfg),
            n_per_prompt=int(args.n_per_prompt),
            shift=args.shift,
            device=str(args.device),
            fake_vae=bool(args.fake_vae),
            latent_only=bool(args.latent_only),
            cfg_values=_parse_float_list(args.cfg_sweep, DEFAULT_CFG_SWEEP),
            step_values=_parse_int_list(args.step_sweep, DEFAULT_STEP_SWEEP),
            sampler_values=_parse_sampler_list(args.sampler_sweep, DEFAULT_SAMPLER_SWEEP),
            shift_values=_parse_float_list(args.shift_sweep, DEFAULT_SHIFT_SWEEP),
            resolution=args.resolution,
            use_ema=bool(args.use_ema),
        )
        if args.metadata_out:
            out = Path(args.metadata_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.print_json:
            print(json.dumps(result.metadata, indent=2, ensure_ascii=False))
        return

    metadata = build_prompt_bank_metadata(
        prompt_root=args.prompt_root,
        prompt_sets=args.prompt_set,
        count_per_set=args.count_per_set,
    )
    if args.metadata_out:
        out = Path(args.metadata_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.print_json or not args.metadata_out:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
