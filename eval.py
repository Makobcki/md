from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    eval_prompts_file: str
    eval_every: int
    eval_seed: int
    eval_sampler: str
    eval_steps: int
    eval_cfg: float
    eval_n: int
