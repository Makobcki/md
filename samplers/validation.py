from __future__ import annotations


def validate_steps(steps: int) -> int:
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps must be >= 1.")
    return steps
