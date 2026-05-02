from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveConfig:
    objective: str = "rectified_flow"
    prediction_type: str = "flow_velocity"

