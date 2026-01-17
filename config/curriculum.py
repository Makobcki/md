from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumConfig:
    curriculum_enabled: bool
    curriculum_steps: int
    curriculum_require_one_person: bool
    curriculum_prefer_solo: bool
    curriculum_exclude_multi: bool
    curriculum_solo_weight: float
    curriculum_non_solo_weight: float
