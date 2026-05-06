from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TargetSpec:
    """Config target registration metadata."""

    name: str
    default_path: str


TARGET_REGISTRY: dict[str, TargetSpec] = {
    "train": TargetSpec(name="train", default_path="configs/train.kdl"),
    "sample": TargetSpec(name="sample", default_path="configs/sample.kdl"),
    "webui": TargetSpec(name="webui", default_path="configs/webui.kdl"),
    "eval": TargetSpec(name="eval", default_path="configs/eval.kdl"),
    "cache": TargetSpec(name="cache", default_path="configs/cache.kdl"),
}


def validate_target_name(target: str) -> str:
    """Validate and normalize a target name."""
    normalized = str(target).strip()
    if normalized not in TARGET_REGISTRY:
        allowed = ", ".join(sorted(TARGET_REGISTRY))
        raise ValueError(f"Unknown config target {target!r}. Allowed: {allowed}.")
    return normalized


def target_default_path(target: str) -> str:
    """Return registered default config path for a target."""
    return TARGET_REGISTRY[validate_target_name(target)].default_path


def target_registry_snapshot() -> dict[str, dict[str, Any]]:
    """Return a serializable copy of the target registry."""
    return {
        name: {"name": spec.name, "default_path": spec.default_path}
        for name, spec in TARGET_REGISTRY.items()
    }
