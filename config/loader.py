from __future__ import annotations

from pathlib import Path
from typing import Any

from .discovery import default_config_path
from .overrides import parse_set_overrides
from .resolver import ResolvedConfig, resolve_config
from .schema import CacheConfig, EvalConfig, SampleConfig, WebUIConfig
from .train import TrainConfig


def _resolve_target_data(
    path: str | Path | None,
    *,
    target: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a KDL target config and return its public data."""

    config_path = Path(path) if path else default_config_path(target)
    return resolve_config(config_path, overrides=overrides, expected_target=target).data


def resolve_target_config(
    target: str,
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Resolve a KDL target config with presets and CLI overrides."""

    config_path = Path(path) if path else default_config_path(target)
    return resolve_config(config_path, overrides=overrides, expected_target=target)


def build_train_config(data: dict[str, Any]) -> TrainConfig:
    """Build the runtime ``TrainConfig`` from a resolved target config."""

    return TrainConfig.from_dict(data)


def load_train_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    target: str = "train",
) -> TrainConfig:
    """Load a train-compatible KDL config.

    ``target`` defaults to ``train``. Cache preparation commands pass ``cache``
    so they can use ``configs/cache.kdl`` while the current cache code still
    receives the flat ``TrainConfig`` runtime object.
    """

    return build_train_config(_resolve_target_data(path, target=target, overrides=overrides))


def load_cache_train_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load the cache target as a train-compatible runtime config."""

    return load_train_config(path, overrides=overrides, target="cache")


def load_sample_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SampleConfig:
    """Load and validate a sample target config."""

    data = _resolve_target_data(path, target="sample", overrides=overrides)
    return SampleConfig.from_dict(data)


def load_webui_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> WebUIConfig:
    """Load and validate a WebUI target config."""

    data = _resolve_target_data(path, target="webui", overrides=overrides)
    return WebUIConfig.from_dict(data)


def load_eval_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvalConfig:
    """Load and validate an eval target config."""

    data = _resolve_target_data(path, target="eval", overrides=overrides)
    return EvalConfig.from_dict(data)


def load_cache_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> CacheConfig:
    """Load and validate a cache target config."""

    data = _resolve_target_data(path, target="cache", overrides=overrides)
    return CacheConfig.from_dict(data)


def parse_cli_overrides(values: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    """Parse repeated ``--set section.key=value`` values."""

    return parse_set_overrides(values)
