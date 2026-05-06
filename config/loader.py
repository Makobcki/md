from __future__ import annotations

from pathlib import Path
from typing import Any

from .discovery import default_config_path
from .formats.yaml_loader import load_yaml
from .overrides import parse_set_overrides
from .resolver import ResolvedConfig, resolve_config
from .schema import CacheConfig, EvalConfig, SampleConfig, WebUIConfig
from .train import TrainConfig


def _resolve_or_load_yaml(
    path: str | Path | None,
    *,
    target: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(path) if path else default_config_path(target)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        # Compatibility path. YAML configs predate target metadata, so they are
        # accepted for the requested target and then converted through the same
        # TrainConfig/from_dict flattening layer where applicable.
        data = load_yaml(config_path)
        if overrides:
            from .resolver import recursive_merge

            data = recursive_merge(data, overrides)
        data.setdefault("target", target)
        data.setdefault("version", 1)
        return data
    return resolve_config(config_path, overrides=overrides, expected_target=target).data


def resolve_target_config(
    target: str,
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Public API for resolving KDL target configs."""

    config_path = Path(path) if path else default_config_path(target)
    return resolve_config(config_path, overrides=overrides, expected_target=target)


def build_train_config(data: dict[str, Any]) -> TrainConfig:
    """Build the legacy flat TrainConfig from a nested target config."""

    return TrainConfig.from_dict(data)


def load_train_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    target: str = "train",
) -> TrainConfig:
    """Load a train-compatible config.

    ``target`` defaults to ``train`` but cache preparation commands pass
    ``cache`` so they can use ``configs/cache.kdl`` while still receiving the
    flat ``TrainConfig`` required by existing cache code.
    """

    return build_train_config(_resolve_or_load_yaml(path, target=target, overrides=overrides))


def load_cache_train_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load the cache target as a train-compatible config."""

    return load_train_config(path, overrides=overrides, target="cache")


def load_sample_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SampleConfig:
    """Load and validate a sample target config."""

    data = _resolve_or_load_yaml(path, target="sample", overrides=overrides)
    return SampleConfig.from_dict(data)


def load_webui_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> WebUIConfig:
    """Load and validate a WebUI target config."""

    data = _resolve_or_load_yaml(path, target="webui", overrides=overrides)
    return WebUIConfig.from_dict(data)


def load_eval_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EvalConfig:
    """Load and validate an eval target config."""

    data = _resolve_or_load_yaml(path, target="eval", overrides=overrides)
    return EvalConfig.from_dict(data)


def load_cache_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> CacheConfig:
    """Load and validate a cache target config."""

    data = _resolve_or_load_yaml(path, target="cache", overrides=overrides)
    return CacheConfig.from_dict(data)


def parse_cli_overrides(values: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    """Compatibility wrapper for parsing repeated ``--set`` values."""

    return parse_set_overrides(values)
