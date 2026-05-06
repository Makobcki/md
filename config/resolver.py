from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .formats.kdl_loader import load_kdl

PRESET_ALIAS_KINDS = ("model", "sampler", "training", "data", "webui")


@dataclass(frozen=True)
class ResolvedConfig:
    """Resolved config payload with origin metadata."""

    data: dict[str, Any]
    target: str
    version: int
    path: Path
    sources: tuple[Path, ...]


def recursive_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge dictionaries recursively with scalar/list replacement semantics."""
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
            and not key.startswith("__")
        ):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Load a KDL config document."""
    file_path = Path(path)
    if file_path.suffix.lower() != ".kdl":
        raise RuntimeError(f"Unsupported config extension for {file_path}; expected .kdl")
    return load_kdl(file_path)


def _strip_internal(data: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in data.items() if not key.startswith("__")}


def _resolve_use_path(base_path: Path, use_path: str) -> Path:
    candidate = Path(use_path)
    if candidate.is_absolute():
        return candidate
    return (base_path.parent / candidate).resolve()


def _resolve_file(
    path: Path,
    *,
    stack: tuple[Path, ...],
    allow_preset_root: bool,
) -> tuple[dict[str, Any], tuple[Path, ...], dict[str, Any]]:
    resolved = path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved))
        raise RuntimeError(f"Cyclic config inheritance detected: {chain}")
    raw = load_raw_config(resolved)
    kind = str(raw.get("__kind__", "config"))
    if kind == "preset" and not allow_preset_root:
        raise RuntimeError(f"Preset files cannot be used as launch configs: {resolved}")
    if kind not in {"config", "preset"}:
        raise RuntimeError(f"Unsupported config root kind {kind!r} in {resolved}")

    sources: list[Path] = []
    merged: dict[str, Any] = {}
    for use_path in raw.get("__uses__", []):
        child_path = _resolve_use_path(resolved, str(use_path))
        child_data, child_sources, _ = _resolve_file(
            child_path,
            stack=(*stack, resolved),
            allow_preset_root=True,
        )
        merged = recursive_merge(merged, child_data)
        sources.extend(child_sources)

    merged = recursive_merge(merged, _strip_internal(raw))
    sources.append(resolved)
    return merged, tuple(sources), raw


def _get_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _preset_path_for_alias(config_path: Path, kind: str, name: str) -> Path:
    if not name:
        raise RuntimeError(f"Empty preset alias for {kind}.preset")
    root = config_path.parent / "presets"
    return (root / kind / f"{name}.kdl").resolve()


def _alias_presets(
    config_path: Path,
    overrides: dict[str, Any],
) -> list[tuple[str, Path]]:
    presets: list[tuple[str, Path]] = []
    for kind in PRESET_ALIAS_KINDS:
        name = _get_nested(overrides, (kind, "preset"))
        if name is not None:
            presets.append((kind, _preset_path_for_alias(config_path, kind, str(name))))
    return presets


def _validate_target(data: dict[str, Any], expected_target: str | None, source: Path) -> tuple[str, int]:
    meta = data.get("__meta__") if isinstance(data.get("__meta__"), dict) else {}
    target = str(meta.get("target", data.get("target", "")) or "")
    version = int(meta.get("version", data.get("version", 1)) or 1)
    if not target:
        raise RuntimeError(
            f"Missing config target metadata in {source}; expected root like "
            'config target="train" version=2.'
        )
    if expected_target and target != expected_target:
        raise RuntimeError(
            f"Config target mismatch for {source}: expected target={expected_target!r}, got {target!r}"
        )
    return target, version


def resolve_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    expected_target: str | None = None,
) -> ResolvedConfig:
    """Resolve a target config, presets, preset aliases, and CLI overrides."""
    config_path = Path(path).resolve()
    raw = load_raw_config(config_path)
    if str(raw.get("__kind__", "config")) == "preset":
        raise RuntimeError(f"Preset files cannot be used as launch configs: {config_path}")
    target, version = _validate_target(raw, expected_target, config_path)

    data, sources, _ = _resolve_file(config_path, stack=(), allow_preset_root=False)
    overrides = deepcopy(overrides or {})

    for _kind, preset_path in _alias_presets(config_path, overrides):
        preset_data, preset_sources, preset_raw = _resolve_file(
            preset_path,
            stack=(config_path,),
            allow_preset_root=True,
        )
        if str(preset_raw.get("__kind__", "")) != "preset":
            raise RuntimeError(f"Preset alias resolved to a non-preset file: {preset_path}")
        data = recursive_merge(data, preset_data)
        sources = (*sources, *preset_sources)

    if overrides:
        data = recursive_merge(data, overrides)

    # Keep target/version in the public resolved payload for debugging and
    # validation, but remove parser-only fields.
    data["target"] = target
    data["version"] = version
    return ResolvedConfig(
        data=data,
        target=target,
        version=version,
        path=config_path,
        sources=tuple(dict.fromkeys(sources)),
    )
