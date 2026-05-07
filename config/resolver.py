from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .discovery import project_root
from .formats.kdl_loader import load_kdl

SECTION_PRESET_KINDS: dict[str, str] = {
    "model": "model",
    "sampler": "sampler",
    "sampling": "sampler",
    "training": "training",
    "data": "data",
    "text": "text",
    "webui": "webui",
}
PRESET_ALIAS_KINDS = tuple(SECTION_PRESET_KINDS)


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


def _strip_internal_value(value: Any) -> Any:
    """Return a deep copy without parser-only ``__*`` keys."""

    if isinstance(value, dict):
        return {
            key: _strip_internal_value(item)
            for key, item in value.items()
            if not key.startswith("__")
        }
    if isinstance(value, list):
        return [_strip_internal_value(item) for item in value]
    return deepcopy(value)


def _strip_internal(data: dict[str, Any]) -> dict[str, Any]:
    """Return public config data without parser-only inheritance metadata."""

    stripped = _strip_internal_value(data)
    return stripped if isinstance(stripped, dict) else {}


def _resolve_use_path(base_path: Path, use_path: str) -> Path:
    candidate = Path(use_path)
    if candidate.is_absolute():
        return candidate
    return (base_path.parent / candidate).resolve()


def _preset_kind_for_section(section: str) -> str:
    try:
        return SECTION_PRESET_KINDS[section]
    except KeyError as exc:
        allowed = ", ".join(sorted(SECTION_PRESET_KINDS))
        raise RuntimeError(
            f"Section {section!r} cannot inherit presets with use. Allowed sections: {allowed}."
        ) from exc


def _preset_root_for(config_path: Path) -> Path:
    """Return the preset root nearest to a config or preset file."""

    resolved = config_path.resolve()
    if "presets" in resolved.parts:
        parts = list(resolved.parts)
        preset_index = len(parts) - 1 - parts[::-1].index("presets")
        return Path(*parts[: preset_index + 1])
    local_root = resolved.parent / "presets"
    if local_root.exists():
        return local_root
    project_presets = project_root(resolved) / "configs" / "presets"
    if project_presets.exists():
        return project_presets
    return local_root


def _preset_path_for_alias(config_path: Path, kind: str, name: str) -> Path:
    if not name:
        raise RuntimeError(f"Empty preset alias for {kind}.use")
    root = _preset_root_for(config_path)
    return (root / kind / f"{name}.kdl").resolve()


def _resolve_scoped_use_path(config_path: Path, section: str, use_path: str) -> Path:
    """Resolve a section-level use as a preset alias or an explicit KDL path."""

    candidate = Path(use_path)
    if candidate.is_absolute() or candidate.suffix or len(candidate.parts) > 1:
        return _resolve_use_path(config_path, use_path)
    return _preset_path_for_alias(config_path, _preset_kind_for_section(section), use_path)


def _validate_preset_kind(raw: dict[str, Any], *, expected_kind: str, source: Path) -> None:
    """Validate that a resolved scoped use points to a preset of the right kind."""

    if str(raw.get("__kind__", "")) != "preset":
        raise RuntimeError(f"Section use resolved to a non-preset file: {source}")
    meta = raw.get("__meta__") if isinstance(raw.get("__meta__"), dict) else {}
    actual_kind = str(meta.get("kind", "") or "")
    if actual_kind and actual_kind != expected_kind:
        raise RuntimeError(
            f"Preset kind mismatch for {source}: expected kind={expected_kind!r}, "
            f"got {actual_kind!r}."
        )


def _collect_scoped_uses(raw: dict[str, Any]) -> list[tuple[str, str]]:
    """Collect direct ``section { use "name" }`` references in document order."""

    scoped_uses: list[tuple[str, str]] = []
    for section, value in raw.items():
        if section.startswith("__") or not isinstance(value, dict):
            continue
        uses = value.get("__uses__", [])
        if isinstance(uses, list):
            scoped_uses.extend((section, str(use_path)) for use_path in uses)
        elif uses:
            scoped_uses.append((section, str(uses)))
    return scoped_uses


def _resolve_scoped_uses(
    raw: dict[str, Any],
    path: Path,
    *,
    stack: tuple[Path, ...],
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Resolve scoped preset references and merge their full preset payloads."""

    sources: list[Path] = []
    merged: dict[str, Any] = {}
    for section, use_path in _collect_scoped_uses(raw):
        preset_path = _resolve_scoped_use_path(path, section, use_path)
        expected_kind = _preset_kind_for_section(section)
        preset_data, preset_sources, preset_raw = _resolve_file(
            preset_path,
            stack=stack,
            allow_preset_root=True,
        )
        _validate_preset_kind(preset_raw, expected_kind=expected_kind, source=preset_path)
        merged = recursive_merge(merged, preset_data)
        sources.extend(preset_sources)
    return merged, tuple(sources)


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
    child_stack = (*stack, resolved)

    for use_path in raw.get("__uses__", []):
        child_path = _resolve_use_path(resolved, str(use_path))
        child_data, child_sources, _ = _resolve_file(
            child_path,
            stack=child_stack,
            allow_preset_root=True,
        )
        merged = recursive_merge(merged, child_data)
        sources.extend(child_sources)

    scoped_data, scoped_sources = _resolve_scoped_uses(raw, resolved, stack=child_stack)
    merged = recursive_merge(merged, scoped_data)
    sources.extend(scoped_sources)

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


def _as_use_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _alias_presets(
    config_path: Path,
    overrides: dict[str, Any],
) -> list[tuple[str, Path]]:
    presets: list[tuple[str, Path]] = []
    for section in PRESET_ALIAS_KINDS:
        preset_kind = _preset_kind_for_section(section)
        for field in ("preset", "use", "__uses__"):
            value = _get_nested(overrides, (section, field))
            for name in _as_use_list(value):
                presets.append(
                    (preset_kind, _preset_path_for_alias(config_path, preset_kind, name))
                )
    return presets


def _validate_target(
    data: dict[str, Any],
    expected_target: str | None,
    source: Path,
) -> tuple[str, int]:
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
            f"Config target mismatch for {source}: expected target={expected_target!r}, "
            f"got {target!r}"
        )
    return target, version


def resolve_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    expected_target: str | None = None,
) -> ResolvedConfig:
    """Resolve a target config, scoped preset uses, aliases, and CLI overrides."""

    config_path = Path(path).resolve()
    raw = load_raw_config(config_path)
    if str(raw.get("__kind__", "config")) == "preset":
        raise RuntimeError(f"Preset files cannot be used as launch configs: {config_path}")
    target, version = _validate_target(raw, expected_target, config_path)

    data, sources, _ = _resolve_file(config_path, stack=(), allow_preset_root=False)
    overrides = deepcopy(overrides or {})

    for expected_kind, preset_path in _alias_presets(config_path, overrides):
        preset_data, preset_sources, preset_raw = _resolve_file(
            preset_path,
            stack=(config_path,),
            allow_preset_root=True,
        )
        _validate_preset_kind(preset_raw, expected_kind=expected_kind, source=preset_path)
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
