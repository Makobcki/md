from __future__ import annotations

import ast
import json
from typing import Any


def parse_override_value(raw: str) -> Any:
    """Parse a CLI ``--set`` value into a Python scalar/list/dict."""
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return ast.literal_eval(text)
    try:
        if any(marker in text for marker in (".", "e", "E")):
            return float(text)
        return int(text, 10)
    except ValueError:
        return raw


def set_nested(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path in a nested mapping."""
    keys = [part for part in path.split(".") if part]
    if not keys:
        raise ValueError("Override path must not be empty.")
    current: dict[str, Any] = data
    for key in keys[:-1]:
        existing = current.get(key)
        if existing is None:
            child: dict[str, Any] = {}
            current[key] = child
            current = child
            continue
        if not isinstance(existing, dict):
            raise ValueError(
                f"Cannot apply override {path!r}: {key!r} already contains a scalar value."
            )
        current = existing
    current[keys[-1]] = value


def parse_set_overrides(values: list[str] | tuple[str, ...] | None) -> dict[str, Any]:
    """Parse repeated ``--set section.key=value`` CLI values."""
    overrides: dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        path, raw_value = item.split("=", 1)
        set_nested(overrides, path.strip(), parse_override_value(raw_value))
    return overrides
