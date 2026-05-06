from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG_BY_TARGET: dict[str, str] = {
    "train": "configs/train.kdl",
    "sample": "configs/sample.kdl",
    "webui": "configs/webui.kdl",
    "eval": "configs/eval.kdl",
    "cache": "configs/cache.kdl",
}

CACHE_TARGETS = {
    "cache-validate": "cache",
    "prepare-latents": "cache",
    "prepare-text-cache": "cache",
    "prepare-training-cache": "cache",
}


def project_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking up to ``pyproject.toml``."""
    current = Path.cwd() if start is None else Path(start).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def default_config_path(target: str, *, root: str | Path | None = None) -> Path:
    """Return the default KDL config path for a target."""
    normalized = CACHE_TARGETS.get(target, target)
    try:
        relative = DEFAULT_CONFIG_BY_TARGET[normalized]
    except KeyError as exc:
        allowed = ", ".join(sorted(DEFAULT_CONFIG_BY_TARGET))
        raise ValueError(f"Unknown config target {target!r}. Allowed: {allowed}.") from exc
    base = project_root(root)
    return base / relative
