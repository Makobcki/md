from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config.formats.kdl_loader import KdlParseError, load_kdl, loads_kdl
from config.loader import load_train_config
from config.resolver import resolve_config
from config.train import TrainConfig

from .atomic import atomic_write_text


def get_config_path(repo_root: Path) -> Path:
    override = os.environ.get("WEBUI_CONFIG_PATH")
    if override:
        return Path(override)
    return repo_root / "configs" / "train.kdl"


def read_config_text(repo_root: Path) -> str:
    path = get_config_path(repo_root)
    return path.read_text(encoding="utf-8")


def _ensure_kdl_config_document(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if not stripped:
        return {}
    if not stripped.startswith(("config ", "preset ")):
        raise ValueError("config editor accepts KDL documents only")
    return loads_kdl(text, source="<webui-config-editor>")


def _resolve_config_text(text: str, config_path: Path) -> dict[str, Any]:
    """Resolve editor text from a temporary sibling path for preset lookup."""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=config_path.parent,
        encoding="utf-8",
        prefix=f".{config_path.stem}.",
        suffix=".kdl",
    ) as temp_file:
        temp_file.write(text)
        temp_path = Path(temp_file.name)
    try:
        return resolve_config(temp_path, expected_target="train").data
    finally:
        temp_path.unlink(missing_ok=True)


def _parse_config_mapping(text: str, config_path: Path | None = None) -> dict[str, Any]:
    data = _ensure_kdl_config_document(text)
    if not data:
        return {}
    if config_path is not None:
        return _resolve_config_text(text, config_path)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def parse_config_text(text: str, config_path: Path | None = None) -> TrainConfig:
    data = _parse_config_mapping(text, config_path=config_path)
    return TrainConfig.from_dict(data)


def validate_config_text(text: str, config_path: Path | None = None) -> dict[str, Any]:
    cfg = parse_config_text(text, config_path=config_path)
    return asdict(cfg)


def write_config_text(repo_root: Path, text: str) -> dict[str, Any]:
    path = get_config_path(repo_root)
    try:
        cfg_dict = validate_config_text(text, config_path=path)
    except (KdlParseError, RuntimeError) as exc:
        raise ValueError(str(exc)) from exc
    atomic_write_text(path, text)
    return cfg_dict


def load_config_dict(repo_root: Path) -> dict[str, Any]:
    path = get_config_path(repo_root)
    return load_train_config(path).to_dict()


def _public_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _public_payload(item)
            for key, item in value.items()
            if not key.startswith("__")
        }
    if isinstance(value, list):
        return [_public_payload(item) for item in value]
    return value


def extract_used_presets(text: str) -> dict[str, list[str]]:
    """Return direct section-level preset aliases used by a KDL config."""

    data = _ensure_kdl_config_document(text)
    used: dict[str, list[str]] = {}
    for section, value in data.items():
        if section.startswith("__") or not isinstance(value, dict):
            continue
        uses = value.get("__uses__", [])
        if isinstance(uses, list):
            names = [str(item) for item in uses]
        elif uses:
            names = [str(uses)]
        else:
            names = []
        if names:
            used[section] = names
    return used


def _first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def _join_summary(parts: list[str]) -> str:
    return ", ".join(part for part in parts if part)


def _preset_summary(kind: str, payload: dict[str, Any]) -> str:
    if kind == "model":
        model = _first_dict(payload.get("model"))
        arch = _first_dict(model.get("architecture"))
        return _join_summary(
            [
                f"family={model.get('family')}" if model.get("family") else "",
                f"variant={model.get('variant')}" if model.get("variant") else "",
                f"image={arch.get('image_size') or payload.get('image_size')}"
                if arch.get("image_size") or payload.get("image_size")
                else "",
                f"hidden={arch.get('hidden_size')}" if arch.get("hidden_size") else "",
                f"depth={arch.get('depth')}" if arch.get("depth") else "",
            ]
        )
    if kind == "training":
        training = _first_dict(payload.get("training"))
        optimizer = _first_dict(training.get("optimizer"))
        precision = _first_dict(training.get("precision"))
        return _join_summary(
            [
                f"optimizer={optimizer.get('name')}" if optimizer.get("name") else "",
                f"dtype={precision.get('dtype') or payload.get('latent_dtype')}"
                if precision.get("dtype") or payload.get("latent_dtype")
                else "",
                f"batch={training.get('batch_size')}" if training.get("batch_size") else "",
                f"steps={training.get('max_steps')}" if training.get("max_steps") else "",
            ]
        )
    if kind == "data":
        return _join_summary(
            [
                f"image={payload.get('image_size')}" if payload.get("image_size") else "",
                f"latents={payload.get('latent_cache_dir')}"
                if payload.get("latent_cache_dir")
                else "",
                "text_cache" if payload.get("text_cache") else "",
            ]
        )
    if kind == "text":
        text = _first_dict(payload.get("text"))
        encoders = text.get("encoder", [])
        encoder_items = encoders if isinstance(encoders, list) else [encoders]
        names = [
            str(item.get("name"))
            for item in encoder_items
            if isinstance(item, dict) and item.get("name")
        ]
        return _join_summary(
            [
                f"text_dim={text.get('text_dim')}" if text.get("text_dim") else "",
                f"encoders={'+'.join(names)}" if names else "",
            ]
        )
    if kind == "sampler":
        sampler = _first_dict(payload.get("sampling"))
        return _join_summary(
            [
                f"sampler={sampler.get('sampler')}" if sampler.get("sampler") else "",
                f"shift={sampler.get('shift')}" if sampler.get("shift") else "",
            ]
        )
    if kind == "webui":
        webui = _first_dict(payload.get("webui"))
        return _join_summary(
            [
                f"host={webui.get('host')}" if webui.get("host") else "",
                f"port={webui.get('port')}" if webui.get("port") else "",
            ]
        )
    return ""


def list_config_presets(repo_root: Path) -> dict[str, Any]:
    """Return available KDL presets grouped by kind for the WebUI editor."""

    preset_root = repo_root / "configs" / "presets"
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(preset_root.glob("*/*.kdl")):
        raw = load_kdl(path)
        meta = raw.get("__meta__", {}) if isinstance(raw.get("__meta__"), dict) else {}
        kind = str(meta.get("kind") or path.parent.name)
        name = str(meta.get("name") or path.stem)
        version = meta.get("version")
        payload = _public_payload(raw)
        if not isinstance(payload, dict):
            payload = {}
        item = {
            "kind": kind,
            "name": name,
            "version": version,
            "path": str(path),
            "relative_path": str(path.relative_to(repo_root)),
            "summary": _preset_summary(kind, payload),
            "content": path.read_text(encoding="utf-8"),
        }
        groups.setdefault(kind, []).append(item)

    active = extract_used_presets(read_config_text(repo_root))
    return {"groups": groups, "active": active}
