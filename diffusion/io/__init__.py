from __future__ import annotations

"""I/O helpers.

This module intentionally avoids importing checkpoint helpers eagerly because
``diffusion.io.ckpt`` imports torch. WebUI/log-only imports need the lightweight
JSONL event helpers without paying the torch import cost or triggering torch
import failures in minimal environments.
"""

from .events import EventBus, JsonlFileSink, StdoutJsonSink, format_event_line

_CKPT_EXPORTS = {
    "CKPT_FORMAT_VERSION",
    "load_ckpt",
    "normalize_state_dict_for_keys",
    "normalize_state_dict_for_model",
    "resolve_resume_path",
    "sanitize_ckpt",
    "sanitize_state_dict",
    "save_ckpt",
    "strip_state_dict_prefixes",
}

__all__ = [
    *_CKPT_EXPORTS,
    "EventBus",
    "JsonlFileSink",
    "StdoutJsonSink",
    "format_event_line",
]


def __getattr__(name: str):
    if name in _CKPT_EXPORTS:
        from . import ckpt as _ckpt

        value = getattr(_ckpt, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
