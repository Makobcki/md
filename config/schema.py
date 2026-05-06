from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sample.api import SampleOptions


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def _sampler_name(value: str) -> str:
    aliases = {
        "rf_euler": "flow_euler",
        "euler": "flow_euler",
        "flow_euler": "flow_euler",
        "rf_heun": "flow_heun",
        "heun": "flow_heun",
        "flow_heun": "flow_heun",
    }
    return aliases.get(value, value)


@dataclass(frozen=True)
class SampleConfig:
    """Typed sample target config used to build ``SampleOptions``."""

    target: str
    version: int
    options: SampleOptions
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleConfig:
        model = _section(data, "model")
        sampling = _section(data, "sampling")
        sampler = _section(data, "sampler")
        prompt = _section(data, "prompt")
        output = _section(data, "output")
        image = _section(data, "image")

        out = str(output.get("path", output.get("file", "")) or "")
        if not out:
            out_dir = str(output.get("dir", "outputs/sample") or "outputs/sample")
            out = f"{out_dir.rstrip('/')}/sample.png"

        checkpoint = str(model.get("checkpoint", data.get("ckpt", "")) or "")
        cfg_value = sampling.get(
            "cfg",
            sampling.get("cfg_scale", sampling.get("guidance_scale", data.get("cfg", 5.0))),
        )
        sampler_value = sampler.get(
            "name",
            sampling.get("sampler", data.get("sampler", "flow_heun")),
        )
        seed_value = sampling.get("seed", data.get("seed", 42))
        shift_value = sampling.get("shift", data.get("shift"))
        width_value = image.get("width", data.get("width"))
        height_value = image.get("height", data.get("height"))

        options = SampleOptions(
            ckpt=checkpoint,
            out=out,
            n=int(sampling.get("n", sampling.get("batch_size", data.get("n", 1)))),
            steps=int(sampling.get("steps", data.get("steps", 30))),
            prompt=str(prompt.get("text", data.get("prompt", "")) or ""),
            neg_prompt=str(prompt.get("negative", data.get("neg_prompt", "")) or ""),
            cfg=float(cfg_value),
            sampler=_sampler_name(str(sampler_value)),
            seed=None if seed_value is None else int(seed_value),
            shift=None if shift_value is None else float(shift_value),
            device=str(sampling.get("device", data.get("device", "auto"))),
            init_image=str(image.get("init", data.get("init_image", "")) or ""),
            strength=float(image.get("strength", data.get("strength", 1.0))),
            mask=str(image.get("mask", data.get("mask", "")) or ""),
            task=str(sampling.get("task", data.get("task", "txt2img"))),
            control_image=str(image.get("control", data.get("control_image", "")) or ""),
            control_strength=float(
                image.get("control_strength", data.get("control_strength", 1.0))
            ),
            control_type=str(image.get("control_type", data.get("control_type", "image"))),
            latent_only=bool(sampling.get("latent_only", data.get("latent_only", False))),
            fake_vae=bool(sampling.get("fake_vae", data.get("fake_vae", False))),
            use_ema=bool(sampling.get("use_ema", data.get("use_ema", True))),
            width=None if width_value is None else int(width_value),
            height=None if height_value is None else int(height_value),
        )
        options.validate()
        return cls(
            target=str(data.get("target", "sample")),
            version=int(data.get("version", 1)),
            options=options,
            raw=dict(data),
        )


@dataclass(frozen=True)
class WebUIConfig:
    """Typed WebUI target config."""

    target: str
    version: int
    host: str
    port: int
    auto_open: bool
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebUIConfig:
        webui = _section(data, "webui")
        return cls(
            target=str(data.get("target", "webui")),
            version=int(data.get("version", 1)),
            host=str(webui.get("host", data.get("host", "127.0.0.1"))),
            port=int(webui.get("port", data.get("port", 7860))),
            auto_open=bool(webui.get("auto_open", data.get("auto_open", True))),
            raw=dict(data),
        )


@dataclass(frozen=True)
class EvalConfig:
    """Typed eval target config."""

    target: str
    version: int
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        return cls(
            target=str(data.get("target", "eval")),
            version=int(data.get("version", 1)),
            raw=dict(data),
        )


@dataclass(frozen=True)
class CacheConfig:
    """Typed cache target config."""

    target: str
    version: int
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheConfig:
        return cls(
            target=str(data.get("target", "cache")),
            version=int(data.get("version", 1)),
            raw=dict(data),
        )
