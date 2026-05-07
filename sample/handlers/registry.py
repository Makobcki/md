from __future__ import annotations

from collections.abc import Callable

SampleHandler = Callable[..., object]


def _latent_flow_handler(**kwargs: object) -> object:
    sampler = str(kwargs.pop("sampler", "flow_heun"))
    if sampler == "flow_euler":
        from samplers import sample_flow_euler

        return sample_flow_euler(**kwargs)
    if sampler == "flow_heun":
        from samplers import sample_flow_heun

        return sample_flow_heun(**kwargs)
    raise ValueError("latent_flow sampler must be one of: flow_euler, flow_heun.")


def _var_autoregressive_handler(**kwargs: object) -> object:
    from model.var import deterministic_decode

    return deterministic_decode(**kwargs)


_SAMPLE_HANDLERS: dict[str, SampleHandler] = {
    "latent_flow": _latent_flow_handler,
    "var_autoregressive": _var_autoregressive_handler,
}


def get_sample_handler(name: str) -> SampleHandler:
    try:
        return _SAMPLE_HANDLERS[str(name)]
    except KeyError as exc:
        allowed = ", ".join(sorted(_SAMPLE_HANDLERS))
        raise ValueError(f"Unknown sample handler {name!r}. Allowed: {allowed}.") from exc
