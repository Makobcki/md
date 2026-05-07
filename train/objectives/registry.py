from __future__ import annotations

from collections.abc import Callable

ObjectiveHandler = Callable[..., object]


def _rectified_flow_handler(**kwargs: object) -> object:
    from diffusion.objectives import RectifiedFlowObjective

    return RectifiedFlowObjective(**kwargs)


def _next_scale_prediction_handler(**kwargs: object) -> object:
    from model.var import next_scale_cross_entropy

    return next_scale_cross_entropy


_OBJECTIVE_HANDLERS: dict[str, ObjectiveHandler] = {
    "rectified_flow": _rectified_flow_handler,
    "next_scale_prediction": _next_scale_prediction_handler,
}


def get_objective_handler(name: str) -> ObjectiveHandler:
    try:
        return _OBJECTIVE_HANDLERS[str(name)]
    except KeyError as exc:
        allowed = ", ".join(sorted(_OBJECTIVE_HANDLERS))
        raise ValueError(f"Unknown objective handler {name!r}. Allowed: {allowed}.") from exc
