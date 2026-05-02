from __future__ import annotations

import torch

from model.text.conditioning import TextConditioning


def concat_text_conditioning(items: list[TextConditioning]) -> TextConditioning:
    return TextConditioning(
        tokens=torch.cat([x.tokens for x in items], dim=0),
        mask=torch.cat([x.mask for x in items], dim=0),
        pooled=torch.cat([x.pooled for x in items], dim=0),
        is_uncond=(
            torch.cat([x.is_uncond for x in items], dim=0)
            if all(x.is_uncond is not None for x in items)
            else None
        ),
    )


@torch.no_grad()
def cfg_predict(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: TextConditioning,
    uncond: TextConditioning | None = None,
    scale: float = 1.0,
    **model_kwargs,
) -> torch.Tensor:
    if uncond is None or scale == 1.0:
        return model(x, t, cond, **model_kwargs)
    x_in = torch.cat([x, x], dim=0)
    t_in = torch.cat([t, t], dim=0)
    text_in = concat_text_conditioning([uncond, cond])
    kwargs_in = {}
    for key, value in model_kwargs.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == x.shape[:1]:
            kwargs_in[key] = torch.cat([value, value], dim=0)
        else:
            kwargs_in[key] = value
    pred_uncond, pred_cond = model(x_in, t_in, text_in, **kwargs_in).chunk(2, dim=0)
    return pred_uncond + float(scale) * (pred_cond - pred_uncond)

