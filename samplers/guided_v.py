from __future__ import annotations

from typing import  Optional

import torch

from diffusion.model import UNet


def _guided_v(
    model: UNet,
    x: torch.Tensor,
    t: torch.Tensor,
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    self_cond: Optional[torch.Tensor],
    cfg_scale: float,
    uncond_ids: Optional[torch.Tensor],
    uncond_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    if txt_ids is None or txt_mask is None:
        return model(x, t, None, None, self_cond)
    v_cond = model(x, t, txt_ids, txt_mask, self_cond)
    if uncond_ids is not None and uncond_mask is not None and cfg_scale != 1.0:
        v_un = model(x, t, uncond_ids, uncond_mask, self_cond)
        return v_un + cfg_scale * (v_cond - v_un)
    return v_cond
