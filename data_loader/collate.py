from __future__ import annotations

from typing import Callable, Optional

import torch


def collate_with_tokenizer(
    batch,
    *,
    latent_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
):
    # В режиме fallback элементы могут содержать флаг is_latent.
    if len(batch) == 0:
        raise RuntimeError("Empty batch.")
    item_len = len(batch[0])
    has_tokens = item_len >= 3
    has_flag = item_len >= 4 or (not has_tokens and item_len >= 2)

    if not has_tokens and not has_flag:
        imgs = torch.stack([b[0] for b in batch], dim=0)
        return imgs, None, None

    latents: list[Optional[torch.Tensor]] = [None] * len(batch)
    to_encode: list[torch.Tensor] = []
    encode_indices: list[int] = []

    for idx, item in enumerate(batch):
        if has_tokens:
            x, ids, mask, is_latent = item[:4]
        else:
            x, is_latent = item[:2]
        if is_latent:
            latents[idx] = x
        else:
            to_encode.append(x)
            encode_indices.append(idx)

    if to_encode:
        if latent_encoder is None:
            raise RuntimeError("latent_encoder is required for fallback encoding.")
        encoded = latent_encoder(torch.stack(to_encode, dim=0))
        for idx, z in zip(encode_indices, encoded):
            latents[idx] = z

    if any(x is None for x in latents):
        raise RuntimeError("Failed to assemble latent batch.")

    imgs = torch.stack([x for x in latents if x is not None], dim=0)
    if not has_tokens:
        return imgs, None, None
    ids = torch.stack([b[1] for b in batch], dim=0)
    mask = torch.stack([b[2] for b in batch], dim=0)
    return imgs, ids, mask
