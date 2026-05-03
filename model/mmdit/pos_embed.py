from __future__ import annotations

import torch


def get_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        return torch.zeros((height * width, dim), device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([out_y.sin(), out_y.cos(), out_x.sin(), out_x.cos()], dim=1)
    return pos.to(dtype=dtype)


def add_2d_pos_embed(tokens: torch.Tensor, grid_hw: tuple[int, int], mode: str) -> torch.Tensor:
    if mode in {"none", "rope_2d"}:
        # RoPE is applied to attention q/k, not additively to tokens.
        return tokens
    if mode != "sincos_2d":
        raise ValueError(f"Unsupported positional embedding mode: {mode}")
    h, w = grid_hw
    pos = get_2d_sincos_pos_embed(h, w, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
    return tokens + pos.unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _axis_rope_angles(length: int, dim: int, *, device: torch.device) -> torch.Tensor:
    idx = torch.arange(length, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / max(dim, 1)))
    freqs = torch.einsum("n,d->nd", idx, inv_freq)
    return torch.repeat_interleave(freqs, 2, dim=-1)


def _build_2d_rope_angles(
    grid_hw: tuple[int, int],
    head_dim: int,
    chunks: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Return repeated 2D RoPE angles as [chunks * H * W, rope_dim].

    The first half of rope_dim receives y positions, the second half receives x
    positions. Any non-divisible head tail remains unrotated by the caller.
    """
    h, w = grid_hw
    rope_dim = (head_dim // 4) * 4
    if rope_dim <= 0:
        return torch.empty(chunks * h * w, 0, device=device), 0
    axis_dim = rope_dim // 2
    y = _axis_rope_angles(h, axis_dim, device=device)
    x = _axis_rope_angles(w, axis_dim, device=device)
    yy = y[:, None, :].expand(h, w, axis_dim)
    xx = x[None, :, :].expand(h, w, axis_dim)
    angles = torch.cat([yy, xx], dim=-1).reshape(h * w, rope_dim)
    if chunks > 1:
        angles = angles.repeat(chunks, 1)
    return angles, rope_dim


def apply_2d_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    grid_hw: tuple[int, int],
    start: int,
    length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary embedding to a slice of shaped q/k tensors.

    Args:
        q, k: [B, heads, tokens, head_dim]
        grid_hw: latent patch grid for one image-like token stream
        start: token start offset in the joint sequence
        length: number of image-like tokens; can contain repeated grid chunks
    """
    if length <= 0:
        return q, k
    h, w = grid_hw
    grid_tokens = h * w
    if grid_tokens <= 0 or length % grid_tokens != 0:
        raise ValueError(
            f"RoPE image token length must be a multiple of grid tokens; got length={length}, grid={grid_hw}."
        )
    angles, rope_dim = _build_2d_rope_angles(grid_hw, q.shape[-1], length // grid_tokens, device=q.device)
    if rope_dim <= 0:
        return q, k
    end = start + length
    cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
    sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]

    def _apply(x: torch.Tensor) -> torch.Tensor:
        head = x[:, :, start:end, :rope_dim]
        tail = x[:, :, start:end, rope_dim:]
        rotated = head * cos + _rotate_half(head) * sin
        x_slice = torch.cat([rotated, tail], dim=-1) if tail.shape[-1] else rotated
        return torch.cat([x[:, :, :start], x_slice, x[:, :, end:]], dim=2)

    return _apply(q), _apply(k)
