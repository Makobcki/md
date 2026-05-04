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


def _axis_rope_scale(length: int, base_length: int | None, scaling: str) -> float:
    if base_length is None or base_length <= 0 or scaling == "none":
        return 1.0
    return max(float(length) / float(base_length), 1.0)


def _ntk_theta(theta: float, scale: float, dim: int) -> float:
    if scale <= 1.0 or dim <= 2:
        return float(theta)
    # Dynamic-NTK style base expansion for extrapolating longer 2D grids while
    # preserving the original frequency layout at or below the base resolution.
    exponent = float(dim) / max(float(dim - 2), 1.0)
    return float(theta) * (float(scale) ** exponent)


def _axis_rope_angles(
    length: int,
    dim: int,
    *,
    device: torch.device,
    base_length: int | None = None,
    scaling: str = "none",
    theta: float = 10000.0,
) -> torch.Tensor:
    scale = _axis_rope_scale(length, base_length, scaling)
    idx = torch.arange(length, device=device, dtype=torch.float32)
    if scaling == "linear" and scale > 1.0:
        idx = idx / scale
    theta_eff = _ntk_theta(theta, scale, dim) if scaling == "ntk" else float(theta)
    inv_freq = 1.0 / (theta_eff ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / max(dim, 1)))
    freqs = torch.einsum("n,d->nd", idx, inv_freq)
    return torch.repeat_interleave(freqs, 2, dim=-1)


def _build_2d_rope_angles(
    grid_hw: tuple[int, int],
    head_dim: int,
    chunks: int,
    *,
    device: torch.device,
    base_grid_hw: tuple[int, int] | None = None,
    scaling: str = "none",
    theta: float = 10000.0,
) -> tuple[torch.Tensor, int]:
    """Return repeated 2D RoPE angles as [chunks * H * W, rope_dim].

    `base_grid_hw` and `scaling` make the rotary frequencies stable when the
    model is trained at one latent grid and sampled/evaluated at another. The
    default behavior is identical to the original fixed-theta RoPE path.
    """
    h, w = grid_hw
    rope_dim = (head_dim // 4) * 4
    if rope_dim <= 0:
        return torch.empty(chunks * h * w, 0, device=device), 0
    axis_dim = rope_dim // 2
    base_h = base_grid_hw[0] if base_grid_hw is not None else None
    base_w = base_grid_hw[1] if base_grid_hw is not None else None
    y = _axis_rope_angles(h, axis_dim, device=device, base_length=base_h, scaling=scaling, theta=theta)
    x = _axis_rope_angles(w, axis_dim, device=device, base_length=base_w, scaling=scaling, theta=theta)
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
    base_grid_hw: tuple[int, int] | None = None,
    scaling: str = "none",
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary embedding to a slice of shaped q/k tensors.

    Args:
        q, k: [B, heads, tokens, head_dim]
        grid_hw: latent patch grid for one image-like token stream
        start: token start offset in the joint sequence
        length: number of image-like tokens; can contain repeated grid chunks
        base_grid_hw: training/base grid used for RoPE extrapolation
        scaling: one of none/linear/ntk
        theta: rotary base frequency
    """
    if length <= 0:
        return q, k
    if scaling not in {"none", "linear", "ntk"}:
        raise ValueError("rope scaling must be one of: none, linear, ntk.")
    h, w = grid_hw
    grid_tokens = h * w
    if grid_tokens <= 0 or length % grid_tokens != 0:
        raise ValueError(
            f"RoPE image token length must be a multiple of grid tokens; got length={length}, grid={grid_hw}."
        )
    angles, rope_dim = _build_2d_rope_angles(
        grid_hw,
        q.shape[-1],
        length // grid_tokens,
        device=q.device,
        base_grid_hw=base_grid_hw,
        scaling=scaling,
        theta=theta,
    )
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


def apply_2d_rope_sections(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    sections: tuple[tuple[int, int, int, int], ...] | tuple[tuple[int, int, int, int, int, int], ...],
    base_grid_hw: tuple[int, int] | None = None,
    scaling: str = "none",
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE to multiple image-like token sections.

    Each section is either ``(start, length, grid_h, grid_w)`` or
    ``(start, length, grid_h, grid_w, base_h, base_w)``. This is used by
    Stage D where target/source/mask/control/coarse streams can have different
    patch sizes, therefore one global image grid is no longer sufficient.
    """
    if scaling not in {"none", "linear", "ntk"}:
        raise ValueError("rope scaling must be one of: none, linear, ntk.")
    for section in sections:
        if len(section) == 4:
            start, length, grid_h, grid_w = section
            section_base = base_grid_hw
        elif len(section) == 6:
            start, length, grid_h, grid_w, base_h, base_w = section
            section_base = (int(base_h), int(base_w))
        else:
            raise ValueError("RoPE section must have 4 or 6 integer values.")
        if int(length) <= 0:
            continue
        q, k = apply_2d_rope(
            q,
            k,
            grid_hw=(int(grid_h), int(grid_w)),
            start=int(start),
            length=int(length),
            base_grid_hw=section_base,
            scaling=scaling,
            theta=theta,
        )
    return q, k
