from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

CONTROL_TYPE_TO_ID: Final[dict[str, int]] = {
    "none": 0,
    "latent_identity": 1,
    "image": 2,
    "canny": 3,
    "depth": 4,
    "pose": 5,
    "lineart": 6,
    "normal": 7,
}
SUPPORTED_CONTROL_TYPES: Final[tuple[str, ...]] = tuple(CONTROL_TYPE_TO_ID)


def control_type_id(name: str) -> int:
    try:
        return CONTROL_TYPE_TO_ID[str(name)]
    except KeyError as exc:
        allowed = ", ".join(SUPPORTED_CONTROL_TYPES)
        raise ValueError(f"Unsupported control_type {name!r}; allowed: {allowed}.") from exc


def _as_bchw(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.dim() == 3:
        return x.unsqueeze(0), True
    if x.dim() == 4:
        return x, False
    raise ValueError("control preprocessing expects [C,H,W] or [B,C,H,W].")


def _restore_rank(x: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return x.squeeze(0) if squeezed else x


def _repeat_to_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.shape[1] == channels:
        return x
    if x.shape[1] > channels:
        return x[:, :channels]
    reps = (channels + x.shape[1] - 1) // x.shape[1]
    return x.repeat(1, reps, 1, 1)[:, :channels]


def _normalize01(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    flat = x.flatten(1)
    lo = flat.amin(dim=1).view(-1, 1, 1, 1)
    hi = flat.amax(dim=1).view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)


def _gray(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x
    if x.shape[1] >= 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x[:, :3] * weights).sum(dim=1, keepdim=True)
    return x.mean(dim=1, keepdim=True)


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    g = _gray(x).float()
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    dx = F.conv2d(g, kx, padding=1)
    dy = F.conv2d(g, ky, padding=1)
    return torch.sqrt(dx.square() + dy.square() + 1.0e-12).to(dtype=x.dtype)


def _edge_proxy(x: torch.Tensor, *, binary: bool) -> torch.Tensor:
    mag = _normalize01(_sobel_magnitude(x))
    if binary:
        flat = mag.flatten(1)
        threshold = (flat.mean(dim=1) + 0.5 * flat.std(dim=1, unbiased=False)).view(-1, 1, 1, 1)
        mag = (mag > threshold).to(dtype=x.dtype)
    return _repeat_to_channels(mag, x.shape[1])


def _depth_proxy(x: torch.Tensor) -> torch.Tensor:
    # Cached latent training does not have a depth estimator. This deterministic
    # low-frequency proxy gives depth/control type a distinct input distribution
    # without depending on optional external models.
    pooled = F.avg_pool2d(_gray(x), kernel_size=5, stride=1, padding=2)
    return _repeat_to_channels(_normalize01(pooled), x.shape[1]).to(dtype=x.dtype)


def _normal_proxy(x: torch.Tensor) -> torch.Tensor:
    g = _gray(x).float()
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    dx = F.conv2d(g, kx, padding=1)
    dy = F.conv2d(g, ky, padding=1)
    mag = torch.sqrt(dx.square() + dy.square() + 1.0e-12)
    normal = torch.cat([_normalize01(dx), _normalize01(dy), _normalize01(mag)], dim=1).to(dtype=x.dtype)
    return _repeat_to_channels(normal, x.shape[1])


def latent_control_preprocess(latents: torch.Tensor, control_type: str) -> torch.Tensor:
    """Preprocess cached latent tensors into a control latent-like tensor.

    The project trains from VAE latent caches, so real image-space preprocessors
    are not available in this path. For `image`/`latent_identity` the source
    latent is used directly. For edge/depth/normal-like controls we build
    deterministic latent-space proxies so control types are no longer aliases of
    the same `latent_identity` stream.
    """
    x, squeezed = _as_bchw(latents)
    name = str(control_type)
    if name == "none":
        out = torch.zeros_like(x)
    elif name in {"latent_identity", "image"}:
        out = x.clone()
    elif name == "canny":
        out = _edge_proxy(x, binary=True)
    elif name == "lineart":
        out = _edge_proxy(x, binary=False)
    elif name == "depth":
        out = _depth_proxy(x)
    elif name == "normal":
        out = _normal_proxy(x)
    elif name == "pose":
        out = _edge_proxy(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1), binary=True)
    else:
        control_type_id(name)
        raise AssertionError("unreachable")
    return _restore_rank(out.to(device=latents.device, dtype=latents.dtype), squeezed)


def image_control_preprocess(images: torch.Tensor, control_type: str) -> torch.Tensor:
    """Preprocess image tensors in [-1, 1] before VAE encoding at sampling time."""
    x, squeezed = _as_bchw(images)
    name = str(control_type)
    if name == "none":
        out = torch.zeros_like(x)
    elif name in {"image", "latent_identity"}:
        out = x.clone()
    elif name == "canny":
        edges = _edge_proxy(x, binary=True)
        out = edges * 2.0 - 1.0
    elif name == "lineart":
        lines = _edge_proxy(x, binary=False)
        out = lines * 2.0 - 1.0
    elif name == "depth":
        depth = _depth_proxy(x)
        out = depth * 2.0 - 1.0
    elif name == "normal":
        normal = _normal_proxy(x)
        out = normal * 2.0 - 1.0
    elif name == "pose":
        pose = _edge_proxy(F.avg_pool2d(x, kernel_size=3, stride=1, padding=1), binary=True)
        out = pose * 2.0 - 1.0
    else:
        control_type_id(name)
        raise AssertionError("unreachable")
    return _restore_rank(out.to(device=images.device, dtype=images.dtype), squeezed)
