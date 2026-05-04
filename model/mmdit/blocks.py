from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .attention import JointAttention
from .norms import AdaLNZero, build_norm, modulate


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float, swiglu: bool) -> None:
        super().__init__()
        inner = int(hidden_dim * mlp_ratio)
        self.swiglu = bool(swiglu)
        if self.swiglu:
            self.fc1 = nn.Linear(hidden_dim, inner * 2)
            self.act = nn.SiLU()
        else:
            self.fc1 = nn.Linear(hidden_dim, inner)
            self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.swiglu:
            x, gate = x.chunk(2, dim=-1)
            x = x * self.act(gate)
        else:
            x = self.act(x)
        return self.fc2(self.drop(x))


def _offset_rope_sections(
    sections: tuple[tuple[int, int, int, int], ...] | tuple[tuple[int, int, int, int, int, int], ...] | None,
    offset: int,
) -> tuple[tuple[int, ...], ...] | None:
    """Move image-local RoPE sections into a joint [text, image] sequence.

    Stage D builds per-stream sections relative to the image-token sequence:
    [source/mask/control/coarse/target]. Joint blocks concatenate text before
    those tokens, so sections must be shifted by text_len before applying RoPE.
    Image-only blocks must keep the original image-local sections.
    """
    if not sections:
        return None
    delta = int(offset)
    if delta == 0:
        return tuple(tuple(int(v) for v in section) for section in sections)
    shifted: list[tuple[int, ...]] = []
    for section in sections:
        if len(section) not in {4, 6}:
            raise ValueError("RoPE section must have 4 or 6 integer values.")
        head = int(section[0]) + delta
        shifted.append((head, *tuple(int(v) for v in section[1:])))
    return tuple(shifted)


class MMDiTDoubleBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        qk_norm: bool,
        rms_norm: bool,
        swiglu: bool,
    ) -> None:
        super().__init__()
        self.img_norm1 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.txt_norm1 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.img_norm2 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.txt_norm2 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.img_adaln = AdaLNZero(hidden_dim)
        self.txt_adaln = AdaLNZero(hidden_dim)
        self.img_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.txt_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn = JointAttention(hidden_dim, num_heads, attn_dropout, qk_norm)
        self.img_out = nn.Linear(hidden_dim, hidden_dim)
        self.txt_out = nn.Linear(hidden_dim, hidden_dim)
        self.img_mlp = FeedForward(hidden_dim, mlp_ratio, dropout, swiglu)
        self.txt_mlp = FeedForward(hidden_dim, mlp_ratio, dropout, swiglu)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        cond: torch.Tensor,
        txt_mask: Optional[torch.Tensor],
        rope_grid_hw: Optional[tuple[int, int]] = None,
        use_rope: bool = False,
        rope_base_grid_hw: Optional[tuple[int, int]] = None,
        rope_scaling: str = "none",
        rope_theta: float = 10000.0,
        rope_sections: tuple[tuple[int, int, int, int], ...] | tuple[tuple[int, int, int, int, int, int], ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_shift, img_scale, img_gate_attn, img_mlp_shift, img_mlp_scale, img_gate_mlp = self.img_adaln(cond)
        txt_shift, txt_scale, txt_gate_attn, txt_mlp_shift, txt_mlp_scale, txt_gate_mlp = self.txt_adaln(cond)

        img_norm = modulate(self.img_norm1(img), img_shift, img_scale)
        txt_norm = modulate(self.txt_norm1(txt), txt_shift, txt_scale)
        q_img, k_img, v_img = self.img_qkv(img_norm).chunk(3, dim=-1)
        q_txt, k_txt, v_txt = self.txt_qkv(txt_norm).chunk(3, dim=-1)

        q = torch.cat([q_txt, q_img], dim=1)
        k = torch.cat([k_txt, k_img], dim=1)
        v = torch.cat([v_txt, v_img], dim=1)
        joint_mask = None
        if txt_mask is not None:
            img_mask = torch.ones(img.shape[:2], device=img.device, dtype=torch.bool)
            joint_mask = torch.cat([txt_mask.to(device=img.device, dtype=torch.bool), img_mask], dim=1)
        joint_rope_sections = _offset_rope_sections(rope_sections, txt.shape[1]) if use_rope else None
        out = self.attn(
            q,
            k,
            v,
            joint_mask,
            rope_grid_hw=rope_grid_hw if use_rope and joint_rope_sections is None else None,
            rope_start=txt.shape[1],
            rope_length=img.shape[1],
            rope_base_grid_hw=rope_base_grid_hw,
            rope_scaling=rope_scaling,
            rope_theta=rope_theta,
            rope_sections=joint_rope_sections,
        )
        out_txt, out_img = out[:, : txt.shape[1]], out[:, txt.shape[1] :]

        txt = txt + txt_gate_attn.unsqueeze(1) * self.txt_out(out_txt)
        img = img + img_gate_attn.unsqueeze(1) * self.img_out(out_img)
        txt = txt + txt_gate_mlp.unsqueeze(1) * self.txt_mlp(modulate(self.txt_norm2(txt), txt_mlp_shift, txt_mlp_scale))
        img = img + img_gate_mlp.unsqueeze(1) * self.img_mlp(modulate(self.img_norm2(img), img_mlp_shift, img_mlp_scale))
        return img, txt


class MMDiTSingleBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        qk_norm: bool,
        rms_norm: bool,
        swiglu: bool,
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.norm2 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.adaln = AdaLNZero(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn = JointAttention(hidden_dim, num_heads, attn_dropout, qk_norm)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = FeedForward(hidden_dim, mlp_ratio, dropout, swiglu)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        cond: torch.Tensor,
        txt_mask: Optional[torch.Tensor],
        rope_grid_hw: Optional[tuple[int, int]] = None,
        use_rope: bool = False,
        rope_base_grid_hw: Optional[tuple[int, int]] = None,
        rope_scaling: str = "none",
        rope_theta: float = 10000.0,
        rope_sections: tuple[tuple[int, int, int, int], ...] | tuple[tuple[int, int, int, int, int, int], ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([txt, img], dim=1)
        shift, scale, gate_attn, mlp_shift, mlp_scale, gate_mlp = self.adaln(cond)
        x_norm = modulate(self.norm1(x), shift, scale)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        joint_mask = None
        if txt_mask is not None:
            img_mask = torch.ones(img.shape[:2], device=img.device, dtype=torch.bool)
            joint_mask = torch.cat([txt_mask.to(device=img.device, dtype=torch.bool), img_mask], dim=1)
        joint_rope_sections = _offset_rope_sections(rope_sections, txt.shape[1]) if use_rope else None
        x = x + gate_attn.unsqueeze(1) * self.out(
            self.attn(
                q,
                k,
                v,
                joint_mask,
                rope_grid_hw=rope_grid_hw if use_rope and joint_rope_sections is None else None,
                rope_start=txt.shape[1],
                rope_length=img.shape[1],
                rope_base_grid_hw=rope_base_grid_hw,
                rope_scaling=rope_scaling,
                rope_theta=rope_theta,
                rope_sections=joint_rope_sections,
            )
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), mlp_shift, mlp_scale))
        return x[:, txt.shape[1] :], x[:, : txt.shape[1]]



class ImageOnlyBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
        qk_norm: bool,
        rms_norm: bool,
        swiglu: bool,
    ) -> None:
        super().__init__()
        self.norm1 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.norm2 = build_norm(hidden_dim, rms_norm=rms_norm)
        self.adaln = AdaLNZero(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attn = JointAttention(hidden_dim, num_heads, attn_dropout, qk_norm)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.mlp = FeedForward(hidden_dim, mlp_ratio, dropout, swiglu)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        cond: torch.Tensor,
        txt_mask: Optional[torch.Tensor],
        rope_grid_hw: Optional[tuple[int, int]] = None,
        use_rope: bool = False,
        rope_base_grid_hw: Optional[tuple[int, int]] = None,
        rope_scaling: str = "none",
        rope_theta: float = 10000.0,
        rope_sections: tuple[tuple[int, int, int, int], ...] | tuple[tuple[int, int, int, int, int, int], ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate_attn, mlp_shift, mlp_scale, gate_mlp = self.adaln(cond)
        x_norm = modulate(self.norm1(img), shift, scale)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        img = img + gate_attn.unsqueeze(1) * self.out(
            self.attn(
                q,
                k,
                v,
                None,
                rope_grid_hw=rope_grid_hw if use_rope else None,
                rope_start=0,
                rope_length=img.shape[1],
                rope_base_grid_hw=rope_base_grid_hw,
                rope_scaling=rope_scaling,
                rope_theta=rope_theta,
                rope_sections=rope_sections,
            )
        )
        img = img + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(img), mlp_shift, mlp_scale))
        return img, txt
