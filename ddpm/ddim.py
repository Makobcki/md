from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image

from ddpm.data import SimpleTokenizer, TextConfig
from ddpm.diffusion import Diffusion, DiffusionConfig
from ddpm.model import UNet, UNetConfig
from ddpm.utils import EMA, load_ckpt


@torch.no_grad()
def ddim_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: torch.Tensor,
    txt_mask: torch.Tensor,
    steps: int = 50,
    eta: float = 0.0,
    cfg_scale: float = 6.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    clamp_x0: bool = True,
) -> torch.Tensor:
    device = diffusion.device
    b, c, h, w = shape
    x = torch.randn(shape, device=device)

    T = diffusion.cfg.timesteps
    ts = torch.linspace(T - 1, 0, steps, device=device).long()

    for i in range(len(ts)):
        t = ts[i].repeat(b)

        v_cond = model(x, t, txt_ids, txt_mask)
        if uncond_ids is not None and uncond_mask is not None and cfg_scale != 1.0:
            v_un = model(x, t, uncond_ids, uncond_mask)
            v = v_un + cfg_scale * (v_cond - v_un)
        else:
            v = v_cond

        x0 = diffusion.v_to_x0(x, t, v)
        eps = diffusion.v_to_eps(x, t, v)
        if clamp_x0:
            x0 = x0.clamp(-1.0, 1.0)

        if i == len(ts) - 1:
            x = x0
            break

        t_prev = ts[i + 1].repeat(b)
        a_t = diffusion.alpha_bar[t].view(-1, 1, 1, 1)
        a_prev = diffusion.alpha_bar[t_prev].view(-1, 1, 1, 1)

        sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev))
        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0)) * eps
        x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise

    return x


@torch.no_grad()
def heun_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: torch.Tensor,
    txt_mask: torch.Tensor,
    steps: int = 30,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = diffusion.device
    b, _, _, _ = shape
    x = torch.randn(shape, device=device)

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 0, steps, device=device).long()

    for i in range(len(t_schedule) - 1):
        t = t_schedule[i].repeat(b)
        t_next = t_schedule[i + 1].repeat(b)
        sigma = diffusion.sigma_from_t(t).view(-1, 1, 1, 1)
        sigma_next = diffusion.sigma_from_t(t_next).view(-1, 1, 1, 1)

        v_cond = model(x, t, txt_ids, txt_mask)
        if uncond_ids is not None and uncond_mask is not None and cfg_scale != 1.0:
            v_un = model(x, t, uncond_ids, uncond_mask)
            v = v_un + cfg_scale * (v_cond - v_un)
        else:
            v = v_cond

        x0 = diffusion.v_to_x0(x, t, v)
        d = (x - x0) / sigma
        x_euler = x + d * (sigma_next - sigma)

        v_cond_next = model(x_euler, t_next, txt_ids, txt_mask)
        if uncond_ids is not None and uncond_mask is not None and cfg_scale != 1.0:
            v_un_next = model(x_euler, t_next, uncond_ids, uncond_mask)
            v_next = v_un_next + cfg_scale * (v_cond_next - v_un_next)
        else:
            v_next = v_cond_next

        x0_next = diffusion.v_to_x0(x_euler, t_next, v_next)
        d_next = (x_euler - x0_next) / sigma_next
        x = x + 0.5 * (d + d_next) * (sigma_next - sigma)

    t_final = t_schedule[-1].repeat(b)
    v_final = model(x, t_final, txt_ids, txt_mask)
    if uncond_ids is not None and uncond_mask is not None and cfg_scale != 1.0:
        v_un_final = model(x, t_final, uncond_ids, uncond_mask)
        v_final = v_un_final + cfg_scale * (v_final - v_un_final)
    x0_final = diffusion.v_to_x0(x, t_final, v_final)
    return x0_final


def _to_pil(x: torch.Tensor) -> Image.Image:
    x = (x.clamp(-1, 1) + 1.0) * 0.5
    x = (x * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--prompt", type=str, default="1girl")
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    ck = load_ckpt(args.ckpt, device)
    cfg = ck.get("cfg", {})

    vocab = ck.get("tokenizer_vocab")
    if vocab is None:
        vocab_path = cfg.get("vocab_path")
        if not vocab_path:
            raise RuntimeError("Checkpoint must contain tokenizer_vocab or vocab_path.")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

    text_cfg = TextConfig(
        vocab_size=len(vocab),
        max_len=int(cfg.get("text_max_len", 64)),
        lowercase=True,
        strip_punct=True,
    )
    tok = SimpleTokenizer(vocab=vocab, text_cfg=text_cfg)

    unet_cfg = UNetConfig(
        image_channels=3,
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 3, 4])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [32, 16])),
        attn_heads=int(cfg.get("attn_heads", 4)),
        attn_head_dim=int(cfg.get("attn_head_dim", 32)),
        vocab_size=len(vocab),
        text_dim=int(cfg.get("text_dim", 256)),
        text_layers=int(cfg.get("text_layers", 4)),
        text_heads=int(cfg.get("text_heads", 4)),
        text_max_len=int(cfg.get("text_max_len", 64)),
        use_scale_shift_norm=bool(cfg.get("use_scale_shift_norm", False)),
    )

    model = UNet(unet_cfg).to(device)
    model.load_state_dict(ck["model"], strict=True)

    if ck.get("ema") is not None:
        ema = EMA(model)
        ema.shadow = ck["ema"]
        ema.copy_to(model)

    model.eval()

    diff = Diffusion(
        DiffusionConfig(
            timesteps=int(cfg.get("timesteps", 1000)),
            beta_start=float(cfg.get("beta_start", 1e-4)),
            beta_end=float(cfg.get("beta_end", 2e-2)),
            prediction_type=str(cfg.get("prediction_type", "v")),
        ),
        device=device,
    )

    ids, mask = tok.encode(args.prompt)
    ids = ids.unsqueeze(0).repeat(args.n, 1).to(device)
    mask = mask.unsqueeze(0).repeat(args.n, 1).to(device)

    un_ids, un_mask = tok.encode("")
    un_ids = un_ids.unsqueeze(0).repeat(args.n, 1).to(device)
    un_mask = un_mask.unsqueeze(0).repeat(args.n, 1).to(device)

    x = ddim_sample(
        model=model,
        diffusion=diff,
        shape=(args.n, 3, 512, 512),
        txt_ids=ids,
        txt_mask=mask,
        steps=args.steps,
        eta=args.eta,
        cfg_scale=args.cfg,
        uncond_ids=un_ids,
        uncond_mask=un_mask,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.n == 1:
        _to_pil(x[0]).save(out_path)
    else:
        cols = min(4, args.n)
        rows = (args.n + cols - 1) // cols
        grid = Image.new("RGB", (512 * cols, 512 * rows))
        for i in range(args.n):
            r = i // cols
            c = i % cols
            grid.paste(_to_pil(x[i]), (c * 512, r * 512))
        grid.save(out_path)


if __name__ == "__main__":
    main()
