from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image

from diffusion.text import BPETokenizer, TextConfig
from diffusion.diffusion import Diffusion, DiffusionConfig
from diffusion.model import UNet, UNetConfig
from diffusion.utils import EMA, load_ckpt


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


@torch.no_grad()
def ddim_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 50,
    eta: float = 0.0,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
    clamp_x0: bool = True,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = diffusion.device
    b, c, h, w = shape
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)

    T = diffusion.cfg.timesteps
    ts = torch.linspace(T - 1, 0, steps + 1, device=device).long()
    total_steps = max(len(ts) - 1, 0)

    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = ts[i].repeat(b)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        x0 = diffusion.v_to_x0(x, t, v)
        eps = diffusion.v_to_eps(x, t, v)
        if clamp_x0:
            x0 = x0.clamp(-1.0, 1.0)
        if self_conditioning:
            self_cond = x0.detach()

        if i == total_steps - 1:
            x = x0
            if progress_cb:
                progress_cb(i + 1, total_steps)
            break

        t_prev = ts[i + 1].repeat(b)
        a_t = diffusion.alpha_bar[t].view(-1, 1, 1, 1)
        a_prev = diffusion.alpha_bar[t_prev].view(-1, 1, 1, 1)

        sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev))
        noise = torch.randn_like(x, generator=generator) if eta > 0 else torch.zeros_like(x)

        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0)) * eps
        x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise

        if progress_cb:
            progress_cb(i + 1, total_steps)
    return x


@torch.no_grad()
def heun_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 30,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = diffusion.device
    b, _, _, _ = shape
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 0, steps + 1, device=device).long()

    total_steps = max(len(t_schedule) - 1, 0)
    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = t_schedule[i].repeat(b)
        t_next = t_schedule[i + 1].repeat(b)
        sigma = diffusion.sigma_from_t(t).view(-1, 1, 1, 1)
        sigma_next = diffusion.sigma_from_t(t_next).view(-1, 1, 1, 1)

        sqrt_a_t = diffusion.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_next = diffusion.sqrt_alpha_bar[t_next].view(-1, 1, 1, 1).to(x.dtype)
        sigma = diffusion.sigma_from_t(t).view(-1, 1, 1, 1).to(x.dtype)
        sigma_next = diffusion.sigma_from_t(t_next).view(-1, 1, 1, 1).to(x.dtype)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        eps = diffusion.v_to_eps(x, t, v)
        x_hat = x / sqrt_a_t
        x_hat_euler = x_hat + (sigma_next - sigma) * eps
        x_euler = x_hat_euler * sqrt_a_next
        if self_conditioning:
            self_cond = diffusion.v_to_x0(x, t, v).detach()

        v_next = _guided_v(
            model,
            x_euler,
            t_next,
            txt_ids,
            txt_mask,
            self_cond,
            cfg_scale,
            uncond_ids,
            uncond_mask,
        )
        eps_next = diffusion.v_to_eps(x_euler, t_next, v_next)
        x_hat_next = x_hat + 0.5 * (eps + eps_next) * (sigma_next - sigma)
        x = x_hat_next * sqrt_a_next

        if progress_cb:
            progress_cb(i + 1, total_steps)

    t_final = t_schedule[-1].repeat(b)
    v_final = _guided_v(model, x, t_final, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
    return diffusion.v_to_x0(x, t_final, v_final)


@torch.no_grad()
def dpm_solver_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 30,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = diffusion.device
    b, _, _, _ = shape
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 0, steps + 1, device=device).long()

    total_steps = max(len(t_schedule) - 1, 0)
    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = t_schedule[i].repeat(b)
        t_next = t_schedule[i + 1].repeat(b)
        t_mid = ((t + t_next) // 2).clamp(min=0)

        sqrt_a_t = diffusion.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_next = diffusion.sqrt_alpha_bar[t_next].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_mid = diffusion.sqrt_alpha_bar[t_mid].view(-1, 1, 1, 1).to(x.dtype)

        sigma = diffusion.sigma_from_t(t).view(-1, 1, 1, 1).to(x.dtype)
        sigma_next = diffusion.sigma_from_t(t_next).view(-1, 1, 1, 1).to(x.dtype)
        sigma_mid = diffusion.sigma_from_t(t_mid).view(-1, 1, 1, 1).to(x.dtype)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        eps = diffusion.v_to_eps(x, t, v)
        x_hat = x / sqrt_a_t
        x_hat_mid = x_hat + (sigma_mid - sigma) * eps
        x_mid = x_hat_mid * sqrt_a_mid
        if self_conditioning:
            self_cond = diffusion.v_to_x0(x, t, v).detach()

        v_mid = _guided_v(model, x_mid, t_mid, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        eps_mid = diffusion.v_to_eps(x_mid, t_mid, v_mid)
        x_hat_next = x_hat + (sigma_next - sigma) * eps_mid
        x = x_hat_next * sqrt_a_next

        if progress_cb:
            progress_cb(i + 1, total_steps)

    t_final = t_schedule[-1].repeat(b)
    v_final = _guided_v(model, x, t_final, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
    return diffusion.v_to_x0(x, t_final, v_final)


@torch.no_grad()
def euler_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 30,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = diffusion.device
    b, _, _, _ = shape
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 0, steps + 1, device=device).long()
    total_steps = max(len(t_schedule) - 1, 0)

    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = t_schedule[i].repeat(b)
        t_next = t_schedule[i + 1].repeat(b)
        sqrt_a_t = diffusion.sqrt_alpha_bar[t].view(-1, 1, 1, 1).to(x.dtype)
        sqrt_a_next = diffusion.sqrt_alpha_bar[t_next].view(-1, 1, 1, 1).to(x.dtype)
        sigma = diffusion.sigma_from_t(t).view(-1, 1, 1, 1).to(x.dtype)
        sigma_next = diffusion.sigma_from_t(t_next).view(-1, 1, 1, 1).to(x.dtype)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        eps = diffusion.v_to_eps(x, t, v)
        if self_conditioning:
            self_cond = diffusion.v_to_x0(x, t, v).detach()
        x_hat = x / sqrt_a_t
        x_hat_next = x_hat + (sigma_next - sigma) * eps
        x = x_hat_next * sqrt_a_next

        if progress_cb:
            progress_cb(i + 1, total_steps)

    t_final = t_schedule[-1].repeat(b)
    v_final = _guided_v(model, x, t_final, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
    return diffusion.v_to_x0(x, t_final, v_final)


@torch.no_grad()
def ddpm_ancestral_sample(
    model: UNet,
    diffusion: Diffusion,
    shape: Tuple[int, int, int, int],
    txt_ids: Optional[torch.Tensor],
    txt_mask: Optional[torch.Tensor],
    steps: int = 50,
    cfg_scale: float = 5.0,
    uncond_ids: Optional[torch.Tensor] = None,
    uncond_mask: Optional[torch.Tensor] = None,
    self_conditioning: bool = False,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    device = diffusion.device
    if noise is None:
        x = torch.randn(shape, device=device, generator=generator)
    else:
        x = noise.to(device)
    b = x.shape[0]

    t_schedule = torch.linspace(diffusion.cfg.timesteps - 1, 0, steps + 1, device=device).long()

    total_steps = max(len(t_schedule) - 1, 0)
    self_cond: Optional[torch.Tensor] = None
    for i in range(total_steps):
        t = t_schedule[i].repeat(b)

        v = _guided_v(model, x, t, txt_ids, txt_mask, self_cond, cfg_scale, uncond_ids, uncond_mask)
        x0 = diffusion.v_to_x0(x, t, v)
        eps = diffusion.v_to_eps(x, t, v)
        if self_conditioning:
            self_cond = x0.detach()

        if i == total_steps - 1:
            x = x0
            if progress_cb:
                progress_cb(i + 1, total_steps)
            break

        t_prev = t_schedule[i + 1].repeat(b)
        a_t = diffusion.alpha_bar[t].view(-1, 1, 1, 1)
        a_prev = diffusion.alpha_bar[t_prev].view(-1, 1, 1, 1)

        sigma = torch.sqrt(torch.clamp((1.0 - a_prev) / (1.0 - a_t) * (1.0 - a_t / a_prev), min=0.0))
        noise = torch.randn_like(x, generator=generator) if sigma.max() > 0 else torch.zeros_like(x)
        dir_xt = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0)) * eps
        x = torch.sqrt(a_prev) * x0 + dir_xt + sigma * noise

        if progress_cb:
            progress_cb(i + 1, total_steps)


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
    use_text_conditioning = bool(cfg.get("use_text_conditioning", True))
    meta_flag = ck.get("meta", {}).get("use_text_conditioning")
    if isinstance(meta_flag, bool):
        use_text_conditioning = meta_flag
    mode_label = "ENABLED" if use_text_conditioning else "DISABLED (unconditional diffusion)"
    print(f"[INFO] Text conditioning: {mode_label}")

    tok = None
    if use_text_conditioning:
        vocab_path = cfg.get("text_vocab_path")
        merges_path = cfg.get("text_merges_path")
        if not vocab_path or not merges_path:
            raise RuntimeError("Checkpoint missing text_vocab_path/text_merges_path.")

        text_cfg = TextConfig(
            vocab_path=str(vocab_path),
            merges_path=str(merges_path),
            max_len=int(cfg.get("text_max_len", 64)),
            lowercase=True,
            strip_punct=True,
        )
        tok = BPETokenizer.from_files(vocab_path, merges_path, text_cfg)

    self_conditioning = bool(cfg.get("self_conditioning", False))
    meta_self = ck.get("meta", {}).get("self_conditioning")
    if isinstance(meta_self, bool):
        self_conditioning = meta_self

    unet_cfg = UNetConfig(
        image_channels=3,
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 3, 4])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        attn_resolutions=tuple(cfg.get("attn_resolutions", [32, 16])),
        attn_heads=int(cfg.get("attn_heads", 4)),
        attn_head_dim=int(cfg.get("attn_head_dim", 32)),
        vocab_size=len(tok.vocab) if tok is not None else 0,
        text_dim=int(cfg.get("text_dim", 256)),
        text_layers=int(cfg.get("text_layers", 4)),
        text_heads=int(cfg.get("text_heads", 4)),
        text_max_len=int(cfg.get("text_max_len", 64)),
        use_text_conditioning=use_text_conditioning,
        use_scale_shift_norm=bool(cfg.get("use_scale_shift_norm", False)),
        self_conditioning=self_conditioning,
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
            noise_schedule=str(cfg.get("noise_schedule", "linear")),
            cosine_s=float(cfg.get("cosine_s", 0.008)),
        ),
        device=device,
    )

    if use_text_conditioning:
        if tok is None:
            raise RuntimeError("Tokenizer is not initialized.")
        ids, mask = tok.encode(args.prompt)
        ids = ids.unsqueeze(0).repeat(args.n, 1).to(device)
        mask = mask.unsqueeze(0).repeat(args.n, 1).to(device)

        un_ids, un_mask = tok.encode("")
        un_ids = un_ids.unsqueeze(0).repeat(args.n, 1).to(device)
        un_mask = un_mask.unsqueeze(0).repeat(args.n, 1).to(device)
    else:
        ids = None
        mask = None
        un_ids = None
        un_mask = None
        if args.prompt.strip():
            print("[WARN] use_text_conditioning=false, prompt ignored")

    x = ddim_sample(
        model=model,
        diffusion=diff,
        shape=(args.n, 3, 512, 512),
        txt_ids=ids,
        txt_mask=mask,
        steps=args.steps,
        eta=args.eta,
        cfg_scale=float(args.cfg),
        uncond_ids=un_ids,
        uncond_mask=un_mask,
        self_conditioning=self_conditioning,
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
