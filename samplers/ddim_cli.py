from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from diffusion.diffusion import Diffusion, DiffusionConfig
from diffusion.model import UNet, UNetConfig
from diffusion.text import BPETokenizer, TextConfig
from diffusion.utils import EMA, load_ckpt

from samplers.ddim import ddim_sample


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

    tok: Optional[BPETokenizer] = None
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
