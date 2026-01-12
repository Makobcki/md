from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from diffusion.text import BPETokenizer, TextConfig
from diffusion.diffusion import Diffusion, DiffusionConfig
from diffusion.model import UNet, UNetConfig
from diffusion.utils import EMA, load_ckpt, save_ckpt, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--out", default="./runs/sanity/ckpt_sanity.pt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed, deterministic=True)

    vocab_path = "diffusion/bpe/vocab.json"
    merges_path = "diffusion/bpe/merges.txt"
    text_cfg = TextConfig(vocab_path=vocab_path, merges_path=merges_path, max_len=8, lowercase=True, strip_punct=True)
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, text_cfg)

    unet_cfg = UNetConfig(
        image_channels=3,
        base_channels=32,
        channel_mults=(1, 2, 3),
        num_res_blocks=1,
        dropout=0.0,
        attn_resolutions=(16,),
        attn_heads=2,
        attn_head_dim=16,
        vocab_size=len(tokenizer.vocab),
        text_dim=64,
        text_layers=2,
        text_heads=2,
        text_max_len=text_cfg.max_len,
        use_scale_shift_norm=True,
    )
    model = UNet(unet_cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, decay=0.9)

    diffusion = Diffusion(DiffusionConfig(timesteps=100, prediction_type="v"), device=device)
    ids, mask = tokenizer.encode("test")
    ids = ids.unsqueeze(0).repeat(args.batch_size, 1).to(device)
    mask = mask.unsqueeze(0).repeat(args.batch_size, 1).to(device)

    model.train()
    for step in range(args.steps):
        x0 = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device) * 2.0 - 1.0
        if not torch.isfinite(x0).all() or x0.min() < -1.0 or x0.max() > 1.0:
            raise RuntimeError("x0 is out of range [-1, 1] in sanity check")
        t = torch.randint(0, diffusion.cfg.timesteps, (args.batch_size,), device=device)
        noise = torch.randn_like(x0)
        xt = diffusion.q_sample(x0, t, noise)
        alpha_bar_t = diffusion.alpha_bar[t]
        if not torch.isfinite(alpha_bar_t).all():
            raise RuntimeError("alpha_bar[t] has NaN/Inf in sanity check")
        v_target = diffusion.v_target(x0, t, noise)
        eps = diffusion.v_to_eps(xt, t, v_target)
        x0_recon = diffusion.v_to_x0(xt, t, v_target)
        if not torch.allclose(eps, noise, atol=1e-4, rtol=1e-4):
            raise RuntimeError("v_to_eps mismatch in sanity check")
        if not torch.allclose(x0_recon, x0, atol=1e-4, rtol=1e-4):
            raise RuntimeError("v_to_x0 mismatch in sanity check")

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=device.type == "cuda"):
            pred = model(xt, t, ids, mask)
            if pred.shape != v_target.shape or pred.dtype != v_target.dtype:
                raise RuntimeError("pred/v_target mismatch in sanity check")
            loss = F.mse_loss(pred, v_target)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        ema.update(model)

    out_path = Path(args.out)
    save_ckpt(str(out_path), {
        "step": args.steps,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "cfg": {
            "image_size": args.image_size,
            "timesteps": diffusion.cfg.timesteps,
            "text_max_len": text_cfg.max_len,
            "prediction_type": "v",
            "text_vocab_path": vocab_path,
            "text_merges_path": merges_path,
        },
    })

    ck = load_ckpt(str(out_path), device)
    assert "model" in ck and "optimizer" in ck and "scaler" in ck and "ema" in ck
    print(f"[OK] sanity check saved and loaded {out_path}")


if __name__ == "__main__":
    main()
