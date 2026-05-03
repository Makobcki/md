from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from diffusion.core.diffusion import Diffusion
from diffusion.domains.domain import Batch
from diffusion.domains.latent import LatentDomain
from diffusion.domains.pixel import PixelDomain
from diffusion.utils import EMA
from model.unet.unet import UNet
from text_enc.tokenizer import BPETokenizer

from data_loader import ImageTextDataset, LatentCacheMetadata, collate_with_tokenizer
from train.loop import _assert_finite, _find_bad_grads


def _sanity_overfit(
    *,
    model: UNet,
    tokenizer: Optional[BPETokenizer],
    entries: list[dict],
    diff: Diffusion,
    domain: PixelDomain | LatentDomain,
    latent_mode: bool,
    latent_cache_dir: str | None,
    latent_cache_sharded: bool,
    latent_cache_index_path: Optional[str],
    latent_dtype: torch.dtype | None,
    latent_cache_strict: bool,
    latent_cache_fallback: bool,
    latent_expected_meta: LatentCacheMetadata | None,
    latent_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]],
    use_text_conditioning: bool,
    self_conditioning: bool,
    self_cond_prob: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
    steps: int,
    max_images: int,
    max_loss: float,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema_switch_step: int,
    ema_decay_fast: float,
    ema_decay_slow: float,
    ema: EMA,
    log_fn: Optional[Callable[[str], None]] = None,
) -> None:
    if steps <= 0 or max_images <= 0:
        return

    if not entries:
        if log_fn is not None:
            log_fn("[SANITY] skip overfit: no training entries found")
        return

    max_images = min(max_images, len(entries))
    sanity_entries = entries[:max_images]
    sanity_ds = ImageTextDataset(
        entries=sanity_entries,
        tokenizer=tokenizer if use_text_conditioning else None,
        cond_drop_prob=0.0 if use_text_conditioning else 1.0,
        seed=0,
        latent_cache_dir=latent_cache_dir,
        latent_cache_sharded=latent_cache_sharded,
        latent_cache_index_path=latent_cache_index_path,
        latent_dtype=latent_dtype,
        return_latents=bool(latent_mode),
        latent_cache_strict=latent_cache_strict,
        latent_cache_fallback=latent_cache_fallback,
        latent_expected_meta=latent_expected_meta,
        include_is_latent=bool(latent_cache_fallback),
    )
    batch = [sanity_ds[i] for i in range(max_images)]
    x0, txt_ids, txt_mask = collate_with_tokenizer(batch, latent_encoder=latent_encoder)
    prepared = domain.prepare_batch(Batch(x=x0, txt_ids=txt_ids, txt_mask=txt_mask, domain=domain.name))
    x0 = prepared.x
    txt_ids = prepared.txt_ids
    txt_mask = prepared.txt_mask

    backup = {
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": {k: v.detach().clone() for k, v in ema.shadow.items()},
    }

    model.train()
    last_loss = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        t = torch.randint(0, diff.cfg.timesteps, (x0.shape[0],), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)

        alpha_bar_t = diff.alpha_bar[t]
        _assert_finite("alpha_bar[t]", alpha_bar_t)

        xt = domain.q_sample(x0, t, noise)
        v_tgt = domain.v_target(x0, t, noise)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            self_cond = None
            if self_conditioning and self_cond_prob > 0:
                if torch.rand((), device=xt.device).item() < self_cond_prob:
                    with torch.no_grad():
                        v_sc = model(xt, t, txt_ids, txt_mask, None)
                        self_cond = diff.v_to_x0(xt, t, v_sc).detach()
            v_pred = model(xt, t, txt_ids, txt_mask, self_cond)
            if v_pred.shape != v_tgt.shape:
                raise RuntimeError("v_pred/v_target shape mismatch in sanity overfit")
            loss = F.mse_loss(v_pred, v_tgt.to(dtype=v_pred.dtype))

        scaler.scale(loss).backward()
        bad_grads = _find_bad_grads(model)
        if bad_grads:
            raise RuntimeError(f"Sanity overfit grads contain NaN/Inf: {bad_grads[:5]}")
        scaler.step(opt)
        scaler.update()
        if ema_switch_step > 0:
            ema.decay = ema_decay_fast if step < ema_switch_step else ema_decay_slow
        ema.update(model)
        last_loss = float(loss.detach().cpu())

        if step % max(steps // 5, 1) == 0 and log_fn is not None:
            log_fn(f"[SANITY] overfit step {step}/{steps} loss={last_loss:.6f}")
        if last_loss <= max_loss:
            break

    if last_loss is None or last_loss > max_loss:
        raise RuntimeError(f"Sanity overfit loss did not reach target: {last_loss} > {max_loss}")

    model.load_state_dict(backup["model"], strict=True)
    opt.load_state_dict(backup["opt"])
    scaler.load_state_dict(backup["scaler"])
    ema.shadow = backup["ema"]
    if log_fn is not None:
        log_fn(f"[SANITY] overfit OK (loss={last_loss:.6f})")
