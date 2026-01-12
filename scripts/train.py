from __future__ import annotations

import argparse
import os
import signal
import time
from dataclasses import replace
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from diffusion.config import TrainConfig
from diffusion.data import (
    DanbooruConfig,
    DanbooruDataset,
    LatentCacheMetadata,
    build_or_load_index,
    build_token_cache_key,
    collate_with_tokenizer,
)
from diffusion.diffusion import Diffusion, DiffusionConfig
from diffusion.domain import Batch, LatentDomain, PixelDomain
from diffusion.events import EventBus, JsonlFileSink, StdoutJsonSink
from diffusion.model import UNet, UNetConfig
from diffusion.perf import PerfConfig, configure_performance
from diffusion.text import BPETokenizer, TextConfig
from diffusion.utils import (
    EMA,
    build_run_metadata,
    load_ckpt,
    normalize_state_dict_for_keys,
    normalize_state_dict_for_model,
    resolve_resume_path,
    save_ckpt,
    seed_everything,
)
from diffusion.vae import VAEWrapper

def _is_webui_mode() -> bool:
    return os.environ.get("WEBUI") == "1"

def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank() -> int:
    if _dist_is_initialized():
        return torch.distributed.get_rank()
    return 0


def _dist_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if _dist_is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _configure_cudagraphs(enabled: bool) -> None:
    if not hasattr(torch, "_inductor"):
        return
    cfg = getattr(torch._inductor, "config", None)
    if cfg is None:
        return
    triton_cfg = getattr(cfg, "triton", None)
    if triton_cfg is not None and hasattr(triton_cfg, "cudagraphs"):
        triton_cfg.cudagraphs = bool(enabled)
    if hasattr(cfg, "cudagraphs"):
        cfg.cudagraphs = bool(enabled)


def _cudagraph_step_begin() -> None:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
        torch.compiler.cudagraph_mark_step_begin()


def _webui_metrics_path() -> Path | None:
    run_dir = os.environ.get("WEBUI_RUN_DIR")
    if not run_dir:
        return None
    return Path(run_dir) / "metrics" / "train_metrics.jsonl"


def _resolve_num_workers(requested: int) -> int:
    if requested == 0:
        return 0
    cpu_count = os.cpu_count() or 0
    max_workers = max(cpu_count - 1, 1) if cpu_count else 0
    if requested < 0:
        return max_workers
    if max_workers:
        return min(requested, max_workers)
    return requested


def _assert_divisible(value: int, divisor: int, name: str) -> None:
    if value % divisor != 0:
        raise RuntimeError(f"{name} must be divisible by {divisor}, got {value}.")


def _prune_checkpoints(out_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    to_remove = ckpts[:-keep_last]
    for p in to_remove:
        try:
            p.unlink()
        except FileNotFoundError:
            continue


@dataclass
class _TimingBucket:
    count: int = 0
    total_sec: float = 0.0
    min_sec: float = float("inf")
    max_sec: float = 0.0

    def add(self, value_sec: float) -> None:
        self.count += 1
        self.total_sec += value_sec
        self.min_sec = min(self.min_sec, value_sec)
        self.max_sec = max(self.max_sec, value_sec)

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0, "total_sec": 0.0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        avg_ms = (self.total_sec / self.count) * 1000.0
        return {
            "count": self.count,
            "total_sec": self.total_sec,
            "avg_ms": avg_ms,
            "min_ms": self.min_sec * 1000.0,
            "max_ms": self.max_sec * 1000.0,
        }


class _TimingStats:
    def __init__(self, *, use_cuda_events: bool, gpu_sections: Optional[set[str]] = None) -> None:
        self.use_cuda_events = bool(use_cuda_events)
        self.gpu_sections = gpu_sections or set()
        self._cpu: dict[str, _TimingBucket] = {}
        self._gpu_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def section(self, name: str):
        return _SectionTimer(self, name)

    def add_cpu(self, name: str, elapsed_sec: float) -> None:
        bucket = self._cpu.setdefault(name, _TimingBucket())
        bucket.add(elapsed_sec)

    def add_gpu_event(self, name: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self._gpu_events.setdefault(name, []).append((start, end))

    def report(self, *, reset: bool = True) -> dict:
        stats: dict[str, dict] = {}
        for name, bucket in self._cpu.items():
            stats[name] = bucket.summary()

        if self._gpu_events:
            torch.cuda.synchronize()
            for name, pairs in self._gpu_events.items():
                bucket = _TimingBucket()
                for start, end in pairs:
                    elapsed_ms = float(start.elapsed_time(end))
                    bucket.add(elapsed_ms / 1000.0)
                stats[f"{name}_gpu"] = bucket.summary()
        if reset:
            self._cpu = {}
            self._gpu_events = {}
        return stats


class _SectionTimer:
    def __init__(self, stats: _TimingStats, name: str) -> None:
        self.stats = stats
        self.name = name
        self.elapsed_sec: float = 0.0
        self._start_cpu: Optional[float] = None
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None

    def __enter__(self) -> "_SectionTimer":
        self._start_cpu = time.perf_counter()
        if self.stats.use_cuda_events and self.name in self.stats.gpu_sections:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start_cpu is not None:
            self.elapsed_sec = time.perf_counter() - self._start_cpu
            self.stats.add_cpu(self.name, self.elapsed_sec)
        if self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            self.stats.add_gpu_event(self.name, self._start_event, self._end_event)



def get_min_snr_weights(alpha_bar_t: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    eps = 1e-8
    a = alpha_bar_t.clamp(min=eps, max=1.0 - eps)
    snr = a / (1.0 - a + eps)
    g = torch.full_like(snr, float(gamma))
    return torch.minimum(snr, g) / (snr + 1.0)


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} has NaN/Inf values")


def _find_bad_grads(model: torch.nn.Module) -> list[str]:
    bad = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            bad.append(name)
    return bad


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        param_norm = param.grad.detach().data.norm(2)
        total += float(param_norm.item()) ** 2
    return total ** 0.5


def _sanity_overfit(
    *,
    model: UNet,
    tokenizer: BPETokenizer,
    entries: list[dict],
    diff: Diffusion,
    domain: PixelDomain | LatentDomain,
    latent_mode: bool,
    latent_cache_dir: str | None,
    latent_dtype: torch.dtype | None,
    latent_cache_strict: bool,
    latent_cache_fallback: bool,
    latent_expected_meta: LatentCacheMetadata | None,
    latent_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]],
    use_amp: bool,
    amp_dtype: torch.dtype,
    steps: int,
    max_images: int,
    max_loss: float,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
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
    sanity_ds = DanbooruDataset(
        entries=sanity_entries,
        tokenizer=tokenizer,
        cond_drop_prob=0.0,
        seed=0,
        latent_cache_dir=latent_cache_dir,
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
            v_pred = model(xt, t, txt_ids, txt_mask)
            if v_pred.shape != v_tgt.shape:
                raise RuntimeError("v_pred/v_target shape mismatch in sanity overfit")
            loss = F.mse_loss(v_pred, v_tgt.to(dtype=v_pred.dtype))

        scaler.scale(loss).backward()
        bad_grads = _find_bad_grads(model)
        if bad_grads:
            raise RuntimeError(f"Sanity overfit grads contain NaN/Inf: {bad_grads[:5]}")
        scaler.step(opt)
        scaler.update()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/train.yaml")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ckpt-keep-last", type=int, default=None)
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = replace(cfg, seed=int(args.seed))
    if args.resume:
        cfg = replace(cfg, resume_ckpt=str(args.resume))
    if args.ckpt_keep_last is not None:
        cfg = replace(cfg, ckpt_keep_last=int(args.ckpt_keep_last))
    run_cfg = cfg

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )

    out_dir = Path(run_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perf_active = configure_performance(
        PerfConfig(
            tf32=bool(run_cfg.tf32),
            cudnn_benchmark=bool(run_cfg.cudnn_benchmark),
            channels_last=bool(run_cfg.channels_last),
            enable_flash_sdp=bool(run_cfg.enable_flash_sdp),
            enable_mem_efficient_sdp=bool(run_cfg.enable_mem_efficient_sdp),
            enable_math_sdp=bool(run_cfg.enable_math_sdp),
        ),
        device,
    )

    seed_everything(int(run_cfg.seed), deterministic=bool(run_cfg.deterministic))

    resume = str(run_cfg.resume_ckpt).strip()
    resume_path = ""
    ck = None
    model_cfg = run_cfg
    if resume:
        resume_path = resolve_resume_path(resume, out_dir)
        ck = load_ckpt(resume_path, device)
        if "cfg" in ck and isinstance(ck["cfg"], dict):
            model_cfg = TrainConfig.from_dict(ck["cfg"])

    cfg_dict = model_cfg.to_dict()
    cfg_dict.update({
        "data_root": run_cfg.data_root,
        "image_dir": run_cfg.image_dir,
        "meta_dir": run_cfg.meta_dir,
        "tags_dir": run_cfg.tags_dir,
        "caption_field": run_cfg.caption_field,
        "min_tag_count": run_cfg.min_tag_count,
        "require_512": run_cfg.require_512,
        "val_ratio": run_cfg.val_ratio,
        "cache_dir": run_cfg.cache_dir,
        "failed_list": run_cfg.failed_list,
        "seed": run_cfg.seed,
        "out_dir": run_cfg.out_dir,
        "batch_size": run_cfg.batch_size,
        "grad_accum_steps": run_cfg.grad_accum_steps,
        "num_workers": run_cfg.num_workers,
        "prefetch_factor": run_cfg.prefetch_factor,
        "pin_memory": run_cfg.pin_memory,
        "persistent_workers": run_cfg.persistent_workers,
        "lr": run_cfg.lr,
        "weight_decay": run_cfg.weight_decay,
        "max_steps": run_cfg.max_steps,
        "log_every": run_cfg.log_every,
        "save_every": run_cfg.save_every,
        "cond_drop_prob": run_cfg.cond_drop_prob,
        "amp": run_cfg.amp,
        "amp_dtype": run_cfg.amp_dtype,
        "compile": run_cfg.compile,
        "compile_warmup_steps": run_cfg.compile_warmup_steps,
        "compile_cudagraphs": run_cfg.compile_cudagraphs,
        "grad_clip_norm": run_cfg.grad_clip_norm,
        "ema_decay": run_cfg.ema_decay,
        "resume_ckpt": resume_path or run_cfg.resume_ckpt,
        "deterministic": run_cfg.deterministic,
        "sanity_overfit_steps": run_cfg.sanity_overfit_steps,
        "sanity_overfit_images": run_cfg.sanity_overfit_images,
        "sanity_overfit_max_loss": run_cfg.sanity_overfit_max_loss,
        "tf32": run_cfg.tf32,
        "cudnn_benchmark": run_cfg.cudnn_benchmark,
        "channels_last": run_cfg.channels_last,
        "enable_flash_sdp": run_cfg.enable_flash_sdp,
        "enable_mem_efficient_sdp": run_cfg.enable_mem_efficient_sdp,
        "enable_math_sdp": run_cfg.enable_math_sdp,
        "mode": run_cfg.mode,
        "latent_channels": run_cfg.latent_channels,
        "latent_downsample_factor": run_cfg.latent_downsample_factor,
        "latent_cache": run_cfg.latent_cache,
        "latent_cache_dir": run_cfg.latent_cache_dir,
        "latent_dtype": run_cfg.latent_dtype,
        "latent_precompute": run_cfg.latent_precompute,
        "latent_cache_fallback": run_cfg.latent_cache_fallback,
        "latent_cache_strict": run_cfg.latent_cache_strict,
        "vae_pretrained": run_cfg.vae_pretrained,
        "vae_freeze": run_cfg.vae_freeze,
        "vae_scaling_factor": run_cfg.vae_scaling_factor,
        "ckpt_keep_last": run_cfg.ckpt_keep_last,
    })

    # ----------------------------
    # Dataset + vocab
    # ----------------------------
    dcfg = DanbooruConfig(
        root=str(run_cfg.data_root),
        image_dir=str(run_cfg.image_dir),
        meta_dir=str(run_cfg.meta_dir),
        tags_dir=str(run_cfg.tags_dir),
        caption_field=str(run_cfg.caption_field),
        min_tag_count=int(run_cfg.min_tag_count),
        require_512=bool(run_cfg.require_512),
        val_ratio=float(run_cfg.val_ratio),
        seed=int(run_cfg.seed),
        cache_dir=str(run_cfg.cache_dir),
        failed_list=str(run_cfg.failed_list),
    )

    text_cfg = TextConfig(
        vocab_path=str(model_cfg.text_vocab_path),
        merges_path=str(model_cfg.text_merges_path),
        max_len=int(model_cfg.text_max_len),
        lowercase=True,
        strip_punct=True,
    )

    mode = str(run_cfg.mode)
    train_entries, _val_entries = build_or_load_index(dcfg)

    tokenizer = BPETokenizer.from_files(
        vocab_path=model_cfg.text_vocab_path,
        merges_path=model_cfg.text_merges_path,
        cfg=text_cfg,
    )

    cfg_dict["text_max_len"] = int(text_cfg.max_len)
    with open(out_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "run_meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(build_run_metadata(perf_active), f, sort_keys=False, allow_unicode=True)

    latent_dtype = torch.bfloat16 if run_cfg.latent_dtype == "bf16" else torch.float16
    latent_expected_meta: LatentCacheMetadata | None = None
    if mode == "latent":
        _assert_divisible(int(run_cfg.image_size), int(run_cfg.latent_downsample_factor), "image_size")
        if not bool(run_cfg.latent_cache_strict) and not bool(run_cfg.latent_cache_fallback):
            raise RuntimeError("latent_cache_strict=false requires latent_cache_fallback=true.")
        latent_side = int(run_cfg.image_size) // int(run_cfg.latent_downsample_factor)
        latent_expected_meta = LatentCacheMetadata(
            vae_pretrained=str(run_cfg.vae_pretrained),
            scaling_factor=float(run_cfg.vae_scaling_factor),
            latent_shape=(int(run_cfg.latent_channels), latent_side, latent_side),
            dtype=str(run_cfg.latent_dtype),
        )
    ds = DanbooruDataset(
        entries=train_entries,
        tokenizer=tokenizer,
        cond_drop_prob=float(run_cfg.cond_drop_prob),
        seed=int(run_cfg.seed),
        cache_dir=str(Path(run_cfg.data_root) / run_cfg.cache_dir),
        token_cache_key=build_token_cache_key(
            vocab_path=model_cfg.text_vocab_path,
            merges_path=model_cfg.text_merges_path,
            caption_field=run_cfg.caption_field,
            max_len=int(text_cfg.max_len),
            lowercase=bool(text_cfg.lowercase),
            strip_punct=bool(text_cfg.strip_punct),
        ),
        latent_cache_dir=str(Path(run_cfg.data_root) / run_cfg.latent_cache_dir),
        latent_dtype=latent_dtype,
        return_latents=(mode == "latent"),
        latent_cache_strict=bool(run_cfg.latent_cache_strict),
        latent_cache_fallback=bool(run_cfg.latent_cache_fallback),
        latent_expected_meta=latent_expected_meta,
        include_is_latent=bool(run_cfg.latent_cache_fallback),
        latent_missing_log_path=out_dir / "latent_missing.txt" if mode == "latent" else None,
    )
    if mode == "latent" and len(ds) == 0:
        raise RuntimeError("latent cache is empty; run scripts/prepare_latents.py before training.")

    latent_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    if mode == "latent":
        if not bool(run_cfg.latent_cache):
            raise RuntimeError("mode=latent requires latent_cache=true (use scripts/prepare_latents.py).")
        if bool(run_cfg.latent_cache_fallback):
            if not str(run_cfg.vae_pretrained):
                raise RuntimeError("latent_cache_fallback requires vae_pretrained.")
            vae = VAEWrapper(
                pretrained=str(run_cfg.vae_pretrained),
                freeze=bool(run_cfg.vae_freeze),
                scaling_factor=float(run_cfg.vae_scaling_factor),
                device=device,
                dtype=latent_dtype,
            )

            def _encode_latents(x: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    z = vae.encode(x.to(device=device, dtype=latent_dtype))
                return z.cpu()

            latent_encoder = _encode_latents

    nw = _resolve_num_workers(int(run_cfg.num_workers))
    if latent_encoder is not None:
        nw = 0
    dl = DataLoader(
        ds,
        batch_size=int(run_cfg.batch_size),
        shuffle=True,
        num_workers=nw,
        pin_memory=bool(run_cfg.pin_memory),
        drop_last=True,
        persistent_workers=bool(run_cfg.persistent_workers) and nw > 0,
        prefetch_factor=int(run_cfg.prefetch_factor) if nw > 0 else None,
        collate_fn=lambda batch: collate_with_tokenizer(batch, latent_encoder=latent_encoder),
    )

    # ----------------------------
    # Model
    # ----------------------------
    if mode == "latent":
        _assert_divisible(int(run_cfg.image_size), int(run_cfg.latent_downsample_factor), "image_size")

    image_channels = 3 if mode == "pixel" else int(run_cfg.latent_channels)
    unet_cfg = UNetConfig(
        image_channels=image_channels,
        base_channels=int(model_cfg.base_channels),
        channel_mults=tuple(model_cfg.channel_mults),
        num_res_blocks=int(model_cfg.num_res_blocks),
        dropout=float(model_cfg.dropout),
        attn_resolutions=tuple(model_cfg.attn_resolutions),
        attn_heads=int(model_cfg.attn_heads),
        attn_head_dim=int(model_cfg.attn_head_dim),
        vocab_size=len(tokenizer.vocab),
        text_dim=int(model_cfg.text_dim),
        text_layers=int(model_cfg.text_layers),
        text_heads=int(model_cfg.text_heads),
        text_max_len=int(model_cfg.text_max_len),
        use_scale_shift_norm=bool(model_cfg.use_scale_shift_norm),
        grad_checkpointing=bool(run_cfg.grad_checkpointing),
    )

    model = UNet(unet_cfg).to(device)
    if bool(run_cfg.channels_last):
        model = model.to(memory_format=torch.channels_last)

    # NOTE: torch.compile can increase startup time and VRAM usage.
    # Compare throughput with compile=False if performance regresses.
    if bool(run_cfg.compile) and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_cfg.lr),
        weight_decay=float(run_cfg.weight_decay),
        fused=(device.type == "cuda"),
    )

    use_amp = bool(run_cfg.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if run_cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=float(run_cfg.ema_decay))

    prediction_type = str(model_cfg.prediction_type)
    diff = Diffusion(
        DiffusionConfig(
            timesteps=int(model_cfg.timesteps),
            beta_start=float(model_cfg.beta_start),
            beta_end=float(model_cfg.beta_end),
            prediction_type=prediction_type,
        ),
        device=device,
    )
    if mode == "latent":
        domain = LatentDomain(
            diffusion=diff,
            device=device,
            channels_last=bool(run_cfg.channels_last),
        )
    else:
        domain = PixelDomain(
            diffusion=diff,
            device=device,
            channels_last=bool(run_cfg.channels_last),
        )

    start_step = 0
    if resume:
        if ck is None:
            ck = load_ckpt(resume, device)
        model_state = normalize_state_dict_for_model(ck["model"], model, name="model")
        model.load_state_dict(model_state, strict=True)
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        elif "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        if "ema" in ck:
            ema_state = normalize_state_dict_for_keys(ck["ema"], ema.shadow.keys(), name="ema")
            ema.shadow = {k: v.to(device) for k, v in ema_state.items()}
        for group in opt.param_groups:
            group["lr"] = float(run_cfg.lr)
        start_step = int(ck.get("step", 0)) + 1

    # ----------------------------
    # Train loop
    # ----------------------------
    max_steps = int(run_cfg.max_steps)
    grad_accum = int(run_cfg.grad_accum_steps)
    log_every = int(run_cfg.log_every)
    save_every = int(run_cfg.save_every)
    min_snr_gamma = float(run_cfg.min_snr_gamma)
    grad_clip = float(run_cfg.grad_clip_norm)
    compile_warmup_steps = int(run_cfg.compile_warmup_steps) if bool(run_cfg.compile) else 0
    warmup_end_step = start_step + max(compile_warmup_steps, 0)
    compile_cudagraphs = bool(run_cfg.compile_cudagraphs)
    if grad_accum > 1:
        compile_cudagraphs = False
    _configure_cudagraphs(compile_cudagraphs)

    webui_mode = _is_webui_mode()
    metrics_path = _webui_metrics_path()

    is_main = _dist_rank() == 0

    pbar = tqdm(
        total=int(run_cfg.max_steps),
        initial=start_step,
        desc="train",
        unit="step",
        disable=webui_mode or not is_main,  # важное: webui -> без tqdm, иначе мусор в stdout
    )

    log_every = int(run_cfg.log_every)
    metrics_dir = Path(run_cfg.out_dir) / "metrics"
    events_path = metrics_dir / "events.jsonl"
    sinks = [JsonlFileSink(events_path)]
    if webui_mode:
        sinks.append(StdoutJsonSink())
    if metrics_path is not None:
        sinks.append(JsonlFileSink(metrics_path, event_types=["metric"]))
    event_bus = EventBus(sinks)
    timing = _TimingStats(
        use_cuda_events=(device.type == "cuda"),
        gpu_sections={"fwd_bwd", "opt_step"},
    )

    def _log(message: str, step: Optional[int] = None) -> None:
        if webui_mode and is_main:
            event_bus.emit({
                "type": "log",
                "message": message,
                "step": int(step) if step is not None else start_step,
            })
        elif is_main:
            print(message)

    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_step = start_step

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


    stop_requested = {"value": False}

    def _request_stop(signum, _frame) -> None:
        stop_requested["value"] = True
        _log(f"[SIGNAL] stop requested via {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    it = iter(dl)
    start_time = time.perf_counter()
    last_log = start_time

    sanity_steps = int(run_cfg.sanity_overfit_steps)
    sanity_images = int(run_cfg.sanity_overfit_images)
    sanity_max_loss = float(run_cfg.sanity_overfit_max_loss)
    _sanity_overfit(
        model=model,
        tokenizer=tokenizer,
        entries=train_entries,
        diff=diff,
        domain=domain,
        latent_mode=(mode == "latent"),
        latent_cache_dir=str(Path(run_cfg.data_root) / run_cfg.latent_cache_dir) if mode == "latent" else None,
        latent_dtype=latent_dtype if mode == "latent" else None,
        latent_cache_strict=bool(run_cfg.latent_cache_strict),
        latent_cache_fallback=bool(run_cfg.latent_cache_fallback),
        latent_expected_meta=latent_expected_meta,
        latent_encoder=latent_encoder,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        steps=sanity_steps,
        max_images=sanity_images,
        max_loss=sanity_max_loss,
        opt=opt,
        scaler=scaler,
        ema=ema,
        log_fn=_log if is_main else None,
    )

    if is_main:
        if mode == "latent":
            cache_hit_rate = ds.latent_cache_hit_rate()
            if cache_hit_rate is None:
                _log("[WARN] latent cache hit rate unavailable")
            else:
                _log(f"[INFO] latent cache hit rate={cache_hit_rate:.2%} missing={ds.latent_cache_missing}")
                if cache_hit_rate == 0.0:
                    _log("[WARN] latent cache appears empty; did you run prepare_latents.py?")
        event_bus.emit({
            "type": "log",
            "message": (
                "runtime flags: "
                f"compile={bool(run_cfg.compile)}, "
                f"compile_cudagraphs={compile_cudagraphs}, "
                f"tf32={perf_active['tf32']}, "
                f"channels_last={perf_active['channels_last']}, "
                f"sdp_flash={perf_active['sdp_flash']}, "
                f"sdp_mem_efficient={perf_active['sdp_mem_efficient']}, "
                f"sdp_math={perf_active['sdp_math']}"
            ),
            "step": start_step,
        })
        event_bus.emit({
            "type": "status",
            "status": "start",
            "step": start_step,
            "resume": bool(resume),
            "out_dir": str(out_dir),
        })

    log_loss_sum = 0.0
    log_loss_count = 0
    best_loss: Optional[float] = None
    best_step: Optional[int] = None

    for step in range(start_step, max_steps):
        if bool(run_cfg.compile) and compile_cudagraphs:
            _cudagraph_step_begin()
        step_start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        data_time = 0.0
        fwd_bwd_time = 0.0
        last_batch_stats = {"x_std": None, "v_std": None}

        for _ in range(grad_accum):
            with timing.section("data_fetch") as t_data:
                try:
                    x0, txt_ids, txt_mask = next(it)
                except StopIteration:
                    it = iter(dl)
                    x0, txt_ids, txt_mask = next(it)
            data_time += t_data.elapsed_sec

            batch = Batch(x=x0, txt_ids=txt_ids, txt_mask=txt_mask, domain=domain.name)
            prepared = domain.prepare_batch(batch)
            x0 = prepared.x
            txt_ids = prepared.txt_ids
            txt_mask = prepared.txt_mask

            b = x0.shape[0]
            t = torch.randint(0, diff.cfg.timesteps, (b,), device=x0.device, dtype=torch.long)
            noise = domain.sample_noise_like(x0)
            alpha_bar_t = diff.alpha_bar[t]
            _assert_finite("alpha_bar[t]", alpha_bar_t)
            xt = domain.q_sample(x0, t, noise)
            v_tgt = domain.v_target(x0, t, noise)
            last_batch_stats["x_std"] = float(x0.detach().std().cpu().item())
            last_batch_stats["v_std"] = float(v_tgt.detach().std().cpu().item())

            with timing.section("fwd_bwd") as t_fwd:
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    v_pred = model(xt, t, txt_ids, txt_mask)
                    _assert_finite("v_pred", v_pred)
                    if v_pred.shape != v_tgt.shape:
                        raise RuntimeError("v_pred/v_target shape mismatch")
                    per = F.mse_loss(
                        v_pred,
                        v_tgt.to(dtype=v_pred.dtype),
                        reduction="none",
                    ).mean(dim=[1, 2, 3])  # [B]
                    w = get_min_snr_weights(diff.alpha_bar[t], gamma=min_snr_gamma)        # [B]
                    loss = (per * w).mean() / grad_accum

            if not torch.isfinite(loss):
                dump_path = out_dir / f"nan_dump_{step:07d}.pt"
                save_ckpt(str(dump_path), {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": cfg_dict,
                    "meta": build_run_metadata(perf_active),
                    "batch_stats": {
                        "x0_min": float(x0.min().item()),
                        "x0_max": float(x0.max().item()),
                        "alpha_bar_min": float(alpha_bar_t.min().item()),
                        "alpha_bar_max": float(alpha_bar_t.max().item()),
                    },
                })
                raise RuntimeError(f"Non-finite loss at step={step}: {loss.item()}")

            total_loss += float(loss.detach().cpu())
            scaler.scale(loss).backward()
            fwd_bwd_time += t_fwd.elapsed_sec

        opt_step_start = time.perf_counter()
        if scaler.is_enabled():
            scaler.unscale_(opt)
        if grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
        else:
            grad_norm = _grad_norm(model)

        with timing.section("opt_step") as t_opt:
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        ema.update(model)

        bad_grads = _find_bad_grads(model)
        if bad_grads:
            dump_path = out_dir / f"nan_dump_{step:07d}.pt"
            save_ckpt(str(dump_path), {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.shadow,
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "cfg": cfg_dict,
                "meta": build_run_metadata(perf_active),
                "batch_stats": {
                    "bad_grads": bad_grads[:10],
                },
            })
            raise RuntimeError(f"Non-finite grads at step={step}: {bad_grads[:5]}")

        opt_time = time.perf_counter() - opt_step_start
        step_time = time.perf_counter() - step_start
        timing.add_cpu("step_total", step_time)

        log_loss_sum += float(total_loss)
        log_loss_count += 1

        if step % log_every == 0 and step >= warmup_end_step:
            if device.type == "cuda":
                torch.cuda.synchronize()

            now = time.perf_counter()
            elapsed = now - last_log_time
            steps_done = max(step - last_log_step, 1)

            # сколько "картинок" реально прошло за этот лог-интервал
            images = steps_done * int(run_cfg.batch_size) * int(run_cfg.grad_accum_steps)
            img_per_sec = images / max(elapsed, 1e-9)

            peak_mem = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda"
                else 0.0
            )

            total_elapsed = now - start_time
            steps_left = int(run_cfg.max_steps) - step - 1
            sec_per_step = elapsed / steps_done
            eta_h = (steps_left * sec_per_step) / 3600.0

            loss_sum = log_loss_sum
            loss_count = log_loss_count
            if _dist_is_initialized():
                stats = torch.tensor([loss_sum, loss_count], device=device, dtype=torch.float64)
                stats = _dist_all_reduce_sum(stats)
                loss_sum = float(stats[0].item())
                loss_count = int(stats[1].item())
            loss_mean = loss_sum / max(loss_count, 1)

            # CLI-UI (только если НЕ webui)
            if not webui_mode and is_main:
                pbar.set_postfix({
                    "loss": f"{loss_mean:.6f}",
                    "img/s": f"{img_per_sec:.2f}",
                    "mem(MB)": f"{peak_mem:.0f}",
                    "eta(h)": f"{eta_h:.2f}",
                })

            if is_main:
                timing_stats = timing.report(reset=True)
                payload = {
                    "type": "metric",
                    "step": step,
                    "loss": float(loss_mean),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "grad_norm": float(grad_norm),
                    "x_std": last_batch_stats["x_std"],
                    "v_std": last_batch_stats["v_std"],
                    "ema_decay": float(run_cfg.ema_decay),
                    "cfg_drop_prob": float(run_cfg.cond_drop_prob),
                    "img_per_sec": float(img_per_sec),
                    "peak_mem_mb": float(peak_mem),
                    "elapsed_sec": float(total_elapsed),
                    "eta_h": float(eta_h),
                    "sec_per_step": float(sec_per_step),
                    "data_time_sec": float(data_time),
                    "forward_backward_time_sec": float(fwd_bwd_time),
                    "optimizer_step_time_sec": float(opt_time),
                    "total_step_time_sec": float(step_time),
                    "timing": timing_stats,
                    "max_steps": int(run_cfg.max_steps),
                    "latent_cache_hit_rate": float(ds.latent_cache_hit_rate() or 0.0) if mode == "latent" else None,
                    "domain": domain.name,
                }
                event_bus.emit(payload)
                if best_loss is None or loss_mean < best_loss:
                    best_loss = float(loss_mean)
                    best_step = int(step)
                    best_path = out_dir / "ckpt_best.pt"
                    save_ckpt(str(best_path), {
                        "step": step,
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "ema": ema.shadow,
                        "cfg": cfg_dict,
                        "meta": build_run_metadata(perf_active),
                    })
                    event_bus.emit({
                        "type": "log",
                        "message": f"saved best checkpoint {best_path}",
                        "step": step,
                        "best_loss": best_loss,
                    })

            last_log_time = now
            last_log_step = step
            log_loss_sum = 0.0
            log_loss_count = 0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        elif step + 1 == warmup_end_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            last_log_time = time.perf_counter()
            last_log_step = step
            log_loss_sum = 0.0
            log_loss_count = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if stop_requested["value"]:
            stop_path = out_dir / f"ckpt_stop_{step:07d}.pt"
            stop_payload = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg_dict,
                "meta": build_run_metadata(perf_active),
            }
            save_ckpt(str(stop_path), stop_payload)
            latest_path = out_dir / "ckpt_latest.pt"
            save_ckpt(str(latest_path), stop_payload)
            if is_main:
                event_bus.emit({
                    "type": "status",
                    "status": "stopped",
                    "step": step,
                    "ckpt": str(stop_path),
                })
            _log(f"[STOP] saved {stop_path}", step=step)
            return

        if step % save_every == 0 and step > 0:
            ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
            ckpt_payload = {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.shadow,
                "cfg": cfg_dict,
                "meta": build_run_metadata(perf_active),
            }
            save_ckpt(str(ckpt_path), ckpt_payload)
            latest_path = out_dir / "ckpt_latest.pt"
            save_ckpt(str(latest_path), ckpt_payload)
            _prune_checkpoints(out_dir, int(run_cfg.ckpt_keep_last))
            if is_main:
                event_bus.emit({
                    "type": "log",
                    "message": f"saved {ckpt_path}",
                    "step": step,
                })
            _log(f"[OK] saved {ckpt_path}", step=step)

    final_path = out_dir / "ckpt_final.pt"
    final_payload = {
        "step": max_steps - 1,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.shadow,
        "cfg": cfg_dict,
        "meta": build_run_metadata(perf_active),
    }
    save_ckpt(str(final_path), final_payload)
    latest_path = out_dir / "ckpt_latest.pt"
    save_ckpt(str(latest_path), final_payload)
    _prune_checkpoints(out_dir, int(run_cfg.ckpt_keep_last))
    if is_main:
        event_bus.emit({
            "type": "status",
            "status": "done",
            "step": max_steps - 1,
            "ckpt": str(final_path),
        })
    _log(f"[DONE] saved {final_path}", step=max_steps - 1)


if __name__ == "__main__":
    main()
