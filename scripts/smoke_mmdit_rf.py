from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
if ROOT_STR in sys.path:
    sys.path.remove(ROOT_STR)
sys.path.insert(0, ROOT_STR)

from config.train import TrainConfig
from data_loader import DataConfig, ImageTextDataset, LatentCacheMetadata, build_or_load_index
from diffusion.objectives import RectifiedFlowObjective
from diffusion.utils import load_ckpt, save_ckpt
from model.mmdit import MMDiTConfig, MMDiTFlowModel
from model.text.cache import TextCache
from model.text.conditioning import TextConditioning, TrainBatch
from samplers.flow_heun import sample_flow_heun
from train.loop_mmdit import training_step_mmdit


def _load_eval_prompts(path: str, count: int) -> list[str]:
    prompts_path = Path(path)
    if not prompts_path.exists():
        raise RuntimeError(f"Eval prompts file not found: {prompts_path}")
    lines = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in lines if line]
    if len(prompts) < count:
        raise RuntimeError(f"Eval prompts file must contain at least {count} non-empty lines.")
    return prompts[:count]


def _resolve_eval_prompts(path: str, count: int, *, use_text_conditioning: bool) -> list[str]:
    if not use_text_conditioning:
        return [""] * count
    return _load_eval_prompts(path, count=count)


def _mmdit_entry_text(entry: dict) -> str:
    caption = str(entry.get("caption", "") or "")
    if caption:
        return caption
    tags = list(entry.get("tags_primary", [])) + list(entry.get("tags_gender", []))
    return " ".join(str(x) for x in tags)


def _mmdit_dataset_hash(entries: list[dict]) -> str:
    h = hashlib.sha256()
    for entry in entries:
        h.update(str(entry.get("md5", "")).encode("utf-8"))
        h.update(b"\0")
        h.update(_mmdit_entry_text(entry).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _validate_text_cache_for_mmdit(cache: TextCache, cfg: TrainConfig, entries: list[dict]) -> None:
    if not cache.index_path.exists():
        raise RuntimeError(f"Missing text cache index: {cache.index_path}")
    if not cache.metadata:
        raise RuntimeError(f"Missing text cache metadata: {cache.metadata_path}")

    meta = cache.metadata
    if int(meta.get("text_dim", -1)) != int(cfg.text_dim):
        raise RuntimeError(f"text cache text_dim mismatch: {meta.get('text_dim')} != {cfg.text_dim}")
    if int(meta.get("pooled_dim", -1)) != int(cfg.pooled_dim):
        raise RuntimeError(f"text cache pooled_dim mismatch: {meta.get('pooled_dim')} != {cfg.pooled_dim}")

    expected_encoders = cfg.extra.get("text", {}).get("encoders", [])
    actual_encoders = meta.get("encoders", [])
    if expected_encoders and actual_encoders:
        expected = [
            {
                "name": str(item.get("name", "")),
                "model_name": str(item.get("model_name", "")),
                "max_length": int(item.get("max_length", 0)),
            }
            for item in expected_encoders
        ]
        actual = [
            {
                "name": str(item.get("name", "")),
                "model_name": str(item.get("model_name", "")),
                "max_length": int(item.get("max_length", 0)),
            }
            for item in actual_encoders
        ]
        if expected != actual:
            raise RuntimeError(f"text cache encoder metadata mismatch: cache={actual!r}, config={expected!r}")

    expected_hash = meta.get("dataset_hash")
    if isinstance(expected_hash, str) and expected_hash:
        actual_hash = _mmdit_dataset_hash(entries)
        if actual_hash != expected_hash:
            raise RuntimeError(f"text cache dataset_hash mismatch: {expected_hash} != {actual_hash}")

    missing = [str(entry.get("md5", "")) for entry in entries if str(entry.get("md5", "")) not in cache.entries]
    if missing:
        examples = ", ".join(missing[:10])
        raise RuntimeError(f"text cache missing {len(missing)} md5 keys used by dataset. Examples: {examples}")


def _batch_text(text: TextConditioning) -> TextConditioning:
    return TextConditioning(
        tokens=text.tokens.unsqueeze(0) if text.tokens.dim() == 2 else text.tokens,
        mask=text.mask.unsqueeze(0) if text.mask.dim() == 1 else text.mask,
        pooled=text.pooled.unsqueeze(0) if text.pooled.dim() == 1 else text.pooled,
        is_uncond=text.is_uncond.view(1) if text.is_uncond is not None and text.is_uncond.dim() == 0 else text.is_uncond,
    )


def _load_first_batch(cfg: TrainConfig) -> tuple[torch.Tensor, TextConditioning, dict[str, object]]:
    dcfg = DataConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        images_only=False,
        use_text_conditioning=True,
        min_tag_count=int(cfg.min_tag_count),
        require_512=bool(cfg.require_512),
        val_ratio=float(cfg.val_ratio),
        seed=int(cfg.seed),
        cache_dir=str(cfg.cache_dir),
        failed_list=str(cfg.failed_list),
    )
    train_entries, val_entries = build_or_load_index(dcfg)
    if int(cfg.dataset_limit) > 0:
        train_entries = train_entries[: int(cfg.dataset_limit)]
        val_entries = []
    if not train_entries:
        raise RuntimeError("No train entries available for MMDiT smoke test.")

    text_cache = TextCache(Path(cfg.data_root) / str(cfg.text_cache_dir), shard_cache_size=int(cfg.text_shard_cache_size))
    _validate_text_cache_for_mmdit(text_cache, cfg, train_entries + val_entries)

    latent_side = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    latent_dtype = torch.bfloat16 if cfg.latent_dtype == "bf16" else torch.float16
    latent_ds = ImageTextDataset(
        entries=train_entries,
        tokenizer=None,
        cond_drop_prob=1.0,
        seed=int(cfg.seed),
        latent_cache_dir=str(Path(cfg.data_root) / cfg.latent_cache_dir),
        latent_cache_sharded=bool(cfg.latent_cache_sharded),
        latent_cache_index_path=str(cfg.latent_cache_index),
        latent_dtype=latent_dtype,
        return_latents=True,
        latent_cache_strict=bool(cfg.latent_cache_strict),
        latent_cache_fallback=False,
        latent_expected_meta=LatentCacheMetadata(
            vae_pretrained=str(cfg.vae_pretrained),
            scaling_factor=float(cfg.vae_scaling_factor),
            latent_shape=(int(cfg.latent_channels), latent_side, latent_side),
            dtype=str(cfg.latent_dtype),
        ),
        include_is_latent=False,
    )
    if len(latent_ds) == 0:
        raise RuntimeError("Latent cache has no entries for MMDiT smoke test.")
    x0 = latent_ds[0][0].unsqueeze(0).float()
    key = str(train_entries[0]["md5"])
    text = _batch_text(text_cache.load(key))
    text = TextConditioning(
        tokens=text.tokens.float(),
        mask=text.mask,
        pooled=text.pooled.float(),
        is_uncond=text.is_uncond,
    )
    diagnostics = {
        "mode": "cache",
        "train_entries": len(train_entries),
        "val_entries": len(val_entries),
        "text_cache_entries": len(text_cache.entries),
        "latent_cache_entries": len(latent_ds),
        "first_md5": key,
    }
    return x0, text, diagnostics


def _synthetic_batch(cfg: TrainConfig) -> tuple[torch.Tensor, TextConditioning, dict[str, object]]:
    configured_latent_side = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    patch = int(cfg.latent_patch_size)
    if configured_latent_side <= 0:
        raise RuntimeError("Synthetic MMDiT smoke requires positive latent side.")
    if patch <= 0:
        raise RuntimeError("Synthetic MMDiT smoke requires positive latent_patch_size.")
    latent_side = max(patch, min(configured_latent_side, 8))
    latent_side = max(patch, (latent_side // patch) * patch)

    gen = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    batch_size = 1
    text_len = max(1, min(int(cfg.text_max_len), 16))
    x0 = torch.randn(batch_size, int(cfg.latent_channels), latent_side, latent_side, generator=gen)
    text = TextConditioning(
        tokens=torch.randn(batch_size, text_len, int(cfg.text_dim), generator=gen),
        mask=torch.ones(batch_size, text_len, dtype=torch.bool),
        pooled=torch.randn(batch_size, int(cfg.pooled_dim), generator=gen),
        is_uncond=torch.zeros(batch_size, dtype=torch.bool),
    )
    diagnostics = {
        "mode": "synthetic",
        "train_entries": batch_size,
        "val_entries": 0,
        "text_cache_entries": "synthetic",
        "latent_cache_entries": "synthetic",
        "first_md5": "synthetic",
        "configured_latent_shape": (int(cfg.latent_channels), configured_latent_side, configured_latent_side),
    }
    return x0, text, diagnostics


def run(config_path: str, *, synthetic: bool = False) -> None:
    cfg = TrainConfig.from_yaml(config_path)
    if cfg.architecture != "mmdit_rf":
        raise RuntimeError("smoke_mmdit_rf requires architecture=mmdit_rf.")
    if not synthetic and int(cfg.eval_every) > 0:
        _resolve_eval_prompts(str(cfg.eval_prompts_file), 5, use_text_conditioning=True)

    x0, text, diagnostics = _synthetic_batch(cfg) if synthetic else _load_first_batch(cfg)
    model = MMDiTFlowModel(MMDiTConfig.from_dict(cfg.to_dict()))
    model_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] smoke mode: {diagnostics.get('mode', 'cache')}")
    print(f"[INFO] train entries: {diagnostics['train_entries']}")
    print(f"[INFO] val entries: {diagnostics['val_entries']}")
    print(f"[INFO] text cache entries: {diagnostics['text_cache_entries']}")
    print(f"[INFO] latent cache entries: {diagnostics['latent_cache_entries']}")
    print(f"[INFO] first md5: {diagnostics['first_md5']}")
    if "configured_latent_shape" in diagnostics:
        print(f"[INFO] configured latent shape: {diagnostics['configured_latent_shape']}")
    print(f"[INFO] first latent shape: {tuple(x0.shape)}")
    print(f"[INFO] text tokens: {tuple(text.tokens.shape)}")
    print(f"[INFO] text pooled: {tuple(text.pooled.shape)}")
    print(f"[INFO] model params: {model_params:,}")
    objective = RectifiedFlowObjective(
        timestep_sampling=str(cfg.flow_timestep_sampling),
        logit_mean=float(cfg.flow_logit_mean),
        logit_std=float(cfg.flow_logit_std),
        train_t_min=float(cfg.flow_train_t_min),
        train_t_max=float(cfg.flow_train_t_max),
        loss_weighting=str(cfg.flow_loss_weighting),
    )

    t = torch.rand(x0.shape[0])
    out = model(x0, t, text)
    if out.shape != x0.shape:
        raise RuntimeError(f"MMDiT smoke forward shape mismatch: {tuple(out.shape)} != {tuple(x0.shape)}")

    loss = training_step_mmdit(
        model=model,
        objective=objective,
        batch=TrainBatch(x0=x0, text=text),
        amp_enabled=False,
    )
    loss.backward()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / ("synthetic_smoke_ckpt.pt" if synthetic else "smoke_ckpt.pt")
    save_ckpt(
        str(ckpt_path),
        {
            "model": model.state_dict(),
            "step": 1,
            "cfg": cfg.to_dict(),
            "architecture": "mmdit_rf",
            "objective": "rectified_flow",
        },
    )
    loaded = load_ckpt(str(ckpt_path), torch.device("cpu"))
    model.load_state_dict(loaded["model"], strict=True)
    print(f"[INFO] checkpoint path: {ckpt_path}")

    gen = torch.Generator().manual_seed(int(cfg.seed))
    sampler_steps = 2 if synthetic else int(cfg.sampling_steps)
    sample_flow_heun(
        model=model,
        shape=tuple(x0.shape),
        text_cond=text,
        uncond=text,
        steps=sampler_steps,
        cfg_scale=float(cfg.sampling_cfg_scale),
        shift=float(cfg.sampling_shift),
        generator=gen,
    )
    print(f"[INFO] sampler status: flow_heun {sampler_steps} steps ok")
    print(f"[OK] MMDiT RF smoke passed: {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/train_smoke.yaml")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run architecture-only smoke without dataset, latent cache, text cache, VAE, or transformers.",
    )
    args = parser.parse_args()
    run(args.config, synthetic=bool(args.synthetic))


if __name__ == "__main__":
    main()
