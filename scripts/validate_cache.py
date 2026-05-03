from __future__ import annotations

import argparse
import json
from pathlib import Path



def _fail_or_warn(message: str, *, strict: bool) -> None:
    if strict:
        raise RuntimeError(message)
    print(f"[WARN] {message}", flush=True)


def validate_cache(cfg, *, strict: bool | None = None, text_only: bool = False, latents_only: bool = False) -> dict:
    from model.text.cache import TextCache

    """Validate the training caches referenced by a resolved TrainConfig.

    The command is intentionally metadata/readability focused. It does not try
    to regenerate cache contents; use ``prepare_training_cache --repair`` for
    that. It is safe to run before a long training job.
    """
    strict = bool(cfg.cache_strict if strict is None else strict)
    report: dict[str, object] = {"ok": True, "text_cache": None, "latent_cache": None, "training_manifest": None}
    data_root = Path(cfg.data_root)

    if not latents_only:
        text_root = data_root / str(cfg.text_cache_dir)
        if not cfg.text_cache:
            _fail_or_warn("text_cache is disabled in config", strict=strict)
        else:
            cache = TextCache(text_root, shard_cache_size=int(cfg.text_shard_cache_size))
            cache.validate_files_readable()
            expected_text_dim = int(cfg.text_dim)
            actual_text_dim = int(cache.metadata.get("text_dim", -1))
            if actual_text_dim != expected_text_dim:
                _fail_or_warn(f"text cache text_dim mismatch: {actual_text_dim} != {expected_text_dim}", strict=strict)
            expected_pooled_dim = int(cfg.pooled_dim)
            actual_pooled_dim = int(cache.metadata.get("pooled_dim", -1))
            if actual_pooled_dim != expected_pooled_dim:
                _fail_or_warn(f"text cache pooled_dim mismatch: {actual_pooled_dim} != {expected_pooled_dim}", strict=strict)
            report["text_cache"] = {
                "root": str(text_root),
                "num_entries": len(cache.entries),
                "num_shards": len(cache.shard_names()),
                "has_manifest": bool(cache.manifest),
            }

    if not text_only:
        latent_root = data_root / str(cfg.latent_cache_dir)
        if not cfg.latent_cache:
            _fail_or_warn("latent_cache is disabled in config", strict=strict)
        else:
            index = Path(cfg.latent_cache_index)
            index_path = index if index.is_absolute() else latent_root / index
            if not index_path.exists():
                _fail_or_warn(f"missing latent cache index: {index_path}", strict=strict)
            else:
                rows = [line for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                report["latent_cache"] = {"root": str(latent_root), "index": str(index_path), "num_entries": len(rows)}

    manifest_path = data_root / str(cfg.cache_dir) / "training_cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["training_manifest"] = {
            "path": str(manifest_path),
            "version": manifest.get("version"),
            "num_samples": manifest.get("num_samples"),
            "dataset_hash": manifest.get("dataset_hash", ""),
        }
    elif strict and not (text_only or latents_only):
        raise RuntimeError(f"missing unified training cache manifest: {manifest_path}")
    else:
        report["training_manifest"] = None
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate text/latent training cache metadata and shard readability.")
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors, overriding config cache.strict.")
    parser.add_argument("--non-strict", action="store_true", help="Turn compatible mismatches into warnings.")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--latents-only", action="store_true")
    args = parser.parse_args(argv)

    if args.text_only and args.latents_only:
        raise SystemExit("--text-only and --latents-only are mutually exclusive")
    from config.train import TrainConfig

    cfg = TrainConfig.from_yaml(args.config)
    strict = True if args.strict else False if args.non_strict else None
    report = validate_cache(cfg, strict=strict, text_only=bool(args.text_only), latents_only=bool(args.latents_only))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
