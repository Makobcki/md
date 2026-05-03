from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.train import TrainConfig
from data_loader import DataConfig, build_or_load_index, latent_cache_path, load_image_tensor
from diffusion.events import EventBus, JsonlFileSink, StdoutJsonSink
from diffusion.utils.oom import is_torch_oom_error, print_torch_oom
from diffusion.utils import build_run_metadata
from diffusion.vae import VAEWrapper


def _latent_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError("latent_dtype must be 'fp16' or 'bf16'.")


@dataclass(frozen=True)
class _LatentPrepareOptions:
    overwrite: bool = False
    limit: Optional[int] = None
    batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    device: str = "auto"
    latent_dtype: str = "fp16"
    autocast_dtype: str = "fp16"
    queue_size: int = 64
    writer_threads: int = 1
    shard_size: int = 4096
    stats_every_sec: float = 5.0
    decode_backend: str = "auto"


def _provided_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_dests: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_dests[option] = action.dest

    provided: set[str] = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_dests.get(option)
        if dest is not None:
            provided.add(dest)
    return provided


def _latent_prepare_config_values(cfg: TrainConfig) -> dict[str, Any]:
    valid = {field.name for field in fields(_LatentPrepareOptions)}
    values: dict[str, Any] = {}

    section = cfg.extra.get("prepare_latents", cfg.extra.get("latent_prepare"))
    if section is not None:
        if not isinstance(section, dict):
            raise RuntimeError("prepare_latents/latent_prepare config section must be a mapping.")
        values.update({str(k).replace("-", "_"): v for k, v in section.items()})

    prefix = "latent_prepare_"
    for key, value in cfg.extra.items():
        if key.startswith(prefix):
            values[key[len(prefix):].replace("-", "_")] = value

    unknown = sorted(set(values) - valid)
    if unknown:
        raise RuntimeError("Unknown latent_prepare config option(s): " + ", ".join(unknown))
    return values


def _coerce_prepare_options(options: _LatentPrepareOptions) -> _LatentPrepareOptions:
    limit = None if options.limit in (None, "") else int(options.limit)
    coerced = _LatentPrepareOptions(
        overwrite=bool(options.overwrite),
        limit=limit,
        batch_size=int(options.batch_size),
        num_workers=int(options.num_workers),
        prefetch_factor=int(options.prefetch_factor),
        pin_memory=bool(options.pin_memory),
        device=str(options.device),
        latent_dtype=str(options.latent_dtype),
        autocast_dtype=str(options.autocast_dtype),
        queue_size=int(options.queue_size),
        writer_threads=int(options.writer_threads),
        shard_size=int(options.shard_size),
        stats_every_sec=float(options.stats_every_sec),
        decode_backend=str(options.decode_backend),
    )
    if coerced.device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError("latent_prepare_device must be one of: auto, cpu, cuda.")
    if coerced.latent_dtype not in {"fp16", "bf16"}:
        raise RuntimeError("latent_prepare_latent_dtype must be 'fp16' or 'bf16'.")
    if coerced.autocast_dtype not in {"fp16", "bf16"}:
        raise RuntimeError("latent_prepare_autocast_dtype must be 'fp16' or 'bf16'.")
    if coerced.decode_backend not in {"auto", "pil", "torchvision"}:
        raise RuntimeError("latent_prepare_decode_backend must be one of: auto, pil, torchvision.")
    return coerced


def prepare_latent_cache_for_config(cfg: TrainConfig, *, overwrite: bool | None = None) -> None:
    """Callable API used by training auto-cache preparation."""
    with tempfile.TemporaryDirectory(prefix="md-latent-config-") as tmp_dir:
        config_path = Path(tmp_dir) / "train.yaml"
        config_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        argv = ["--config", str(config_path)]
        if overwrite is not None:
            argv.append("--overwrite" if overwrite else "--no-overwrite")
        _main_impl(argv)


def _latent_meta_mismatch_reason(expected: dict[str, Any], actual: dict[str, Any]) -> str | None:
    comparisons = (
        ("vae_pretrained", str(expected.get("vae_pretrained", "")), str(actual.get("vae_pretrained", ""))),
        ("scaling_factor", float(expected.get("scaling_factor", 0.0)), float(actual.get("scaling_factor", 0.0))),
        ("latent_shape", list(expected.get("latent_shape", [])), list(actual.get("latent_shape", []))),
        ("dtype", str(expected.get("dtype", "")), str(actual.get("dtype", ""))),
        ("format_version", int(expected.get("format_version", 0)), int(actual.get("format_version", 0))),
    )
    for key, expected_value, actual_value in comparisons:
        if key == "scaling_factor":
            if abs(float(expected_value) - float(actual_value)) <= 1e-6:
                continue
        elif actual_value == expected_value:
            continue
        return f"{key}: {actual_value!r} != {expected_value!r}"
    return None


def _sharded_cache_mismatch_reason(
    *,
    index_path: Path,
    shard_dir: Path,
    expected_meta: dict[str, Any],
) -> str | None:
    if not index_path.exists():
        return None
    shards = sorted(shard_dir.glob("shard_*.pt"))
    if not shards:
        return "index exists but no shard files were found"
    try:
        payload = torch.load(shards[0], map_location="cpu")
    except Exception as exc:
        return f"cannot read {shards[0]}: {exc}"
    if not isinstance(payload, dict) or int(payload.get("format_version", 0)) != 3:
        return f"invalid shard payload format in {shards[0]}"
    actual_meta = payload.get("meta_common")
    if not isinstance(actual_meta, dict):
        return f"missing shard metadata in {shards[0]}"
    return _latent_meta_mismatch_reason(expected_meta, actual_meta)


def _resolve_prepare_options(
    cfg: TrainConfig,
    args: argparse.Namespace,
    provided_dests: set[str],
) -> _LatentPrepareOptions:
    options = _LatentPrepareOptions(
        limit=int(cfg.dataset_limit) if int(getattr(cfg, "dataset_limit", 0)) > 0 else None,
        latent_dtype=str(cfg.latent_dtype),
        autocast_dtype=str(cfg.latent_dtype),
        shard_size=4096 if bool(cfg.latent_cache_sharded) else 0,
    )
    config_values = _latent_prepare_config_values(cfg)
    if config_values:
        options = replace(options, **config_values)

    cli_values = {
        key: getattr(args, key)
        for key in provided_dests
        if key != "config" and hasattr(args, key)
    }
    if cli_values:
        options = replace(options, **cli_values)
    return _coerce_prepare_options(options)


def _init_latent_stats() -> dict:
    return {
        "count": 0,
        "mean": 0.0,
        "m2": 0.0,
        "min": None,
        "max": None,
    }


def _update_latent_stats(stats: dict, batch: torch.Tensor) -> None:
    if batch.numel() == 0:
        return
    values = batch.detach()
    if values.dtype != torch.float32:
        values = values.float()
    batch_min = float(values.min().item())
    batch_max = float(values.max().item())
    batch_mean = float(values.mean().item())
    batch_var = float(values.var(unbiased=False).item())
    batch_count = int(values.numel())
    batch_m2 = batch_var * batch_count

    stats["min"] = batch_min if stats["min"] is None else float(min(stats["min"], batch_min))
    stats["max"] = batch_max if stats["max"] is None else float(max(stats["max"], batch_max))

    count = int(stats["count"])
    mean = float(stats["mean"])
    m2 = float(stats["m2"])
    if count == 0:
        stats["count"] = batch_count
        stats["mean"] = batch_mean
        stats["m2"] = batch_m2
        return

    total = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * (batch_count / total)
    m2 = m2 + batch_m2 + delta * delta * (count * batch_count / total)

    stats["count"] = total
    stats["mean"] = mean
    stats["m2"] = m2


def _finalize_latent_stats(stats: dict) -> dict:
    count = int(stats["count"])
    if count <= 1:
        std = 0.0
    else:
        std = float(np.sqrt(float(stats["m2"]) / (count - 1)))
    return {
        "count": count,
        "mean": float(stats["mean"]),
        "std": std,
        "min": stats["min"],
        "max": stats["max"],
    }


class _LatentPrepDataset(Dataset):
    def __init__(
        self,
        entries: list[dict],
        *,
        limit: Optional[int] = None,
        decode_backend: str = "auto",
    ) -> None:
        self.entries = entries[:limit] if limit is not None else entries
        self.decode_backend = decode_backend

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        md5 = entry.get("md5", "")
        path = entry.get("img", "")
        start = time.perf_counter()
        try:
            x = _load_image_tensor(path, backend=self.decode_backend)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {"md5": md5, "x": x, "error": None, "decode_ms": elapsed_ms}
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return {"md5": md5, "x": None, "error": str(e), "decode_ms": elapsed_ms}


def _collate_items(batch: list[dict]) -> list[dict]:
    return batch


def _load_image_tensor(path: str, *, backend: str) -> torch.Tensor:
    if backend == "pil":
        return load_image_tensor(path)
    if backend == "torchvision":
        return _load_image_tensor_torchvision(path)
    if backend == "auto":
        try:
            return _load_image_tensor_torchvision(path)
        except Exception:
            return load_image_tensor(path)
    raise ValueError(f"Unknown decode backend: {backend}")


def _load_image_tensor_torchvision(path: str) -> torch.Tensor:
    try:
        from torchvision.io import ImageReadMode, read_image
    except Exception as exc:
        raise RuntimeError("torchvision is required for decode_backend=torchvision.") from exc
    x = read_image(path, mode=ImageReadMode.RGB)
    if x.shape[-2:] != (512, 512):
        raise RuntimeError(f"Unexpected image size: {tuple(x.shape[-2:])}")
    x = x.to(dtype=torch.float32) / 255.0
    return x * 2.0 - 1.0


@dataclass
class _SaveTask:
    md5: str
    out_path: Path
    latent: torch.Tensor


class _SaveStats:
    def __init__(self) -> None:
        self.saved = 0
        self.errors = 0
        self.save_ms = 0.0
        self.save_items = 0
        self.lock = threading.Lock()

    def add_saved(self, count: int, elapsed_ms: float) -> None:
        with self.lock:
            self.saved += count
            self.save_ms += elapsed_ms
            self.save_items += count

    def add_error(self, count: int = 1) -> None:
        with self.lock:
            self.errors += count

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "saved": self.saved,
                "errors": self.errors,
                "save_ms": self.save_ms,
                "save_items": self.save_items,
            }


class _ShardWriter:
    def __init__(
        self,
        *,
        shard_dir: Path,
        shard_size: int,
        index_path: Path,
        meta_common: dict,
        start_shard_id: int,
    ) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive.")
        self.shard_dir = shard_dir
        self.shard_size = shard_size
        self.index_path = index_path
        self.meta_common = meta_common
        self.current_md5s: list[str] = []
        self.current_latents: list[torch.Tensor] = []
        self.current_count = 0
        self.shard_id = start_shard_id
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp_index_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        if self.index_path.exists():
            self.tmp_index_path.write_text(self.index_path.read_text(encoding="utf-8"), encoding="utf-8")
            self.index_fp = self.tmp_index_path.open("a", encoding="utf-8")
        else:
            self.index_fp = self.tmp_index_path.open("w", encoding="utf-8")

    def add(self, md5: str, latent: torch.Tensor) -> Optional[tuple[Path, list[str]]]:
        self.current_md5s.append(md5)
        self.current_latents.append(latent)
        self.current_count += 1
        if self.current_count >= self.shard_size:
            return self.flush()
        return None

    def flush(self) -> Optional[tuple[Path, list[str]]]:
        if self.current_count == 0:
            return None
        shard_name = f"shard_{self.shard_id:06d}.pt"
        shard_path = self.shard_dir / shard_name
        latents = torch.stack(self.current_latents, dim=0).contiguous()
        payload = {
            "format_version": 3,
            "meta_common": self.meta_common,
            "latents": latents,
        }
        tmp_path = shard_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(shard_path)
        for idx, md5 in enumerate(self.current_md5s):
            line = {"md5": md5, "shard": shard_name, "idx": idx}
            self.index_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
        self.index_fp.flush()
        md5s = list(self.current_md5s)
        self.current_md5s = []
        self.current_latents = []
        self.current_count = 0
        self.shard_id += 1
        return shard_path, md5s

    def close(self) -> None:
        self.flush()
        self.index_fp.close()
        self.tmp_index_path.replace(self.index_path)


def _main_impl(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config/train.yaml")
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--latent-dtype", default='fp16', choices=("fp16", "bf16"))
    ap.add_argument("--autocast-dtype", default='fp16', choices=("fp16", "bf16"))
    ap.add_argument("--queue-size", type=int, default=64)
    ap.add_argument("--writer-threads", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=4096)
    ap.add_argument("--stats-every-sec", type=float, default=5.0)
    ap.add_argument("--decode-backend", default="auto", choices=("auto", "pil", "torchvision"))
    argv = argv if argv is not None else sys.argv[1:]
    provided_dests = _provided_cli_dests(ap, argv)
    args = ap.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config)
    options = _resolve_prepare_options(cfg, args, provided_dests)
    if cfg.mode != "latent":
        raise RuntimeError("prepare_latents requires mode=latent in config.")
    if not cfg.vae_pretrained:
        raise RuntimeError("vae_pretrained is required to prepare latents.")

    root = Path(cfg.data_root)
    cache_dir = root / (cfg.latent_cache_dir or ".cache/latents")
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = cache_dir / "shards"
    failed_path = root / "failed_latents.txt"
    run_dir = os.environ.get("WEBUI_RUN_DIR")
    metrics_dir = Path(run_dir) / "metrics" if run_dir else Path(cfg.out_dir) / "metrics"
    metrics_path = metrics_dir / "latent_prepare.jsonl"
    sinks = [JsonlFileSink(metrics_path), StdoutJsonSink()]
    event_bus = EventBus(sinks)

    code_version = build_run_metadata().get("git_commit")

    dcfg = DataConfig(
        root=str(cfg.data_root),
        image_dir=str(cfg.image_dir),
        meta_dir=str(cfg.meta_dir),
        tags_dir=str(cfg.tags_dir),
        caption_field=str(cfg.caption_field),
        text_field=str(cfg.text_field),
        text_fields=list(cfg.text_fields),
        images_only=bool(cfg.images_only),
        use_text_conditioning=bool(cfg.use_text_conditioning),
        min_tag_count=int(cfg.min_tag_count),
        require_512=bool(cfg.require_512),
        val_ratio=float(cfg.val_ratio),
        seed=int(cfg.seed),
        cache_dir=str(cfg.cache_dir),
        failed_list=str(cfg.failed_list),
    )
    train_entries, _ = build_or_load_index(dcfg)
    if options.limit is not None:
        checked_entries = train_entries[: int(options.limit)]
    else:
        checked_entries = train_entries
    if not checked_entries:
        raise RuntimeError(
            "No train entries found for latent preparation. "
            f"Check data_root={cfg.data_root!r}, image_dir={cfg.image_dir!r}, "
            f"meta_dir={cfg.meta_dir!r}, metadata.jsonl, images_only={cfg.images_only}, "
            f"and min_tag_count={cfg.min_tag_count}."
        )

    if options.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = options.device
    device = torch.device(device_str)
    dtype = _latent_dtype(options.latent_dtype)
    autocast_dtype = _latent_dtype(options.autocast_dtype)
    vae = VAEWrapper(
        pretrained=str(cfg.vae_pretrained),
        freeze=True,
        scaling_factor=float(cfg.vae_scaling_factor),
        device=device,
        dtype=dtype,
    )

    dataset = _LatentPrepDataset(
        train_entries,
        limit=options.limit,
        decode_backend=str(options.decode_backend),
    )
    dl = DataLoader(
        dataset,
        batch_size=int(options.batch_size),
        num_workers=int(options.num_workers),
        pin_memory=bool(options.pin_memory),
        prefetch_factor=int(options.prefetch_factor) if int(options.num_workers) > 0 else None,
        shuffle=False,
        collate_fn=_collate_items,
    )

    if options.queue_size <= 0:
        raise RuntimeError("--queue-size must be positive.")
    if options.writer_threads <= 0:
        raise RuntimeError("--writer-threads must be positive.")
    if options.shard_size < 0:
        raise RuntimeError("--shard-size must be >= 0.")
    if options.stats_every_sec <= 0:
        raise RuntimeError("--stats-every-sec must be positive.")
    if options.shard_size > 0 and options.writer_threads != 1:
        raise RuntimeError("Sharded mode requires --writer-threads=1.")

    total = len(dataset)
    skipped = 0
    stats = _init_latent_stats()
    start = time.perf_counter()
    last_log = start
    save_stats = _SaveStats()
    timing = {
        "decode_ms": 0.0,
        "h2d_ms": 0.0,
        "encode_ms": 0.0,
        "cpu_copy_ms": 0.0,
        "queue_wait_ms": 0.0,
        "total_ms": 0.0,
        "items": 0,
    }

    meta_common = {
        "format_version": 3,
        "vae_id": str(cfg.vae_pretrained),
        "vae_pretrained": str(cfg.vae_pretrained),
        "scaling_factor": float(cfg.vae_scaling_factor),
        "latent_shape": [int(cfg.latent_channels),
                         int(cfg.image_size) // int(cfg.latent_downsample_factor),
                         int(cfg.image_size) // int(cfg.latent_downsample_factor)],
        "dtype": str(options.latent_dtype),
        "layout": "contiguous",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code_version": code_version,
    }

    shard_writer: Optional[_ShardWriter] = None
    existing_md5s: set[str] = set()
    if options.shard_size > 0:
        shard_dir.mkdir(parents=True, exist_ok=True)
        index_path = Path(str(cfg.latent_cache_index))
        if not index_path.is_absolute():
            index_path = cache_dir / index_path
        cache_mismatch_reason = None
        if not options.overwrite:
            cache_mismatch_reason = _sharded_cache_mismatch_reason(
                index_path=index_path,
                shard_dir=shard_dir,
                expected_meta=meta_common,
            )
        if cache_mismatch_reason is not None:
            event_bus.emit({
                "type": "status",
                "status": "rebuild",
                "reason": f"latent cache metadata mismatch: {cache_mismatch_reason}",
            })
            if index_path.exists():
                index_path.unlink()
            for shard_path in shard_dir.glob("shard_*.pt"):
                shard_path.unlink()
        if options.overwrite:
            if index_path.exists():
                index_path.unlink()
            for shard_path in shard_dir.glob("shard_*.pt"):
                shard_path.unlink()
        if index_path.exists() and not options.overwrite:
            for line in index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                md5 = obj.get("md5")
                if isinstance(md5, str):
                    existing_md5s.add(md5)
        existing_shards = sorted(shard_dir.glob("shard_*.pt"))
        start_shard_id = 0
        if existing_shards:
            last_name = existing_shards[-1].stem
            try:
                start_shard_id = int(last_name.split("_")[-1]) + 1
            except ValueError:
                start_shard_id = len(existing_shards)
        shard_writer = _ShardWriter(
            shard_dir=shard_dir,
            shard_size=int(options.shard_size),
            index_path=index_path,
            meta_common=meta_common,
            start_shard_id=start_shard_id,
        )

    task_queue: queue.Queue[Optional[_SaveTask]] = queue.Queue(maxsize=int(options.queue_size))
    queue_wait_ms = {"value": 0.0}

    error_examples: list[dict[str, str]] = []

    def _record_error(md5: str, stage: str, message: object) -> None:
        if len(error_examples) < 5:
            error_examples.append({"md5": str(md5), "stage": str(stage), "error": str(message)})

    def _enqueue_task(task: _SaveTask) -> None:
        start_wait = time.perf_counter()
        task_queue.put(task)
        queue_wait_ms["value"] += (time.perf_counter() - start_wait) * 1000.0

    def _writer_loop() -> None:
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break
            try:
                start_save = time.perf_counter()
                _save_latent_cpu(
                    task.latent,
                    task.out_path,
                    cfg,
                    dtype,
                    str(options.latent_dtype),
                    code_version,
                )
                elapsed_ms = (time.perf_counter() - start_save) * 1000.0
                save_stats.add_saved(1, elapsed_ms)
            except Exception as exc:
                save_stats.add_error()
                _record_error(str(task.md5), "save", exc)
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{task.md5}\tsave\t{exc}\n")
            finally:
                task_queue.task_done()

    def _writer_loop_sharded() -> None:
        if shard_writer is None:
            return
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                break
            try:
                start_save = time.perf_counter()
                _validate_latent_tensor(task.latent, cfg)
                result = shard_writer.add(task.md5, task.latent)
                if result is not None:
                    elapsed_ms = (time.perf_counter() - start_save) * 1000.0
                    _, md5s = result
                    save_stats.add_saved(len(md5s), elapsed_ms)
            except Exception as exc:
                save_stats.add_error()
                _record_error(str(task.md5), "save", exc)
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{task.md5}\tsave\t{exc}\n")
            finally:
                task_queue.task_done()

    writer_threads = []
    for _ in range(int(options.writer_threads)):
        target = _writer_loop_sharded if options.shard_size > 0 else _writer_loop
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        writer_threads.append(thread)

    event_bus.emit({"type": "status", "status": "start", "total": total})

    webui_mode = os.environ.get("WEBUI") == "1"
    errors = 0
    last_save_snapshot = save_stats.snapshot()
    for batch in tqdm(dl, desc="prepare latents", disable=webui_mode):
        batch_start = time.perf_counter()
        batch_items = []
        decode_ms = 0.0
        for item in batch:
            decode_ms += float(item.get("decode_ms") or 0.0)
            if item["error"] is not None:
                errors += 1
                _record_error(str(item["md5"]), "load", item["error"])
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{item['md5']}\tload\t{item['error']}\n")
                continue
            md5 = item["md5"]
            if options.shard_size > 0:
                if md5 in existing_md5s and not options.overwrite:
                    skipped += 1
                    continue
                out_path = cache_dir
            else:
                out_path = latent_cache_path(cache_dir, md5)
                if out_path.exists() and not options.overwrite:
                    skipped += 1
                    continue
            batch_items.append((md5, out_path, item["x"]))

        if not batch_items:
            continue

        md5s, paths, xs = zip(*batch_items)
        h2d_start = time.perf_counter()
        x = torch.stack(xs, dim=0).to(device=device, dtype=dtype, non_blocking=True)
        h2d_ms = (time.perf_counter() - h2d_start) * 1000.0

        encode_start = time.perf_counter()
        try:
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast("cuda", dtype=autocast_dtype):
                        z_batch = vae.encode(x)
                else:
                    z_batch = vae.encode(x)
            encode_ms = (time.perf_counter() - encode_start) * 1000.0
            _update_latent_stats(stats, z_batch)
        except RuntimeError as e:
            if is_torch_oom_error(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("OOM during VAE encode; reduce --batch-size or use fewer workers.") from e
            for md5, path, x_single in batch_items:
                try:
                    with torch.inference_mode():
                        if device.type == "cuda":
                            with torch.autocast("cuda", dtype=autocast_dtype):
                                z_single = vae.encode(x_single.unsqueeze(0).to(device=device, dtype=dtype)).squeeze(0)
                        else:
                            z_single = vae.encode(x_single.unsqueeze(0).to(device=device, dtype=dtype)).squeeze(0)
                    _update_latent_stats(stats, z_single.unsqueeze(0))
                    z_cpu = z_single.to(dtype=dtype, device="cpu")
                    _enqueue_task(_SaveTask(md5=md5, out_path=Path(path), latent=z_cpu))
                except Exception as inner:
                    errors += 1
                    _record_error(str(md5), "encode", inner)
                    with open(failed_path, "a", encoding="utf-8") as f:
                        f.write(f"{md5}\tencode\t{inner}\n")
            continue

        cpu_copy_start = time.perf_counter()
        z_cpu = z_batch.to(device="cpu", dtype=dtype)
        cpu_copy_ms = (time.perf_counter() - cpu_copy_start) * 1000.0

        for md5, out_path, z in zip(md5s, paths, z_cpu):
            _enqueue_task(_SaveTask(md5=md5, out_path=Path(out_path), latent=z))

        batch_total_ms = (time.perf_counter() - batch_start) * 1000.0
        timing["decode_ms"] += decode_ms
        timing["h2d_ms"] += h2d_ms
        timing["encode_ms"] += encode_ms
        timing["cpu_copy_ms"] += cpu_copy_ms
        timing["total_ms"] += batch_total_ms
        timing["queue_wait_ms"] += queue_wait_ms["value"]
        queue_wait_ms["value"] = 0.0
        timing["items"] += len(batch_items)

        now = time.perf_counter()
        if now - last_log >= float(options.stats_every_sec):
            snapshot = save_stats.snapshot()
            save_delta_ms = snapshot["save_ms"] - last_save_snapshot["save_ms"]
            save_delta_items = snapshot["save_items"] - last_save_snapshot["save_items"]
            processed = snapshot["saved"] + skipped + errors + snapshot["errors"]
            elapsed = max(now - start, 1e-9)
            rate = (snapshot["saved"] + skipped) / elapsed
            avg_decode = timing["decode_ms"] / max(timing["items"], 1)
            avg_h2d = timing["h2d_ms"] / max(timing["items"], 1)
            avg_encode = timing["encode_ms"] / max(timing["items"], 1)
            avg_cpu = timing["cpu_copy_ms"] / max(timing["items"], 1)
            avg_total = timing["total_ms"] / max(timing["items"], 1)
            avg_queue_wait = timing["queue_wait_ms"] / max(timing["items"], 1)
            avg_save = save_delta_ms / max(save_delta_items, 1)
            event_bus.emit({
                "type": "metric",
                "processed": processed,
                "saved": snapshot["saved"],
                "skipped": skipped,
                "errors": errors + snapshot["errors"],
                "items_per_sec": rate,
                "imgs_per_sec": rate,
                "decode_ms": avg_decode,
                "h2d_ms": avg_h2d,
                "encode_ms": avg_encode,
                "cpu_copy_ms": avg_cpu,
                "save_ms": avg_save,
                "queue_wait_ms": avg_queue_wait,
                "total_ms": avg_total,
                "disk_write_queue_len": task_queue.qsize(),
                "max_steps": total,
            })
            timing = {
                "decode_ms": 0.0,
                "h2d_ms": 0.0,
                "encode_ms": 0.0,
                "cpu_copy_ms": 0.0,
                "queue_wait_ms": 0.0,
                "total_ms": 0.0,
                "items": 0,
            }
            last_save_snapshot = snapshot
            last_log = now

    for _ in writer_threads:
        task_queue.put(None)
    task_queue.join()
    for thread in writer_threads:
        thread.join()
    if shard_writer is not None:
        start_flush = time.perf_counter()
        result = shard_writer.flush()
        shard_writer.close()
        if result is not None:
            elapsed_ms = (time.perf_counter() - start_flush) * 1000.0
            _, md5s = result
            save_stats.add_saved(len(md5s), elapsed_ms)

    total_time = time.perf_counter() - start
    latent_stats = _finalize_latent_stats(stats)
    snapshot = save_stats.snapshot()
    event_bus.emit({
        "type": "status",
        "status": "done",
        "saved": snapshot["saved"],
        "skipped": skipped,
        "errors": errors + snapshot["errors"],
        "elapsed_sec": total_time,
        "latent_stats": latent_stats,
        "error_examples": error_examples,
    })


def _save_latent_cpu(
    z: torch.Tensor,
    out_path: Path,
    cfg: TrainConfig,
    dtype: torch.dtype,
    dtype_name: str,
    code_version: Optional[str],
) -> None:
    _validate_latent_tensor(z, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format_version": 1,
        "vae_id": str(cfg.vae_pretrained),
        "vae_pretrained": str(cfg.vae_pretrained),
        "scaling_factor": float(cfg.vae_scaling_factor),
        "latent_shape": list(z.shape),
        "dtype": dtype_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code_version": code_version,
    }
    payload = {
        "latent": z.to(dtype=dtype, device="cpu"),
        "meta": meta,
    }
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(out_path)


def _validate_latent_tensor(z: torch.Tensor, cfg: TrainConfig) -> None:
    if z.dim() != 3:
        raise RuntimeError(f"latent must be 3D, got {tuple(z.shape)}")
    if not torch.isfinite(z).all():
        raise RuntimeError("latent has NaN/Inf values")
    if z.shape[0] != int(cfg.latent_channels):
        raise RuntimeError(f"latent_channels mismatch: {z.shape[0]} != {cfg.latent_channels}")
    h = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    w = int(cfg.image_size) // int(cfg.latent_downsample_factor)
    if z.shape[-2:] != (h, w):
        raise RuntimeError(f"latent spatial mismatch: {tuple(z.shape[-2:])} != {(h, w)}")


def main(argv: Optional[list[str]] = None) -> None:
    try:
        _main_impl(argv)
    except Exception as exc:
        if is_torch_oom_error(exc):
            print_torch_oom(exc, context="latent preparation")
            raise SystemExit(2) from None
        raise


if __name__ == "__main__":
    main()
