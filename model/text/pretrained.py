from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .conditioning import TextConditioning


@dataclass(frozen=True)
class FrozenTextEncoderSpec:
    name: str
    model_name: str
    max_length: int
    trainable: bool = False
    cache: bool = True


def _encoder_type_id(name: str) -> int:
    lowered = str(name).lower()
    if lowered.startswith("clip"):
        return 0
    if "t5" in lowered:
        return 1
    return 2


def _fit_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _as_prompt_list(prompts: str | Iterable[str]) -> list[str]:
    if isinstance(prompts, str):
        return [prompts]
    return [str(p) for p in prompts]


def _from_pretrained_with_local_fallback(factory: Any, model_name: str, **kwargs: Any) -> Any:
    try:
        return factory.from_pretrained(model_name, **kwargs)
    except TypeError:
        if not kwargs:
            raise
        return factory.from_pretrained(model_name)
    except (OSError, RuntimeError) as exc:
        try:
            return factory.from_pretrained(model_name, local_files_only=True, **kwargs)
        except Exception:
            raise exc


def _resolve_cached_model_path(model_name: str) -> str:
    if Path(model_name).exists():
        return model_name
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


class FrozenTextEncoderBundle(nn.Module):
    """Frozen text-conditioning bundle used by train/sample code.

    The normal backend loads frozen CLIP/T5-like HuggingFace encoders. The
    ``backend='fake'`` mode is intentionally deterministic and has no external
    dependencies; it is used by unit tests and latent-only smoke sampling.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        backend: str | None = None,
    ) -> None:
        super().__init__()
        text_cfg = cfg.get("text", cfg) if isinstance(cfg, dict) else {}
        self.backend = str(backend or text_cfg.get("backend", text_cfg.get("fake_or_cached", "real"))).lower()
        if self.backend in {"true", "1"}:
            self.backend = "fake"
        self.text_dim = int(text_cfg.get("text_dim", cfg.get("text_dim", 1024) if isinstance(cfg, dict) else 1024))
        self.pooled_dim = int(text_cfg.get("pooled_dim", cfg.get("pooled_dim", self.text_dim) if isinstance(cfg, dict) else self.text_dim))
        self._device = torch.device(device or "cpu")
        self._dtype = dtype

        raw_specs = text_cfg.get("encoders", []) if isinstance(text_cfg, dict) else []
        if self.backend == "fake":
            fake_max_length = int(text_cfg.get("fake_max_length", text_cfg.get("max_length", 8)))
            self.specs = [
                FrozenTextEncoderSpec(
                    name=str(item.get("name", "fake")),
                    model_name=str(item.get("model_name", "fake")),
                    max_length=int(item.get("max_length", fake_max_length)),
                    trainable=False,
                    cache=True,
                )
                for item in raw_specs
            ] or [FrozenTextEncoderSpec("fake", "fake", fake_max_length, trainable=False, cache=True)]
            # A buffer gives the module a reliable device/dtype anchor without parameters.
            self.register_buffer("_fake_anchor", torch.empty((), device=self._device, dtype=self._dtype), persistent=False)
            return

        self.backend = "real"
        try:
            from transformers import AutoModel, AutoTokenizer, CLIPTextModel, T5EncoderModel
        except ImportError as exc:
            raise RuntimeError("FrozenTextEncoderBundle requires transformers to be installed, or text.backend='fake'.") from exc

        specs = [
            FrozenTextEncoderSpec(
                name=str(item["name"]),
                model_name=str(item["model_name"]),
                max_length=int(item.get("max_length", 77)),
                trainable=bool(item.get("trainable", False)),
                cache=bool(item.get("cache", True)),
            )
            for item in raw_specs
        ]
        if not specs:
            raise ValueError("At least one text encoder spec is required for the real text backend.")
        self.specs = specs
        self.tokenizers = {}
        self.encoders = nn.ModuleDict()
        for spec in specs:
            model_path = _resolve_cached_model_path(spec.model_name)
            self.tokenizers[spec.name] = _from_pretrained_with_local_fallback(AutoTokenizer, model_path)
            model_key = f"{spec.name} {spec.model_name}".lower()
            if spec.name.startswith("clip"):
                model_cls = CLIPTextModel
            elif "t5" in model_key:
                model_cls = T5EncoderModel
            else:
                model_cls = AutoModel
            model_kwargs = {"use_safetensors": False} if "t5" in model_key else {}
            encoder = _from_pretrained_with_local_fallback(model_cls, model_path, **model_kwargs)
            encoder.requires_grad_(bool(spec.trainable))
            encoder.eval()
            self.encoders[spec.name] = encoder
        self.to(device=device, dtype=dtype)

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "FrozenTextEncoderBundle":
        return cls(cfg, device=device, dtype=dtype)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._fake_anchor.device if hasattr(self, "_fake_anchor") else self._device

    @property
    def output_dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return self._fake_anchor.dtype if hasattr(self, "_fake_anchor") else self._dtype

    def _fake_token_values(self, prompt: str, length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length <= 0:
            return torch.empty(0, dim, device=device, dtype=dtype)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        values = torch.randn(length, dim, generator=gen, dtype=torch.float32)
        return values.to(device=device, dtype=dtype)

    @torch.no_grad()
    def _forward_fake(self, prompts: list[str]) -> TextConditioning:
        device = self.device
        dtype = self.output_dtype
        b = len(prompts)
        total_len = sum(int(spec.max_length) for spec in self.specs)
        tokens = torch.zeros(b, total_len, self.text_dim, device=device, dtype=dtype)
        mask = torch.zeros(b, total_len, device=device, dtype=torch.bool)
        token_types = torch.full((b, total_len), 2, device=device, dtype=torch.long)
        pooled_chunks: list[torch.Tensor] = []
        offset = 0
        for spec in self.specs:
            n = int(spec.max_length)
            chunk = torch.zeros(b, n, self.text_dim, device=device, dtype=dtype)
            chunk_mask = torch.zeros(b, n, device=device, dtype=torch.bool)
            for row, prompt in enumerate(prompts):
                stripped = prompt.strip()
                if stripped:
                    # Not a tokenizer: just deterministic, length-sensitive masking.
                    words = stripped.split()
                    used = min(max(len(words), 1), n)
                    chunk_mask[row, :used] = True
                    chunk[row, :used] = self._fake_token_values(
                        f"{spec.name}:{prompt}", used, self.text_dim, device=device, dtype=dtype
                    )
            tokens[:, offset : offset + n] = chunk
            mask[:, offset : offset + n] = chunk_mask
            token_types[:, offset : offset + n] = int(_encoder_type_id(spec.name))
            denom = chunk_mask.to(dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (chunk * chunk_mask.unsqueeze(-1).to(dtype)).sum(dim=1) / denom
            pooled_chunks.append(_fit_dim(pooled, self.pooled_dim))
            offset += n
        pooled = torch.stack(pooled_chunks, dim=0).mean(dim=0) if pooled_chunks else torch.zeros(b, self.pooled_dim, device=device, dtype=dtype)
        is_uncond = torch.tensor([not p.strip() for p in prompts], device=device, dtype=torch.bool)
        return TextConditioning(tokens=tokens, mask=mask, pooled=pooled, is_uncond=is_uncond, token_types=token_types)

    @torch.no_grad()
    def forward(self, prompts: str | Iterable[str]) -> TextConditioning:
        prompts = _as_prompt_list(prompts)
        if self.backend == "fake":
            return self._forward_fake(prompts)

        token_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        pooled_chunks: list[torch.Tensor] = []
        token_type_chunks: list[torch.Tensor] = []
        device = self.device
        dtype = self.output_dtype
        for spec in self.specs:
            tokenizer = self.tokenizers[spec.name]
            encoded = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=spec.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = self.encoders[spec.name](**encoded)
            hidden = getattr(out, "last_hidden_state")
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                attn = encoded.get("attention_mask", torch.ones(hidden.shape[:2], device=device)).to(hidden.dtype)
                pooled = (hidden * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(dim=1, keepdim=True).clamp_min(1.0)
            token_chunks.append(_fit_dim(hidden.to(dtype), self.text_dim))
            mask_chunk = encoded["attention_mask"].to(torch.bool)
            mask_chunks.append(mask_chunk)
            token_type_chunks.append(torch.full_like(mask_chunk, int(_encoder_type_id(spec.name)), dtype=torch.long))
            pooled_chunks.append(_fit_dim(pooled.to(dtype), self.pooled_dim))
        tokens = torch.cat(token_chunks, dim=1)
        mask = torch.cat(mask_chunks, dim=1)
        token_types = torch.cat(token_type_chunks, dim=1)
        pooled = torch.stack(pooled_chunks, dim=0).mean(dim=0)
        is_uncond = torch.tensor([not p.strip() for p in prompts], device=device, dtype=torch.bool)
        return TextConditioning(tokens=tokens, mask=mask, pooled=pooled, is_uncond=is_uncond, token_types=token_types)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "encoders": [
                {
                    "name": spec.name,
                    "model_name": spec.model_name,
                    "max_length": spec.max_length,
                    "trainable": spec.trainable,
                    "cache": spec.cache,
                }
                for spec in self.specs
            ],
            "text_dim": self.text_dim,
            "pooled_dim": self.pooled_dim,
        }
