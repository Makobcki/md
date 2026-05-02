from __future__ import annotations

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


def _fit_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


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
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer, CLIPTextModel, T5EncoderModel
        except ImportError as exc:
            raise RuntimeError("FrozenTextEncoderBundle requires transformers to be installed.") from exc

        text_cfg = cfg.get("text", cfg)
        self.text_dim = int(text_cfg.get("text_dim", cfg.get("text_dim", 1024)))
        self.pooled_dim = int(text_cfg.get("pooled_dim", cfg.get("pooled_dim", self.text_dim)))
        specs = [
            FrozenTextEncoderSpec(
                name=str(item["name"]),
                model_name=str(item["model_name"]),
                max_length=int(item.get("max_length", 77)),
                trainable=bool(item.get("trainable", False)),
                cache=bool(item.get("cache", True)),
            )
            for item in text_cfg.get("encoders", [])
        ]
        if not specs:
            raise ValueError("At least one text encoder spec is required.")
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

    @torch.no_grad()
    def forward(self, prompts: str | Iterable[str]) -> TextConditioning:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        token_chunks: list[torch.Tensor] = []
        mask_chunks: list[torch.Tensor] = []
        pooled_chunks: list[torch.Tensor] = []
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
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
            mask_chunks.append(encoded["attention_mask"].to(torch.bool))
            pooled_chunks.append(_fit_dim(pooled.to(dtype), self.pooled_dim))
        tokens = torch.cat(token_chunks, dim=1)
        mask = torch.cat(mask_chunks, dim=1)
        pooled = torch.stack(pooled_chunks, dim=0).mean(dim=0)
        is_uncond = torch.tensor([not p.strip() for p in prompts], device=device, dtype=torch.bool)
        return TextConditioning(tokens=tokens, mask=mask, pooled=pooled, is_uncond=is_uncond)

    def metadata(self) -> dict[str, Any]:
        return {
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
