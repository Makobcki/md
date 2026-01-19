from __future__ import annotations

from typing import Any, Mapping, Tuple

from .tokenizer import BPETokenizer, TextConfig


def build_text_config_from_dict(cfg: Mapping[str, Any]) -> TextConfig:
    vocab_path = cfg.get("text_vocab_path")
    merges_path = cfg.get("text_merges_path")
    if not vocab_path or not merges_path:
        raise RuntimeError("text_vocab_path/text_merges_path are required for tokenizer.")
    return TextConfig(
        vocab_path=str(vocab_path),
        merges_path=str(merges_path),
        max_len=int(cfg.get("text_max_len", 128)),
        lowercase=True,
        strip_punct=True,
    )


def build_tokenizer(text_cfg: TextConfig) -> BPETokenizer:
    return BPETokenizer.from_files(
        vocab_path=text_cfg.vocab_path,
        merges_path=text_cfg.merges_path,
        cfg=text_cfg,
    )


def build_tokenizer_from_dict(cfg: Mapping[str, Any]) -> Tuple[BPETokenizer, TextConfig]:
    text_cfg = build_text_config_from_dict(cfg)
    return build_tokenizer(text_cfg), text_cfg
