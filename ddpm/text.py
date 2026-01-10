from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)

PAD = 0
BOS = 1
EOS = 2
UNK = 3


def normalize_text(s: str) -> str:
    s = s.lower()
    toks = _TOKEN_RE.findall(s)
    return " ".join(toks)


@dataclass(frozen=True)
class Vocab:
    itos: list[str]   # id -> token
    stoi: dict[str, int]  # token -> id

    @property
    def size(self) -> int:
        return len(self.itos)

    @staticmethod
    def load(path: str | Path) -> "Vocab":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = obj["itos"]
        stoi = {t: i for i, t in enumerate(itos)}
        return Vocab(itos=itos, stoi=stoi)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")


class SimpleTokenizer:
    def __init__(self, vocab: Vocab, max_len: int) -> None:
        self.vocab = vocab
        self.max_len = int(max_len)

    def encode(self, text: str) -> torch.LongTensor:
        text = normalize_text(text)
        ids = [BOS]
        for tok in text.split():
            ids.append(self.vocab.stoi.get(tok, UNK))
        ids.append(EOS)

        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [PAD] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, texts: list[str]) -> torch.LongTensor:
        return torch.stack([self.encode(t) for t in texts], dim=0)

    def empty_tokens(self, batch: int) -> torch.LongTensor:
        # [BOS, EOS, PAD, PAD, ...]
        ids = [BOS, EOS] + [PAD] * (self.max_len - 2)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).repeat(batch, 1)


def build_vocab_from_texts(
    texts: Iterable[str],
    vocab_size: int = 50000,
    min_freq: int = 2,
) -> Vocab:
    cnt = Counter()
    for s in texts:
        s = normalize_text(s)
        if not s:
            continue
        cnt.update(s.split())

    # special tokens
    itos = ["<pad>", "<bos>", "<eos>", "<unk>"]
    for tok, c in cnt.most_common():
        if c < min_freq:
            break
        if tok in itos:
            continue
        itos.append(tok)
        if len(itos) >= vocab_size:
            break
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(itos=itos, stoi=stoi)


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        max_len: int,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_len = int(max_len)

        self.tok_emb = nn.Embedding(vocab_size, self.dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len, self.dim))

        layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=heads,
            dim_feedforward=self.dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.ln = nn.LayerNorm(self.dim)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        # tokens: [B, T]
        b, t = tokens.shape
        assert t == self.max_len, f"tokens len {t} != max_len {self.max_len}"
        x = self.tok_emb(tokens) + self.pos_emb[:, :t, :]
        key_padding = tokens.eq(PAD)  # [B, T]
        x = self.enc(x, src_key_padding_mask=key_padding)
        x = self.ln(x)
        return x  # [B, T, dim]
