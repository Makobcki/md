from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from .config import TextConfig

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def normalize_text(text: str, lowercase: bool = True, strip_punct: bool = True) -> str:
    text = text.strip()
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def bytes_to_unicode() -> Dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class BPETokenizer:
    def __init__(self, vocab: Dict[str, int], merges: List[Tuple[str, str]], cfg: TextConfig) -> None:
        self.vocab = vocab
        self.cfg = cfg
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache: Dict[str, Tuple[str, ...]] = {}

        self.pad_id = vocab.get(PAD_TOKEN, 0)
        self.bos_id = vocab.get(BOS_TOKEN, 1)
        self.eos_id = vocab.get(EOS_TOKEN, 2)
        self.unk_id = vocab.get(UNK_TOKEN, 3)

        self.bpe_ranks = {pair: idx for idx, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_path: str | Path, merges_path: str | Path, cfg: TextConfig) -> "BPETokenizer":
        vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        merges = cls._load_merges(merges_path)
        return cls(vocab=vocab, merges=merges, cfg=cfg)

    @staticmethod
    def _load_merges(path: str | Path) -> List[Tuple[str, str]]:
        merges: List[Tuple[str, str]] = []
        text = Path(path).read_text(encoding="utf-8")
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        return merges

    @staticmethod
    def _get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
        pairs = set()
        if not word:
            return pairs
        prev = word[0]
        for ch in word[1:]:
            pairs.add((prev, ch))
            prev = ch
        return pairs

    def _bpe(self, token: str) -> Tuple[str, ...]:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return (token,)
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        self.cache[token] = word
        return word

    def _tokenize_word(self, word: str, add_space: bool) -> List[str]:
        if add_space:
            word = " " + word
        encoded = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
        return list(self._bpe(encoded))

    def encode(self, text: str) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        text = normalize_text(text, self.cfg.lowercase, self.cfg.strip_punct)
        words = text.split(" ") if text else []
        tokens: List[str] = []
        for i, w in enumerate(words):
            if not w:
                continue
            tokens.extend(self._tokenize_word(w, add_space=(i != 0)))

        ids = [self.bos_id]
        for tok in tokens:
            ids.append(self.vocab.get(tok, self.unk_id))

        if len(ids) >= self.cfg.max_len:
            ids = ids[: self.cfg.max_len]
            ids[-1] = self.eos_id
        else:
            ids.append(self.eos_id)
            while len(ids) < self.cfg.max_len:
                ids.append(self.pad_id)

        attn_mask = [tok_id != self.pad_id for tok_id in ids]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.bool)

    def batch_encode(self, texts: Iterable[str]) -> torch.LongTensor:
        ids = [self.encode(t)[0] for t in texts]
        return torch.stack(ids, dim=0)

    def empty_tokens(self, batch: int) -> torch.LongTensor:
        ids = [self.bos_id, self.eos_id] + [self.pad_id] * (self.cfg.max_len - 2)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
