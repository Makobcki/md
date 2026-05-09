from __future__ import annotations

import torch

from data_loader.token_cache import (
    MultiscaleTokenDataset,
    TokenCacheMetadata,
    build_synthetic_token_entries,
    load_token_cache_metadata,
    save_token_cache_metadata,
    validate_tokenizer_metadata,
)


def test_token_cache_metadata_roundtrip_and_validation(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    meta = TokenCacheMetadata(
        kind="vq",
        codebook_size=32,
        codebook_dim=8,
        scale_schedule=(1, 2, 4),
        max_token_length=21,
    )

    save_token_cache_metadata(path, meta)

    loaded = load_token_cache_metadata(path)
    assert loaded == meta
    validate_tokenizer_metadata(loaded, meta)


def test_synthetic_multiscale_token_batches_load() -> None:
    meta = TokenCacheMetadata(
        kind="synthetic",
        codebook_size=16,
        codebook_dim=8,
        scale_schedule=(1, 2),
        max_token_length=5,
    )
    dataset = MultiscaleTokenDataset(build_synthetic_token_entries(meta, count=3, seed=123), meta)

    batch = dataset[0]

    assert [tokens.shape for tokens in batch["tokens"]] == [torch.Size([1]), torch.Size([4])]
    root_token = int(batch["tokens"][0][0].item())
    expected_next = (root_token + 997 + torch.arange(4, dtype=torch.long)) % 16
    assert torch.equal(batch["tokens"][1], expected_next)
    assert batch["metadata"].codebook_size == 16
