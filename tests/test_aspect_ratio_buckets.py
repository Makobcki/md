from __future__ import annotations

import pytest

from data_loader.buckets import (
    AspectBucketBatchSampler,
    AspectRatioBucket,
    assign_bucket,
    group_entries_by_bucket,
    parse_buckets,
    validate_buckets,
)


def test_bucket_validation_checks_latent_and_patch_divisibility() -> None:
    buckets = validate_buckets(
        [AspectRatioBucket(512, 512), AspectRatioBucket(640, 384), AspectRatioBucket(384, 640)],
        latent_downsample_factor=8,
        latent_patch_size=2,
    )
    assert [(b.width, b.height) for b in buckets] == [(512, 512), (640, 384), (384, 640)]

    with pytest.raises(ValueError, match="latent_downsample_factor"):
        validate_buckets([AspectRatioBucket(513, 512)])
    with pytest.raises(ValueError, match="latent_patch_size"):
        validate_buckets([AspectRatioBucket(520, 520)], latent_downsample_factor=8, latent_patch_size=4)


def test_parse_and_assign_bucket() -> None:
    buckets = parse_buckets(["512x512", {"width": 640, "height": 384}, (384, 640)])
    bucket = assign_bucket(800, 480, buckets)
    assert (bucket.width, bucket.height) == (640, 384)


def test_bucket_batch_sampler_keeps_same_bucket_per_batch() -> None:
    buckets = validate_buckets([AspectRatioBucket(512, 512), AspectRatioBucket(640, 384), AspectRatioBucket(384, 640)])
    entries = [
        {"width": 512, "height": 512},
        {"width": 500, "height": 500},
        {"width": 640, "height": 384},
        {"width": 1280, "height": 768},
        {"width": 384, "height": 640},
        {"width": 768, "height": 1280},
    ]
    groups = group_entries_by_bucket(entries, buckets)
    sampler = AspectBucketBatchSampler(entries, buckets, batch_size=2, shuffle=False, drop_last=False)

    assert all(len(batch) <= 2 for batch in sampler)
    for batch in sampler:
        containing = [key for key, indices in groups.items() if set(batch).issubset(indices)]
        assert len(containing) == 1


def test_bucket_validation_accepts_all_stage_d_patch_sizes() -> None:
    buckets = validate_buckets(
        [AspectRatioBucket(640, 384)],
        latent_downsample_factor=8,
        latent_patch_size=[2, 4, 8],
    )
    assert [(b.width, b.height) for b in buckets] == [(640, 384)]

    with pytest.raises(ValueError, match="all requested latent_patch_size"):
        validate_buckets(
            [AspectRatioBucket(672, 384)],
            latent_downsample_factor=8,
            latent_patch_size=[2, 4, 8],
        )
