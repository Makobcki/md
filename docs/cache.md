# Cache preparation

Real training uses prepared text and latent caches.

Caches improve throughput and make training startup deterministic. They also catch dataset/config mismatches early when validation is enabled.

---

## Cache layout

Typical layout:

```text
data/dataset/
  .cache/
    text/
    latents/
```

---

## Text cache

Prepare text cache:

```bash
python -m scripts.prepare_text_cache --config configs/train.kdl
```

Expected output:

```text
data/dataset/.cache/text/
  metadata.json
  manifest.json
  index.jsonl
  empty_prompt.safetensors
  shards/
    text_00000.safetensors
    ...
```

Validate text cache:

```bash
python -m scripts.validate_cache --config configs/train.kdl --text
```

---

## Latent cache

Prepare latent cache:

```bash
python -m scripts.prepare_latents --config configs/train.kdl
```

Expected output:

```text
data/dataset/.cache/latents/
  metadata.json
  shard_index.jsonl
  shards/
    *.pt
    *.safetensors
```

Validate latent cache:

```bash
python -m scripts.validate_cache --config configs/train.kdl --latents
```

---

## Unified training cache

Prepare text cache, latent cache and manifest in one command:

```bash
python -m scripts.prepare_training_cache --config configs/train.kdl
```

Repair modes:

```bash
python -m scripts.prepare_training_cache --config configs/train.kdl --repair
python -m scripts.prepare_training_cache --config configs/train.kdl --repair --force
python -m scripts.prepare_training_cache --config configs/train.kdl --repair --rebuild
```

---

## Auto-prepare

Training can prepare cache automatically when enabled:

```yaml
cache:
  auto_prepare: true
```

For large datasets, explicit manual cache preparation is recommended:

```bash
python -m scripts.prepare_text_cache --config configs/train.kdl
python -m scripts.prepare_latents --config configs/train.kdl
python -m scripts.validate_cache --config configs/train.kdl
```

---

## Rebuild text cache

Rebuild text cache when changing:

- prompts;
- captions;
- tags;
- `text_field`;
- `text_fields`;
- tokenizer or text encoder config;
- dataset index logic.

Command:

```bash
rm -rf data/dataset/.cache/text
python -m scripts.prepare_text_cache --config configs/train.kdl
```

---

## Rebuild latent cache

Rebuild latent cache when changing:

- VAE;
- image size;
- latent downsample factor;
- latent patch size;
- latent dtype;
- dataset images;
- latent shape.

Command:

```bash
rm -rf data/dataset/.cache/latents
python -m scripts.prepare_latents --config configs/train.kdl
```

Alternative overwrite mode:

```bash
python -m scripts.prepare_latents --config configs/train.kdl --overwrite
```

---

## Validate all caches

```bash
python -m scripts.validate_cache --config configs/train.kdl
```

Use validation before long training runs.
