# Troubleshooting

Common errors and fixes.

---

## `Text cache missing key`

The dataset index changed but the text cache is old.

Rebuild text cache:

```bash
rm -rf data/dataset/.cache/text
python -m scripts.prepare_text_cache --config config/train.yaml
```

---

## `text cache dataset_hash mismatch`

Prompts, captions, tags, `text_field` or `text_fields` changed.

Rebuild text cache:

```bash
rm -rf data/dataset/.cache/text
python -m scripts.prepare_text_cache --config config/train.yaml
```

---

## `Latent cache is stale`

The latent cache no longer matches the current dataset/config.

Common causes:

- changed VAE;
- changed image size;
- changed latent dtype;
- changed dataset images;
- changed latent shape;
- changed latent downsample factor.

Rebuild latent cache:

```bash
rm -rf data/dataset/.cache/latents
python -m scripts.prepare_latents --config config/train.yaml
```

Or overwrite:

```bash
python -m scripts.prepare_latents --config config/train.yaml --overwrite
```

---

## `latent shape mismatch`

The cached latent shape does not match the model/config.

Check:

```yaml
image_size: 512
latent_channels: 4
latent_downsample_factor: 8
latent_patch_size: 2
```

Then rebuild latent cache:

```bash
rm -rf data/dataset/.cache/latents
python -m scripts.prepare_latents --config config/train.yaml
```

---

## `Checkpoint incompatible: hidden_dim mismatch`

The checkpoint was created with a different architecture/config.

Fix:

- use the original config from the run directory;
- use a checkpoint trained with the current config;
- do not mix checkpoints across incompatible model sizes.

Relevant files:

```text
runs/.../
  config.yaml
  config_resolved.yaml
  config_snapshot.yaml
```

---

## `task=inpaint requires --mask`

Inpaint requires both an initial image and a mask.

Correct command:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task inpaint \
  --prompt "replace the background" \
  --init-image input.png \
  --mask mask.png \
  --out samples/inpaint.png
```

---

## `task=img2img requires --init-image`

Image-to-image requires an initial image.

Correct command:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task img2img \
  --prompt "same character, winter outfit" \
  --init-image input.png \
  --strength 0.55 \
  --out samples/img2img.png
```

---

## `text_cache=false is only allowed when allow_on_the_fly_text=true`

Real training should use text cache.

Recommended:

```yaml
cache:
  text_cache: true
```

Debug-only mode:

```yaml
text_cache: false
allow_on_the_fly_text: true
```

---

## CUDA OOM

Reduce memory usage:

```yaml
training:
  batch_size: 1
  grad_accum_steps: 32

model:
  gradient_checkpointing: true
```

Also use a smaller profile first:

```bash
python -m train.cli --profile smoke
python -m train.cli --profile overfit
python -m train.cli --profile dev
```

---

## WebUI cannot access file path

By default, WebUI restricts allowed paths.

Allow additional paths:

```bash
export WEBUI_ALLOWED_PATHS="/path/to/data:/path/to/samples"
```

Restart WebUI after changing the environment variable.

---

## HuggingFace model cannot be downloaded

Set a token:

```bash
export HF_TOKEN=...
```

Then rerun the command that downloads or loads the model.

---

## Cache validation before long training

Before expensive runs:

```bash
python -m train.cli --config config/train.yaml --dry-run
python -m scripts.validate_cache --config config/train.yaml
```

This catches most dataset/cache/config mismatches before training starts.
