# Configuration

## Model Families

New presets should use the nested canonical shape:

```yaml
model:
  family: mmdit | pixart_sigma | var
  variant: <family_variant>
  architecture: {}
```

`mmdit` remains the default for legacy configs that do not declare `model.family`.
See [Model Families](model_families.md) and [Checkpoint Metadata v2](checkpoint_metadata_v2.md) for the family matrix and metadata format.

This project uses KDL target configs plus reusable presets for architecture, dataset, cache, training and sampling settings.

---

## Scoped preset inheritance

Target configs now attach reusable presets to the section that consumes them. Use a short preset name inside the section instead of a top-level preset path:

```kdl
config target="train" version=2 {
  model {
    use "mmdit_576"
    checkpoint null
  }

  training {
    use "bf16_adamw"
    batch_size 1
    grad_accum_steps 4
    max_steps 100000
  }

  data {
    use "latent_cache_576"
    dataset_path "data/dataset"
  }

  output {
    dir "runs/train"
  }
}
```

A scoped `use` resolves to `configs/presets/<section>/<name>.kdl`. The `sampling` section maps to sampler presets, so both `sampler { use "rf_heun" }` and CLI overrides such as `--set sampling.use=rf_heun` use `configs/presets/sampler/rf_heun.kdl`.

Legacy top-level path inheritance remains supported for older configs, but new configs and templates should prefer scoped preset names. CLI overrides should use `section.use=name`; `section.preset=name` remains accepted for compatibility.

Reusable examples live under:

```text
configs/templates/
  train_576_bf16.kdl
  train_1024_bf16_text.kdl
  train_debug.kdl
  sample_txt2img.kdl
  cache_prepare_576.kdl
  presets/
    model.kdl
    training.kdl
    data.kdl
    text.kdl
```

## Minimal latent MMDiT RF config

```yaml
architecture: mmdit_rf
mode: latent

image_size: 512
latent_channels: 4
latent_downsample_factor: 8
latent_patch_size: 2

objective: rectified_flow
prediction_type: flow_velocity
```

---

## Dataset config

```yaml
data_root: ./data/dataset
image_dir: images
meta_dir: ""
tags_dir: ""

text_field: prompt
text_fields: [prompt, caption, text]

require_512: true
min_tag_count: 0
```

Notes:

- `text_field` is the preferred primary field.
- `text_fields` defines fallback fields.
- `caption_field` may exist for older configs, but prompt-first configs should prefer `text_field/text_fields`.

---

## VAE config

Real latent training requires a VAE compatible with `diffusers.AutoencoderKL.from_pretrained`.

```yaml
vae:
  pretrained: ./vae_sd_mse
  freeze: true
  scaling_factor: 0.18215
```

`pretrained` may be:

- a local diffusers VAE directory;
- a HuggingFace model name.

If the model is private or gated:

```bash
export HF_TOKEN=...
```

---

## Text encoder config

Example CLIP + T5 setup:

```yaml
text:
  enabled: true
  encoders:
    - name: clip_l
      model_name: openai/clip-vit-large-patch14
      max_length: 77
      trainable: false
      cache: true

    - name: t5
      model_name: google/t5-v1_1-base
      max_length: 256
      trainable: false
      cache: true

  text_dim: 1024
  pooled_dim: 1024
  empty_prompt_cache: true
```

For real training, text encoders are normally frozen and cached.

---

## Cache config

```yaml
cache:
  latent_cache: true
  text_cache: true
  auto_prepare: true
  validate_on_start: true
  strict: true
  rebuild_if_stale: false
  sharded: true
  dtype: bf16
```

Recommended behavior:

- `text_cache: true` for real training;
- `latent_cache: true` for real latent training;
- `validate_on_start: true` to fail early on stale or incompatible cache;
- `strict: true` for production training;
- `rebuild_if_stale: false` to avoid accidental expensive rebuilds.

---

## Debug-only on-the-fly text encoding

For real training, keep text cache enabled.

Debug-only mode:

```yaml
text_cache: false
allow_on_the_fly_text: true
```

This is useful for small smoke tests, not large training runs.

---

## Training config example

```yaml
training:
  batch_size: 1
  grad_accum_steps: 32
  lr: 0.0001
  max_steps: 100000
  mixed_precision: bf16
  gradient_clip_norm: 1.0

model:
  gradient_checkpointing: true
```

For CUDA OOM, reduce `batch_size`, increase `grad_accum_steps`, and enable `gradient_checkpointing`.

---

## Config validation

Dry-run a config:

```bash
python -m train.cli --config configs/train.kdl --dry-run
```

Dry-run a profile:

```bash
python -m train.cli --dry-run --set training.use=single_gpu_debug
```

---

## Config snapshots

Each run writes resolved config files to the run directory:

```text
runs/
  <run_name>/
    config.yaml
```

Use these files to reproduce a previous run.
