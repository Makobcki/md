# MMDiT Rectified Flow image model

This repository trains and samples a latent MMDiT rectified-flow image model. The supported model line is **MMDiT RF-only**: latent space, frozen pretrained text encoders, joint text/image attention, flow-matching training, and Flow Euler/Heun sampling.

## Install

```bash
python -m pip install -e .
```

Install the project dependencies required by your environment, including PyTorch, torchvision, safetensors, transformers, sentencepiece, and the VAE/text encoder dependencies used by the configured checkpoints.

## Architecture

```text
image / source image / mask / control latent
        ↓
VAE encoder / latent cache
        ↓
latent z: [B, 4, 64, 64] for 512×512
        ↓
patchify, p=2
        ↓
image tokens: [B, 1024, D]
        ↓
MMDiT / Flux-like transformer
        ↑
frozen text encoders: CLIP/T5 through text cache
        ↓
unpatchify
        ↓
predicted flow velocity
        ↓
Flow sampler: Euler / Heun
        ↓
VAE decoder
        ↓
image
```

Implemented conditioning paths:

- `txt2img`: noisy target latent + text tokens;
- `img2img`: noisy target latent + source latent tokens + text tokens;
- `inpaint`: noisy target latent + masked source latent tokens + mask tokens + text tokens;
- `control_latents`: control token stream for ControlNet/IP-Adapter-style extensions.

## Dataset layout

The default profiles expect:

```text
data/dataset/pixso_512/
  images/
  metadata.jsonl or per-image metadata files
  .cache/
    text/
    latents/
```

`config/train.yaml` is the main full training profile. Smaller profiles are available for smoke, overfit, and development runs.

## Training

```bash
python -m train.cli --config config/train.yaml
```

Training does the full setup:

- builds or loads the dataset index;
- prepares the text cache when missing;
- prepares the latent cache when missing;
- fails clearly on stale or mismatched caches unless `cache.rebuild_if_stale: true`;
- trains MMDiT RF with rectified-flow objective;
- logs train/validation loss, including loss-by-timestep bins;
  Loss-by-timestep bins use the stable event keys `loss_t_bin_00_01` through `loss_t_bin_09_10`, matching `[0.0, 0.1)` through `[0.9, 1.0]`.
- saves checkpoints, config snapshots, metrics, and eval outputs.

Useful profiles:

```bash
python -m train.cli --profile smoke
python -m train.cli --profile overfit
python -m train.cli --profile dev
python -m train.cli --profile base
python -m train.cli --profile milestone_a
python -m train.cli --profile milestone_b
python -m train.cli --profile milestone_c
```

Equivalent explicit config form:

```bash
python -m train.cli --config config/train_smoke.yaml
python -m train.cli --config config/train_overfit.yaml
python -m train.cli --config config/train_dev.yaml
python -m train.cli --config config/train_base.yaml
python -m train.cli --config config/train_milestone_a.yaml
python -m train.cli --config config/train_milestone_b.yaml
python -m train.cli --config config/train_milestone_c.yaml
```

Dry run:

```bash
python -m train.cli --profile smoke --dry-run
python -m train.cli --profile overfit --dry-run
python -m train.cli --profile dev --dry-run
python -m train.cli --profile base --dry-run
```

## Sampling

Text-to-image:

```bash
python -m sample.cli \
  --ckpt ./runs/.../ckpt_final.pt \
  --prompt "1girl, blue hair, white dress" \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --shift 3.0 \
  --seed 42 \
  --out ./samples/out.png
```

Image-to-image:

```bash
python -m sample.cli \
  --ckpt ./runs/.../ckpt_final.pt \
  --prompt "same character, winter outfit" \
  --init-image ./input.png \
  --strength 0.55 \
  --sampler flow_heun \
  --steps 28 \
  --out ./samples/img2img.png
```

Inpainting:

```bash
python -m sample.cli \
  --ckpt ./runs/.../ckpt_final.pt \
  --task inpaint \
  --prompt "replace the background with a city at night" \
  --init-image ./input.png \
  --mask ./mask.png \
  --sampler flow_heun \
  --steps 28 \
  --out ./samples/inpaint.png
```

Supported samplers are `flow_euler` and `flow_heun`. The inpainting sampler preserves the unmasked source region at every flow step. Sample metadata is written beside the output image for reproducibility.

## Manual cache tools

Normally `train` prepares caches automatically. These scripts remain available for manual precomputation and debugging:

```bash
python -m scripts.prepare_text_cache --config config/train.yaml
python -m scripts.prepare_latents --config config/train.yaml
python -m scripts.prepare_training_cache --config config/train.yaml
```

## Tests

```bash
python -m pytest -q
```

## WebUI

WebUI is experimental and follows the MMDiT RF-only backend. Training and sampling CLIs are the source of truth.

## Production checklist

Run stages in order:

```bash
python -m train.cli --config config/train_overfit.yaml
python -m train.cli --config config/train_dev.yaml
python -m train.cli --config config/train_base.yaml
python -m train.cli --config config/train_milestone_a.yaml
python -m train.cli --config config/train_milestone_b.yaml
python -m train.cli --config config/train_milestone_c.yaml
```

Move from overfit to dev when loss falls sharply. Move from dev to full when there are no NaN/Inf failures, resume works, sampling creates images, and VRAM remains stable.

## Evaluation

```bash
python -m train.eval_cli --ckpt ./runs/.../checkpoints/latest.pt --out-dir ./runs/... --prompt-set core --fake-vae
md-eval --ckpt ./runs/.../checkpoints/latest.pt --out-dir ./runs/... --prompt-set core --fake-vae
```

The eval CLI can build fixed-seed grids and sweep CFG, step count, sampler, shift, and resolution.
