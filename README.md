# MMDiT Rectified Flow image model

This repository trains and samples a latent MMDiT rectified-flow image model. The legacy U-Net, BPE, DDPM, DDIM, and DPM paths are no longer part of the supported workflow.

## Install

```bash
python -m pip install -e .
```

Install the project dependencies required by your environment, including PyTorch, torchvision, safetensors, and the text encoder/VAE dependencies used by the configured checkpoints.

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

- builds or loads the dataset index
- prepares the text cache when missing
- prepares the latent cache when missing
- fails clearly on stale or incompatible caches unless `cache.rebuild_if_stale: true`
- trains MMDiT RF
- saves checkpoints, config snapshots, metrics, and eval outputs

Useful profiles:

```bash
python -m train.cli --config config/train_smoke.yaml
python -m train.cli --config config/train_overfit.yaml
python -m train.cli --config config/train_dev.yaml
python -m train.cli --config config/train.yaml
```

Dry run:

```bash
python -m train.cli --config config/train.yaml --dry-run
```

## Sampling

```bash
python -m sample.cli \
  --ckpt ./runs/.../ckpt_final.pt \
  --prompt "1girl, blue hair, white dress" \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out ./samples/out.png
```

Supported samplers are `flow_euler` and `flow_heun`.

## Tests

```bash
python -m pytest -q
```

## Optional Cache Tools

Normally `train` prepares caches automatically. These scripts remain available for manual precomputation and debugging:

```bash
python -m scripts.prepare_text_cache --config config/train.yaml
python -m scripts.prepare_latents --config config/train.yaml
```

## Production Checklist

Run stages in order:

```bash
python -m train.cli --config config/train_overfit.yaml
python -m train.cli --config config/train_dev.yaml
python -m train.cli --config config/train.yaml
```

Move from overfit to dev when loss falls sharply. Move from dev to full when there are no NaN/Inf failures, resume works, sampling creates images, and VRAM remains stable.
