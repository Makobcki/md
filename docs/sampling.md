# Sampling

Sampling is done through `sample.cli`.

Supported samplers:

- `flow_euler`
- `flow_heun`

---

## Text-to-image

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "1girl, blue hair, white dress" \
  --neg_prompt "low quality, blurry" \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --shift 3.0 \
  --seed 42 \
  --out samples/txt2img.png
```

---

## Euler sampler

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "a cat on a table" \
  --sampler flow_euler \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/euler.png
```

---

## Image-to-image

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task img2img \
  --prompt "same character, winter outfit" \
  --init-image input.png \
  --strength 0.55 \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/img2img.png
```

Required arguments:

- `--task img2img`
- `--init-image`
- `--strength`

---

## Inpaint

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task inpaint \
  --prompt "replace the background with a neon city" \
  --init-image input.png \
  --mask mask.png \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/inpaint.png
```

Required arguments:

- `--task inpaint`
- `--init-image`
- `--mask`

---

## Control path

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task control \
  --prompt "edge guided cat" \
  --control-image control.png \
  --control-strength 0.75 \
  --sampler flow_heun \
  --steps 28 \
  --out samples/control.png
```

Control mode is experimental unless the checkpoint was trained with compatible control conditioning.

Required arguments:

- `--task control`
- `--control-image`

---

## EMA

Use EMA weights:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "a cat" \
  --use-ema \
  --out samples/ema.png
```

Use raw weights:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "a cat" \
  --no-ema \
  --out samples/raw.png
```

---

## Latent-only smoke sample

Use this for smoke tests without VAE decode:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "smoke" \
  --latent-only \
  --device cpu \
  --steps 2 \
  --n 1 \
  --out samples/latent.pt
```

---

## Fake VAE smoke sample

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "smoke" \
  --fake-vae \
  --device cpu \
  --steps 2 \
  --n 1 \
  --out samples/fake.png
```

---

## Metadata sidecar

Sampling writes a JSON sidecar next to the output:

```text
samples/txt2img.png
samples/txt2img.json
```

Metadata contains:

- checkpoint path;
- checkpoint step;
- prompt;
- negative prompt;
- sampler;
- steps;
- CFG;
- seed;
- shift;
- latent shape;
- model config;
- VAE config;
- text encoder config.
