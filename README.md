# MMDiT Rectified Flow image model

Latent **MMDiT Rectified Flow** image generation project.

The supported model line is **MMDiT RF-only**:

- latent-space image generation;
- frozen CLIP/T5 text encoders;
- joint text/image attention;
- flow-matching objective;
- Flow Euler / Flow Heun sampling;
- text-to-image, image-to-image, inpaint and experimental control paths.

Legacy **U-Net / DDPM / DDIM / DPM** workflows are not used.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install PyTorch separately using the command recommended for your CUDA driver or CPU environment from the official PyTorch selector.

Then install the project:

```bash
python -m pip install -e ".[all]"
```

Run a smoke dry-run:

```bash
python -m train.cli --profile smoke --dry-run
```

Run smoke training:

```bash
python -m train.cli --profile smoke
```

Generate a sample:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "a cat on a table" \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/test.png
```

---

## Documentation

| Topic                    | File                                                 |
| ------------------------ | ---------------------------------------------------- |
| General usage            | [`docs/usage.md`](docs/usage.md)                     |
| Dataset format           | [`docs/dataset.md`](docs/dataset.md)                 |
| Configuration            | [`docs/configuration.md`](docs/configuration.md)     |
| Cache preparation        | [`docs/cache.md`](docs/cache.md)                     |
| Training                 | [`docs/training.md`](docs/training.md)               |
| Sampling                 | [`docs/sampling.md`](docs/sampling.md)               |
| Evaluation               | [`docs/evaluation.md`](docs/evaluation.md)           |
| WebUI                    | [`docs/webui.md`](docs/webui.md)                     |
| Distributed / Accelerate | [`docs/distributed.md`](docs/distributed.md)         |
| Troubleshooting          | [`docs/troubleshooting.md`](docs/troubleshooting.md) |

---

## Supported workflows

<details>
<summary><strong>Text-to-image</strong></summary>

Uses text tokens from frozen CLIP/T5 encoders and generates an image latent from noise using a Rectified Flow sampler.

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "1girl, blue hair, white dress" \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --out samples/txt2img.png
```

</details>

<details>
<summary><strong>Image-to-image</strong></summary>

Uses an initial image latent as source conditioning.

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
  --out samples/img2img.png
```

</details>

<details>
<summary><strong>Inpaint</strong></summary>

Uses source image latent and mask conditioning.

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
  --out samples/inpaint.png
```

</details>

<details>
<summary><strong>Control</strong></summary>

Experimental control path for future ControlNet/IP-Adapter-style development.

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

A checkpoint must be trained with compatible control conditioning for meaningful results.

</details>

---

## Requirements

Recommended environment:

- Linux;
- Python 3.11+;
- NVIDIA GPU for real training and inference;
- CUDA-compatible PyTorch build;
- `git`, `python-venv`, `build-essential`;
- `npm` only for the WebUI frontend.

For HuggingFace-hosted encoders or VAE models, set a token when needed:

```bash
export HF_TOKEN=...
```

---

## Installation

Base development install:

```bash
python -m pip install -e ".[all]"
```

Minimal ML install without WebUI/dev extras:

```bash
python -m pip install -e ".[ml]"
```

WebUI install:

```bash
python -m pip install -e ".[web,ml]"
cd webui/frontend
npm install
cd ../..
```

Development/test install:

```bash
python -m pip install -e ".[all,dev]"
```

After editable installation, console scripts are available:

```bash
md-train --help
md-sample --help
md-prepare-latents --help
md-prepare-text-cache --help
md-prepare-training-cache --help
md-eval --help
md-cache-validate --help
md-webui --help
```

Equivalent module commands are used throughout the documentation:

```bash
python -m train.cli --help
python -m sample.cli --help
python -m scripts.prepare_latents --help
python -m scripts.prepare_text_cache --help
python -m scripts.prepare_training_cache --help
python -m train.eval_cli --help
python -m scripts.validate_cache --help
python -m main --help
```

---

## Dataset summary

Minimal dataset layout:

```text
data/dataset/
  images/
    <hash>.png
    <hash>.jpg
  metadata.jsonl
```

Minimal `metadata.jsonl` row:

```json
{ "md5": "hash", "file_name": "hash.png", "prompt": "a cat on a table" }
```

Recommended config fields:

```yaml
data_root: ./data/dataset
image_dir: images
meta_dir: ""
tags_dir: ""

text_field: prompt
text_fields: [prompt, caption, text]
```

See [`docs/dataset.md`](docs/dataset.md).

---

## Architecture

```text
image / source image / mask / control image
        ↓
VAE encoder or latent cache
        ↓
latent z: [B, 4, 64, 64] for 512×512
        ↓
patchify, p=2
        ↓
image tokens: [B, 1024, D]
        ↓
MMDiT / Flux-like transformer
        ↑
frozen text encoders: CLIP + T5 via text cache
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

Supported conditioning modes:

| Mode      | Conditioning                                                           |
| --------- | ---------------------------------------------------------------------- |
| `txt2img` | target noisy latent + text tokens                                      |
| `img2img` | target noisy latent + source latent tokens + text tokens               |
| `inpaint` | target noisy latent + source latent tokens + mask tokens + text tokens |
| `control` | control latent token stream; experimental                              |

---

## Recommended first run

1. Prepare dataset.
2. Verify config.
3. Prepare text cache.
4. Prepare latent cache.
5. Validate cache.
6. Run smoke training.
7. Run overfit training.
8. Start dev/base training.
9. Sample checkpoint.
10. Run eval grids.

Commands:

```bash
python -m train.cli --config config/train.yaml --dry-run

python -m scripts.prepare_text_cache --config config/train.yaml
python -m scripts.prepare_latents --config config/train.yaml
python -m scripts.validate_cache --config config/train.yaml

python -m train.cli --profile smoke
python -m train.cli --profile overfit
python -m train.cli --profile dev
```

---

## Current limitations

- Only the MMDiT Rectified Flow line is supported.
- U-Net/DDPM/DDIM/DPM workflow is intentionally not part of the active pipeline.
- Real training expects prepared text and latent caches.
- CPU mode is intended for tests and smoke paths only.
- Control mode is experimental unless the checkpoint was trained with compatible control conditioning.
- FSDP is documented as a template but is not enabled in the default trainer.

---

## Repository layout

Typical project layout:

```text
project/
  config/
  docs/
  sample/
  scripts/
  train/
  webui/
  data/
  runs/
  README.md
```

Generated outputs are usually written to:

```text
runs/
  <timestamp>_<profile>/
    config.yaml
    config_resolved.yaml
    config_snapshot.yaml
    train.log
    events.jsonl
    checkpoints/
    samples/
    eval/
    cache_manifest.json
```

---

## WebUI

Backend + frontend:

```bash
python -m main \
  --host 127.0.0.1 \
  --port 8000 \
  --frontend \
  --frontend-host 127.0.0.1 \
  --frontend-port 5173
```

Frontend URL:

```text
http://127.0.0.1:5173
```

Do not expose WebUI to the public internet without authentication.

See [`docs/webui.md`](docs/webui.md).

---

## Troubleshooting

Common issues:

- text cache key missing;
- text cache dataset hash mismatch;
- stale latent cache;
- latent shape mismatch;
- incompatible checkpoint config;
- missing inpaint mask;
- CUDA out of memory.

See [`docs/troubleshooting.md`](docs/troubleshooting.md).

---

## License

Add the project license here.
