# Usage guide

This document describes the recommended end-to-end workflow for training and sampling a latent MMDiT Rectified Flow image model.

---

## 1. Environment

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install PyTorch separately using the command appropriate for your CUDA driver or CPU environment.

Then install the project:

```bash
python -m pip install -e ".[all]"
```

For a minimal ML-only install:

```bash
python -m pip install -e ".[ml]"
```

For WebUI:

```bash
python -m pip install -e ".[web,ml]"
cd webui/frontend
npm install
cd ../..
```

For development and tests:

```bash
python -m pip install -e ".[all,dev]"
```

---

## 2. Verify installation

Run tests:

```bash
pytest -q
```

Run local CI-style checks:

```bash
bash scripts/check_project.sh
```

Run dry-run profiles:

```bash
python -m train.cli --profile smoke --dry-run
python -m train.cli --profile overfit --dry-run
python -m train.cli --profile dev --dry-run
python -m train.cli --profile base --dry-run
```

Milestone profiles:

```bash
python -m train.cli --profile milestone_a --dry-run
python -m train.cli --profile milestone_b --dry-run
python -m train.cli --profile milestone_c --dry-run
```

Distributed/FSDP templates:

```bash
python -m train.cli --profile distributed_smoke --dry-run
python -m train.cli --profile fsdp_template --dry-run
```

---

## 3. Prepare dataset

Minimal structure:

```text
data/dataset/
  images/
    <hash>.png
    <hash>.jpg
  metadata.jsonl
```

Minimal metadata row:

```json
{ "md5": "hash", "file_name": "hash.png", "prompt": "a cat on a table" }
```

Recommended config:

```yaml
data_root: ./data/dataset
image_dir: images
meta_dir: ""
tags_dir: ""

text_field: prompt
text_fields: [prompt, caption, text]
```

More details: [`dataset.md`](dataset.md).

---

## 4. Configure VAE and text encoders

Real latent training requires a VAE compatible with `diffusers.AutoencoderKL.from_pretrained`.

Example:

```yaml
vae:
  pretrained: ./vae_sd_mse
  freeze: true
  scaling_factor: 0.18215
```

Example text encoder config:

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

When HuggingFace authentication is required:

```bash
export HF_TOKEN=...
```

---

## 5. Prepare cache

For large datasets, prepare caches explicitly:

```bash
python -m scripts.prepare_text_cache --config config/train.yaml
python -m scripts.prepare_latents --config config/train.yaml
python -m scripts.validate_cache --config config/train.yaml
```

One-command cache preparation:

```bash
python -m scripts.prepare_training_cache --config config/train.yaml
```

More details: [`cache.md`](cache.md).

---

## 6. Train

Smoke run:

```bash
python -m train.cli --profile smoke
```

Overfit run:

```bash
python -m train.cli --profile overfit
```

Development run:

```bash
python -m train.cli --profile dev
```

Base run:

```bash
python -m train.cli --profile base
```

Explicit config:

```bash
python -m train.cli --config config/train.yaml
```

Resume:

```bash
python -m train.cli \
  --config config/train.yaml \
  --resume runs/.../checkpoints/latest.pt
```

More details: [`training.md`](training.md).

---

## 7. Sample

Text-to-image:

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

More details: [`sampling.md`](sampling.md).

---

## 8. Evaluate

Print prompt bank:

```bash
python -m train.eval_cli \
  --prompt-set core \
  --count-per-set 3 \
  --print
```

Generate fixed-seed eval grids:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --prompt-set style \
  --seed 42 \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5
```

More details: [`evaluation.md`](evaluation.md).

---

## 9. Run WebUI

Backend + frontend:

```bash
python -m main \
  --host 127.0.0.1 \
  --port 8000 \
  --frontend \
  --frontend-host 127.0.0.1 \
  --frontend-port 5173
```

Frontend:

```text
http://127.0.0.1:5173
```

More details: [`webui.md`](webui.md).
