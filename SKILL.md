---
name: md-diffusion-skills
description: "Use when developing, testing, caching, training, or sampling with the md-diffusion latent image generation stack."
category: project-reference
risk: safe
source: community
date_added: "2026-06-06"
---

# md-diffusion Development Skills

A comprehensive developer manual and skill guide for the **md-diffusion** Latent MMDiT Rectified Flow image generation and training stack.

---

## ⚡ Quick Decision Tree

### What do you need to do?

1. **Set Up the Environment / Verify Installation:**
   - Install dependencies: `pip install -e ".[all]"`
   - Verify environment and syntax: `python -m scripts.lint` or `md-lint`
   - Run project validation suite: `bash scripts/check_project.sh`

2. **Prepare Dataset and Cache:**
   - Prepare text encodings (CLIP/T5): `md-prepare-text-cache`
   - Prepare image latents (VAE): `md-prepare-latents`
   - Validate cache integrity: `md-cache-validate`

3. **Train the Model:**
   - Run smoke dry-run: `md-train --dry-run --set training.use=single_gpu_debug`
   - Run smoke training: `md-train --set training.use=single_gpu_debug`
   - Use custom config: `md-train --config configs/train.kdl`

4. **Inference & Sampling:**
   - Text-to-Image: `md-sample --ckpt <path> --prompt "..." --sampler flow_heun`
   - Image-to-Image: `md-sample --ckpt <path> --task img2img --init-image <path> --strength 0.55`
   - Inpainting: `md-sample --ckpt <path> --task inpaint --init-image <path> --mask <path>`

5. **WebUI Operations:**
   - Launch WebUI server: `md-webui` or `python -m main --frontend`
   - Build/run React frontend: `cd webui/frontend && npm install && npm run dev`

---

## 📚 Component Index & Commands

| CLI Command / Script | Module Path | Purpose |
|----------------------|-------------|---------|
| `md-train` | `train.cli` | Main model training entry point |
| `md-sample` | `sample.cli` | Latent inference / generation entry point |
| `md-prepare-text-cache` | `scripts.prepare_text_cache` | Extracts and caches frozen CLIP/T5 text embeddings |
| `md-prepare-latents` | `scripts.prepare_latents` | Precomputes and caches VAE image latents |
| `md-cache-validate` | `scripts.validate_cache` | Asserts latent/text cache alignment with dataset metadata |
| `md-eval` | `train.eval_cli` | Runs grid sampling and evaluation checks |
| `md-webui` | `main` | Runs the FastAPI + React WebUI server |
| `md-lint` | `scripts.lint` | Project linter (compilation, hygiene check) |
| `md-config-resolve` | `config.cli` | Resolves KDL/YAML configuration objects |

---

## 🛠️ Developer Workflows

### 1. Verification & Formatting Loop
Always run this loop before committing any changes. 
The project includes a validation script `scripts/check_project.sh` that checks for:
- Syntax errors (`py_compile`)
- Coding style & hygiene (no trailing whitespace, correct UTF-8, LF endings)
- Ruff lint errors (Ruff is the primary linter tool)
- Test suite pass rates (`pytest`)
- Dry-runs for training configs and sample CLIs
- Banned legacy keywords

**Command:**
```bash
bash scripts/check_project.sh
```

**Banned Legacy Terms Check:**
The project is strictly **MMDiT Rectified Flow only**. Do not introduce legacy architectures or keywords:
- 🚫 `unet` / `U-Net`
- 🚫 `DDPM` / `DDIM` / `DPM`
- 🚫 `BPE`
- 🚫 `v_prediction`
- 🚫 `min_snr`

### 2. Cache Preparation
Training requires pre-cached text embeddings and image latents for high performance.

**Step 1: Text Cache**
```bash
python -m scripts.prepare_text_cache --config configs/train.kdl
```
**Step 2: Latents Cache**
```bash
python -m scripts.prepare_latents --config configs/train.kdl
```
**Step 3: Validate**
```bash
python -m scripts.validate_cache --config configs/train.kdl
```

### 3. Running WebUI Frontend
The frontend is built using Vite and React.
```bash
cd webui/frontend
npm install
npm run dev
```
To run backend and frontend jointly:
```bash
python -m main --host 127.0.0.1 --port 8000 --frontend --frontend-host 127.0.0.1 --frontend-port 5173
```

---

## 📝 Coding Standards

1. **Style**: Follow the Google Python Style Guide.
2. **Type Hints**: Mandatory for all new or modified functions/classes.
3. **Config**: Nested configurations using YAML or KDL. Flat configurations are prohibited.
4. **Exceptions**: Do not use empty `except:` blocks. Raise specific exceptions (e.g. `ValueError`, `RuntimeError`).
5. **Linting**:
   ```bash
   # Automatically fix formatting and safe lint errors
   ruff check --fix .
   ruff format .
   python -m scripts.lint --fix
   ```

---

## 🔗 Related Documentation

- Detailed Dataset Specs: [docs/dataset.md](file:///home/frosty/md/docs/dataset.md)
- Training Details: [docs/training.md](file:///home/frosty/md/docs/training.md)
- Troubleshooting Guide: [docs/troubleshooting.md](file:///home/frosty/md/docs/troubleshooting.md)
