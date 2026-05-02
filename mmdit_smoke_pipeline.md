# MMDiT RF smoke pipeline

This pipeline verifies the `architecture: mmdit_rf` path from cache preparation through a tiny train run, checkpoint resume, and PNG sampling.

## 1. Run tests

```bash
make test-mmdit
make test
```

## 2. Prepare text cache

CUDA:

```bash
make prepare-mmdit-text
```

CPU fallback:

```bash
make prepare-mmdit-text-cpu
```

## 3. Prepare latent cache

```bash
make prepare-mmdit-latents
```

## 4. Run smoke script

```bash
make smoke-mmdit
```

The smoke script checks cache coverage, runs a forward/backward pass, saves and reloads a checkpoint, and runs `flow_heun` for two latent steps without VAE decode.

## 5. Run tiny training

```bash
make train-mmdit-smoke
```

## 6. Resume tiny training

```bash
make train-mmdit-smoke-resume
```

Expected resume log:

```text
[INFO] Resumed mmdit_rf from ./runs/mmdit_smoke/ckpt_latest.pt at step 10
```

After resume, `./runs/mmdit_smoke/ckpt_final.pt` should contain `step=15`.

## 7. Generate sample

```bash
make sample-mmdit-smoke
```

Equivalent command:

```bash
md-sample \
  --ckpt ./runs/mmdit_smoke/ckpt_final.pt \
  --prompt "1girl, simple background" \
  --sampler flow_heun \
  --steps 2 \
  --cfg 1 \
  --seed 42 \
  --out ./samples/mmdit_smoke.png
```

Expected output:

```text
samples/mmdit_smoke.png
samples/mmdit_smoke.json
```

The JSON sidecar records checkpoint path, architecture, prompt, negative prompt, sampler, steps, CFG, seed, and sample count.

## 8. Optional architecture-only synthetic smoke

Use this when you need a fast CI check without dataset, VAE, transformers, latent cache, or text cache:

```bash
make smoke-mmdit-synthetic
```

Equivalent command:

```bash
md-smoke-mmdit-rf \
  --config config/train_mmdit_rf_smoke.yaml \
  --synthetic
```

This still creates random latents and random `TextConditioning`, runs forward/backward, saves and reloads a checkpoint, and runs `flow_heun` for two steps.

## 9. Next control stage: overfit run

After the smoke pipeline passes, use the overfit profile instead of `dev` to prove that MMDiT RF can memorize a tiny dataset.

```bash
make prepare-mmdit-overfit-text
make prepare-mmdit-overfit-latents
make train-mmdit-overfit
make sample-mmdit-overfit
```

CPU text-cache fallback:

```bash
make prepare-mmdit-overfit-text-cpu
```

Equivalent commands:

```bash
md-prepare-text-cache \
  --config config/train_mmdit_rf_overfit.yaml \
  --device cuda

md-prepare-latents \
  --config config/train_mmdit_rf_overfit.yaml

md-train \
  --config config/train_mmdit_rf_overfit.yaml

md-sample \
  --ckpt ./runs/mmdit_overfit/ckpt_final.pt \
  --prompt "1girl, simple background" \
  --sampler flow_heun \
  --steps 16 \
  --cfg 3 \
  --seed 42 \
  --out ./samples/mmdit_overfit.png
```

Success criteria:

```text
loss noticeably decreases
no NaN/Inf
ckpt_final.pt is saved
sample after overfit is reproducible
resume still works on the smoke profile
```
