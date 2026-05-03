# MMDiT RF smoke pipeline

This pipeline verifies the `architecture: mmdit_rf` path from cache preparation through a tiny train run, checkpoint resume, PNG sampling, and the next overfit control stage.

## 0. Expected dataset and prompt layout

Datasets live under `data/dataset/` and remain ignored by git. Eval prompts are tracked separately under `data/eval_prompts/`. The smoke and overfit profiles read the dataset root from their config files. By default they expect:

```text
data/dataset/pixso_512/
data/eval_prompts/mmdit_core.txt
```

The default dataset profile uses `image_dir: images` and supports `metadata.jsonl` at the dataset root. Tests read these values from `config/train_mmdit_rf_smoke.yaml` instead of hardcoding a dataset name.

With `eval_every: 0`, the smoke script does not require `data/eval_prompts/mmdit_core.txt`; it should fail on missing cache/dataset state instead of missing eval prompts.

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

After resume, verify that `./runs/mmdit_smoke/ckpt_final.pt` contains `step=15`:

```bash
make check-mmdit-smoke-resume
```

## 7. Generate sample

```bash
make sample-mmdit-smoke
```

This target requires `./runs/mmdit_smoke/ckpt_final.pt`, so run it after `make train-mmdit-smoke` or the resume target.

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
make train-mmdit-overfit-resume
make check-mmdit-overfit
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

md-train \
  --config config/train_mmdit_rf_overfit_resume.yaml

python scripts/check_checkpoint_step.py \
  --ckpt ./runs/mmdit_overfit/ckpt_final.pt \
  --step 2000

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
resume works on the smoke profile and has an overfit resume profile available
```
