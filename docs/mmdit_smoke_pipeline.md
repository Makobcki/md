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
```
