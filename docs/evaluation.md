# Evaluation

Evaluation is done through `train.eval_cli`.

---

## Print prompt bank

```bash
python -m train.eval_cli \
  --prompt-set core \
  --count-per-set 3 \
  --print
```

---

## Fixed-seed eval grids

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

---

## CFG sweep

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --cfg-sweep 1.0 2.5 4.5 7.0
```

---

## Step sweep

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --step-sweep 8 16 28 40
```

---

## Sampler sweep

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --sampler-sweep flow_euler flow_heun
```

---

## Shift sweep

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --shift-sweep 1.0 2.0 3.0 4.0
```

---

## Combined step/sampler/shift sweep

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --step-sweep 8 16 28 40 \
  --sampler-sweep flow_euler flow_heun \
  --shift-sweep 1.0 2.0 3.0 4.0
```

---

## Resolution eval

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --resolution 768
```

---

## Output layout

```text
runs/.../eval/eval_768/step_000100/
  core_grid.png
  metadata.json
  events.jsonl
```

---

## Recommended evaluation routine

For a new checkpoint:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --seed 42 \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5
```

Then run sweeps only after confirming the checkpoint produces usable images.
