# Training

Training is done through `train.cli`.

---

## Dry-runs

Before starting real training, validate profiles and configs.

Profiles:

```bash
python -m train.cli --profile smoke --dry-run
python -m train.cli --profile overfit --dry-run
python -m train.cli --profile dev --dry-run
python -m train.cli --profile base --dry-run
```

Explicit config:

```bash
python -m train.cli --config config/train.yaml --dry-run
```

---

## Training profiles

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

Milestone runs:

```bash
python -m train.cli --profile milestone_a
python -m train.cli --profile milestone_b
python -m train.cli --profile milestone_c
```

---

## Explicit config

```bash
python -m train.cli --config config/train.yaml
```

---

## Resume training

```bash
python -m train.cli \
  --config config/train.yaml \
  --resume runs/.../checkpoints/latest.pt
```

Use a checkpoint compatible with the same architecture and config.

---

## Run directory

Training writes outputs to a run directory:

```text
runs/
  2026-05-03_001_dev768/
    config.yaml
    config_resolved.yaml
    config_snapshot.yaml
    train.log
    events.jsonl
    checkpoints/
      step_000100.pt
      latest.pt
      final.pt
    samples/
    eval/
    cache_manifest.json
```

---

## Train events

Example train event:

```json
{
  "type": "train",
  "step": 100,
  "loss": 0.123,
  "train_loss": 0.123,
  "lr": 0.0001,
  "grad_norm": 0.91,
  "grad_norm_total": 0.91,
  "has_nan_grad": false,
  "has_inf_grad": false,
  "samples_per_sec": 3.2,
  "loss_t_bin_00_01": 0.12,
  "loss_t_bin_09_10": 0.17
}
```

---

## Validation events

Example validation event:

```json
{
  "type": "eval",
  "step": 1000,
  "val_loss": 0.101,
  "val_loss_t_bin_00_01": 0.09
}
```

---

## Recommended run order

1. Verify dataset.
2. Dry-run config.
3. Prepare text cache.
4. Prepare latent cache.
5. Validate cache.
6. Run smoke training.
7. Run overfit training.
8. Run dev or base training.
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

## CUDA OOM mitigation

Reduce batch size:

```yaml
training:
  batch_size: 1
  grad_accum_steps: 32
```

Enable gradient checkpointing:

```yaml
model:
  gradient_checkpointing: true
```

Use a smaller profile first:

```bash
python -m train.cli --profile smoke
python -m train.cli --profile overfit
python -m train.cli --profile dev
```
