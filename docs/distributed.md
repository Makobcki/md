# Distributed / Accelerate

Distributed training uses Accelerate.

---

## Dry-run distributed profile

```bash
python -m train.cli --profile distributed_smoke --dry-run
```

---

## Configure Accelerate

```bash
accelerate config
```

---

## Launch distributed training

```bash
accelerate launch -m train.cli --config config/train_distributed_smoke.yaml
```

---

## Rank behavior

Checkpoint and event writing are performed only on rank 0.

Metrics are aggregated through the distributed context.

---

## FSDP status

FSDP is not enabled in the default trainer.

Template and documentation files:

```text
config/train_fsdp_template.yaml
docs/distributed_and_fsdp.md
```

Validate FSDP template:

```bash
python -m train.cli --profile fsdp_template --dry-run
```

Use the template as a starting point only after the standard single-GPU and distributed smoke paths work.
