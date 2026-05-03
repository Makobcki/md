# Distributed training and FSDP plan

## Supported now: Accelerate/DDP

The trainer has a rank-aware distributed path through `accelerate`:

```bash
accelerate config
accelerate launch -m train.cli --config config/train_distributed_smoke.yaml
```

The current implementation keeps single-process behavior unchanged. With
`distributed.backend: accelerate`, Accelerate prepares the model, optimizer and
dataloaders for DDP-style execution. Checkpoints, events, eval samples and run
metadata are written only on rank 0 by default.

Relevant config:

```yaml
distributed:
  backend: accelerate
  save_on_rank0_only: true
  metrics_aggregation: true
```

Supported behavior:

- single-GPU training remains the default (`backend: none`);
- Accelerate launch can wrap the trainer without changing CLI entry points;
- dataloaders are prepared for distributed execution;
- checkpoints are rank-0 only;
- train/eval scalar metrics are reduced before logging;
- AMP/bf16 flags continue to use the trainer's explicit AMP path.

## Planned later: FSDP

FSDP is intentionally **not enabled by default** and the trainer rejects
`fsdp.enabled: true` for now. The config path is reserved in
`config/train_fsdp_template.yaml` so the future large-model path has a stable
shape without silently enabling untested behavior.

Only consider enabling FSDP later when at least one of these is true:

- the model is larger than single-GPU VRAM;
- T5-large/XXL or multiple large text encoders are used;
- large effective batch size is required;
- `hidden_dim >= 1024` and checkpointing/SDPA are still insufficient.

Planned FSDP config shape:

```yaml
fsdp:
  enabled: false
  min_hidden_dim: 1024
  min_num_params: 500000000
  sharding_strategy: full_shard
  auto_wrap_policy: transformer_block
  cpu_offload: false
```

This is a deliberate safety boundary: DDP/Accelerate is available now; FSDP is a
future path that should only be enabled after a dedicated multi-GPU smoke test.
