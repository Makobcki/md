# Checkpoint Metadata v2

Metadata v2 stores model-family identity and runtime compatibility data under `checkpoint`.

```yaml
checkpoint:
  metadata_version: 2
  model:
    family: pixart_sigma
    variant: pixart_sigma_512
    architecture: pixart_sigma_rf
    objective: rectified_flow
    prediction_type: velocity
    model_config: {}
    config_hash: ""
  io:
    input_kind: latent
    output_kind: latent_velocity
  capabilities: {}
  text_config: {}
  vae_config: {}
  tokenizer_config: null
  optimizer_config: {}
  ema_config: {}
  training_state:
    global_step: 0
    epoch: 0
```

Compatibility rules:

- Same family and compatible dimensions are accepted.
- Different families are rejected.
- New families without metadata v2 are rejected.
- Legacy MMDiT checkpoints continue through the existing MMDiT compatibility path.
- VAR metadata includes tokenizer kind, codebook size, codebook dim, scale schedule, max token length, and tokenizer config hash.
