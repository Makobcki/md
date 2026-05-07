# Model Families

`md-diffusion` supports three v1 model families through `model.family`.

| Family | Objective | Input | Sampler | Train | Sample | img2img | inpaint | control |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mmdit` | `rectified_flow` / velocity | latents + text | `flow_euler`, `flow_heun` | yes | yes | yes | yes | yes |
| `pixart_sigma` | `rectified_flow` / velocity | latents + text | `flow_euler`, `flow_heun` | yes | yes | no | no | no |
| `var` | `next_scale_prediction` / token logits | discrete multiscale tokens | `var_autoregressive` | yes | yes | no | no | no |

## MMDiT

Old configs without `model.family` still normalize to `mmdit`. The legacy flat `architecture="mmdit_rf"` path remains valid.

```yaml
model:
  family: mmdit
  variant: mmdit_rf_base
  architecture:
    image_size: 576
  diffusion:
    objective: rectified_flow
    prediction_type: velocity
```

## PixArt-Sigma-Style

PixArt-Sigma support is project-native RF support. It is not an external checkpoint compatibility promise.

```yaml
model:
  family: pixart_sigma
  variant: pixart_sigma_512
  architecture:
    image_size: 512
    latent_size: 64
    latent_channels: 4
    patch_size: 2
    hidden_size: 1152
    depth: 28
    num_heads: 16
    caption_channels: 4096
    max_text_tokens: 300
  diffusion:
    objective: rectified_flow
    prediction_type: velocity
```

Non-goals for v1: external PixArt-Sigma checkpoint import, img2img, inpaint, and control.

## VAR

VAR uses discrete multiscale tokens and cross-entropy. It does not use RF samplers, VAE latent denoising, or diffusion targets.

```yaml
model:
  family: var
  variant: var_d16
  architecture:
    scale_schedule: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    max_token_length: 680
  tokenizer:
    kind: vq
    codebook_size: 4096
    codebook_dim: 32
    downsample_factor: 16
    checkpoint: null
  autoregressive:
    objective: next_scale_prediction
    prediction_type: token_logits
    conditioning: none
    causal_mode: scale_causal
    loss: cross_entropy
```

Non-goals for v1: diffusion samplers, img2img, inpaint, control, and RF latent velocity prediction.
