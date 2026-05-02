# Model Development Plan

The project should prioritize the model and training loop over WebUI work.
WebUI remains a launcher and monitor for the same CLI/config surfaces.

## Tracks

- `config/train_image_only.yaml` is the stable latent image-only baseline.
- `config/train_text_to_image.yaml` is the main text-to-image quality track.
- `config/train.yaml` may remain a local working profile.

## Quality Loop

1. Prepare or verify the sharded latent cache before each run.
2. Run a short smoke test with the selected profile.
3. Keep fixed seeds and fixed eval prompts for every comparable run.
4. Compare EMA samples at the same steps before changing architecture again.
5. Record the winning config, checkpoint, visual notes, and obvious failures.

## Experiment Order

1. Establish image-only baseline quality and resume stability.
2. Bring up text-to-image conditioning with captions, tags, CFG dropout, and eval prompts.
3. Run architecture sweeps for UNet capacity: channels, residual depth, attention heads, and attention resolutions.
4. Tune conditioning dropout only after the text-conditioned model reacts to prompts.
5. Enable compile and heavier performance options only for long stable runs.

## Acceptance Criteria

- Image-only eval works with an empty prompt file and produces stable EMA samples.
- Text-to-image eval requires a prompt file and produces prompt-dependent samples.
- Resume fails clearly when model architecture or conditioning mode is incompatible.
- Profile YAML files load through `TrainConfig` without relying on WebUI defaults.
