# MMDiT Rectified Flow image model

Проект обучает и запускает **latent MMDiT Rectified Flow** модель для генерации изображений. Текущая линия модели — **MMDiT RF-only**: latent space, frozen CLIP/T5 text encoders, joint text/image attention, flow-matching objective, Flow Euler/Heun sampling. Старый U-Net/DDPM/DDIM/DPM workflow не используется.

Поддерживаемые режимы:

- `txt2img` — генерация из текста;
- `img2img` — source latent + text;
- `inpaint` — source latent + mask + text;
- `control` — control-image/control-latent stream + text;
- `latent-only` — smoke/debug output без VAE decode.

## 1. Требования и установка

Рекомендуемое окружение:

- Linux;
- Python 3.11+;
- NVIDIA GPU для реального обучения/инференса;
- CUDA-сборка PyTorch, подходящая под драйвер;
- `git`, `python-venv`, `build-essential`;
- Node.js `^20.19.0 || >=22.12.0` только для frontend build/dev mode.

Создание Python окружения:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Установи PyTorch под свою CUDA/CPU сборку, затем проект:

```bash
python -m pip install -e ".[all]"
```

Минимальная ML-установка без WebUI/dev extras:

```bash
python -m pip install -e ".[ml]"
```

WebUI-зависимости:

```bash
python -m pip install -e ".[web,ml]"
```

Dev/test-зависимости:

```bash
python -m pip install -e ".[all,dev]"
```

Frontend build:

```bash
cd webui/frontend
npm ci
npm run build
cd ../..
```

`webui/frontend/dist` используется как static frontend fallback. В dev checkout можно запускать Vite dev server, если есть `webui/frontend/node_modules`.

Если text encoders скачиваются с HuggingFace, задай token:

```bash
export HF_TOKEN=...
```

Offline/local-only режим для text encoders:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# или
export MD_LOCAL_FILES_ONLY=1
```

## 2. Архитектура

```text
image / source image / mask / control image
        ↓
VAE encoder или latent cache
        ↓
latent z: [B, 4, 64, 64] для 512×512
        ↓
patchify, p=2
        ↓
image tokens: [B, 1024, D]
        ↓
MMDiT / Flux-like transformer
        ↑
frozen text encoders: CLIP + T5 через text cache
        ↓
unpatchify
        ↓
predicted flow velocity
        ↓
Flow sampler: Euler / Heun
        ↓
VAE decoder
        ↓
image
```

Важные детали текущей реализации:

- Conditioning streams имеют per-sample masks, поэтому mixed-task batch не даёт `txt2img` строкам лишние `img2img/inpaint/control` tokens.
- Control stream gated целиком: `control_gate * (control_base + type_token)`.
- Inpaint loss weighting применяется только к rows с `task == "inpaint"`.
- `PatchEmbed` и direct model API валидируют divisibility latent H/W by patch size.
- `cfg_predict()` корректно дублирует batch-aligned tensor/list/tuple kwargs.
- `TextConditioning.replace_where()` корректно обрабатывает `token_types`.
- Flow timesteps и objective валидируют positive `shift/timestep_shift`.
- Sampling API использует строгую matrix допустимых task/input combinations.
- Sampling checkpoint грузится на CPU перед переносом модели на target device.

## 3. Dataset

Минимальная структура:

```text
data/dataset/
  images/
    <hash>.png
    <hash>.jpg
  metadata.jsonl
```

Каждая строка `metadata.jsonl` — JSON object:

```json
{ "md5": "hash", "file_name": "hash.png", "prompt": "a cat on a table" }
```

Поддерживаются per-image JSON sidecars через `meta_dir`:

```text
data/dataset/
  images/hash.png
  meta/hash.json
```

Пример:

```json
{
  "md5": "hash",
  "file_name": "hash.png",
  "prompt": "a cat on a table",
  "caption": "cat sitting on a wooden table",
  "tags": ["cat", "table"]
}
```

Поле изображения может называться `file_name`, `filename`, `image`, `img`, `path` или `image_path`. Если поле не указано, loader ищет `images/<md5>.*`.

### Prompt-first training

Для обучения на prompt-полях:

```yaml
text_field: prompt
text_fields: [prompt, caption, text]
```

или nested-вариант:

```yaml
dataset:
  text_field: prompt
  text_fields: [prompt, caption, text]
```

`caption_field` остаётся legacy-compatible, но для новых запусков предпочтительны `text_field/text_fields`.

Dataset index сохраняет:

```json
{
  "text": "resolved training text",
  "text_source": "prompt",
  "caption": "resolved training text"
}
```

`caption` заполняется тем же текстом для обратной совместимости.

### Prompts as sidecar `.txt`

Если данные лежат так:

```text
data/dataset/
  images/hash.png
  prompts/hash.txt
```

собери `metadata.jsonl`:

```bash
python - <<'PY'
from pathlib import Path
import json

root = Path("data/dataset")
images = root / "images"
prompts = root / "prompts"
out = root / "metadata.jsonl"

allowed = {".png", ".jpg", ".jpeg", ".webp"}
rows = []
for img in sorted(images.iterdir()):
    if img.suffix.lower() not in allowed:
        continue
    prompt_path = prompts / f"{img.stem}.txt"
    if not prompt_path.exists():
        continue
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        continue
    rows.append({"md5": img.stem, "file_name": img.name, "prompt": prompt})

out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {len(rows)} rows to {out}")
PY
```

Config:

```yaml
data_root: ./data/dataset
image_dir: images
meta_dir: ""
tags_dir: ""
text_field: prompt
text_fields: [prompt, caption, text]
```

## 4. Config profiles

Основные профили:

```bash
python -m train.cli --profile smoke --dry-run
python -m train.cli --profile overfit --dry-run
python -m train.cli --profile dev --dry-run
python -m train.cli --profile base --dry-run
python -m train.cli --profile milestone_a --dry-run
python -m train.cli --profile milestone_b --dry-run
python -m train.cli --profile milestone_c --dry-run
python -m train.cli --profile distributed_smoke --dry-run
python -m train.cli --profile fsdp_template --dry-run
```

`config/train.yaml` alias-compatible с `config/train_base.yaml`.

Nested config keys поддерживаются для основных секций:

```yaml
model:
  hidden_dim: 1024
  depth: 24
  num_heads: 16
  mlp_ratio: 4.0
  qk_norm: true
  rms_norm: true
  swiglu: true

training:
  batch_size: 2
  grad_accum_steps: 32
  num_workers: 8
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true
  lr: 1.0e-4
  lr_scheduler: cosine
  min_lr_ratio: 0.1
  decay_steps: 300000
  fail_on_nonfinite_grad: false
  ckpt_keep_last: 5
  deterministic: false
  amp: true
  amp_dtype: bf16
```

`text:` может быть dict или отсутствовать. Некорректный non-dict `text:` больше не должен ронять config flattening.

## 5. VAE и text encoders

Для реального latent training нужен VAE, совместимый с `diffusers.AutoencoderKL.from_pretrained`:

```yaml
vae:
  pretrained: ./vae_sd_mse
  freeze: true
  scaling_factor: 0.18215
```

Text encoders:

```yaml
text:
  enabled: true
  encoders:
    - name: clip_l
      model_name: openai/clip-vit-large-patch14
      max_length: 77
      trainable: false
      cache: true
    - name: t5
      model_name: google/t5-v1_1-base
      max_length: 256
      trainable: false
      cache: true
  text_dim: 1024
  pooled_dim: 1024
  empty_prompt_cache: true
```

Для smoke/debug есть fake/latent-only paths, но real training использует text/latent caches.

## 6. Cache preparation и validation

Training умеет auto-prepare, если в config:

```yaml
cache:
  auto_prepare: true
```

Для больших датасетов лучше готовить cache вручную.

### Text cache

```bash
python -m scripts.prepare_text_cache --config config/train.yaml
```

Output:

```text
data/dataset/.cache/text/
  manifest.json          # status: preparing|complete
  metadata.json
  index.jsonl
  empty_prompt.safetensors
  shards/text_00000.safetensors
```

Проверка:

```bash
python -m scripts.validate_cache --config config/train.yaml --text
```

### Latent cache

```bash
python -m scripts.prepare_latents --config config/train.yaml
```

Output:

```text
data/dataset/.cache/latents/
  manifest.json          # status: preparing|complete
  metadata.json
  shard_index.jsonl
  shards/*.pt
```

Текущее поведение:

- manifest пишется как `preparing -> complete`;
- training принимает cache только со статусом `complete`;
- non-sharded cache валидируется по metadata на старте;
- sharded cache валидирует metadata всех shards;
- aspect buckets пишут `latent_shape: null`, `latent_shapes`, `bucket_shapes`;
- per-entry expected shape проверяется против assigned bucket.

Aspect buckets требуют валидные dimensions в dataset entries: `width/height`, `image_width/image_height` или `size: [width, height]`.

### Unified training cache

```bash
python -m scripts.prepare_training_cache --config config/train.yaml
```

Repair/rebuild:

```bash
python -m scripts.prepare_training_cache --config config/train.yaml --repair
python -m scripts.prepare_training_cache --config config/train.yaml --repair --force
python -m scripts.prepare_training_cache --config config/train.yaml --repair --rebuild
```

Пересборка при смене prompts/text fields:

```bash
rm -rf data/dataset/.cache/text
python -m scripts.prepare_text_cache --config config/train.yaml
```

Пересборка при смене VAE/image size/latent dtype/buckets:

```bash
rm -rf data/dataset/.cache/latents
python -m scripts.prepare_latents --config config/train.yaml
```

## 7. Training

Smoke:

```bash
python -m train.cli --profile smoke
```

Overfit:

```bash
python -m train.cli --profile overfit
```

Dev/base:

```bash
python -m train.cli --profile dev
python -m train.cli --profile base
```

Явный config:

```bash
python -m train.cli --config config/train.yaml
```

Resume:

```bash
python -m train.cli --config config/train.yaml --resume latest
python -m train.cli --config config/train.yaml --resume runs/.../checkpoints/latest.pt
```

Run directory:

```text
runs/
  2026-05-03_001_dev768/
    config.yaml                # canonical resolved config
    config_resolved.yaml        # compatibility alias
    config_snapshot.yaml        # compatibility alias / WebUI snapshot
    config_manifest.yaml
    train.log
    events.jsonl
    checkpoints/
      step_000100.pt
      step_000100.pt.metadata.json
      latest.pt                 # hardlink/symlink alias, not full duplicate copy
      latest.pt.metadata.json
      final.pt
      final.pt.metadata.json
    ckpt_0000100.pt             # legacy alias
    ckpt_latest.pt              # legacy alias
    ckpt_final.pt               # legacy alias
    eval/
```

Training loop behavior:

- EMA обновляется только если optimizer step реально выполнен.
- Если AMP GradScaler пропустил step, EMA не обновляется.
- Nonfinite gradients не попадают в optimizer step.
- Nonfinite/AMP skipped steps логируются.
- `event_bus.close()` и `pbar.close()` вызываются в `finally`.
- `AspectBucketBatchSampler` поддерживает `set_epoch(epoch)` и меняет shuffle по epoch.
- Task sampling в mixed-task dataset детерминирован относительно `seed + idx + epoch`.

## 8. Checkpoints

Canonical checkpoint layout:

```text
<run_dir>/checkpoints/
  step_000100.pt
  step_000100.pt.metadata.json
  latest.pt
  latest.pt.metadata.json
  final.pt
  final.pt.metadata.json
```

Legacy root aliases:

```text
<run_dir>/ckpt_0000100.pt
<run_dir>/ckpt_latest.pt
<run_dir>/ckpt_final.pt
```

Root aliases создаются как hardlink/symlink, а не как полноценная вторая копия checkpoint bytes.

Metadata sidecar содержит:

```json
{
  "metadata": {
    "architecture": "mmdit_rf",
    "objective": "rectified_flow",
    "prediction_type": "flow_velocity",
    "model_config": {},
    "text_config": {},
    "vae_config": {},
    "flow_config": {},
    "train_config_hash": "...",
    "dataset_hash": "...",
    "step": 100
  },
  "cfg": {},
  "step": 100
}
```

Compatibility check учитывает architecture-affecting fields:

```text
hidden_dim
depth
num_heads
mlp_ratio
qk_norm
rms_norm
swiglu
double_stream_blocks
single_stream_blocks
pos_embed
patch_size
latent_channels
text_dim
pooled_dim
source_patch_size
mask_patch_size
control_patch_size
coarse_patch_size
text_resampler_enabled
text_resampler_num_tokens
text_resampler_depth
text_resampler_mlp_ratio
attention_schedule
early_joint_blocks
late_joint_blocks
```

`x0_aux_weight` — loss hyperparameter, не model compatibility field.

Checkpoint loading uses safe loader first:

```text
torch.load(..., weights_only=True)
```

Legacy unsafe fallback should only be used for trusted local artifacts. For stricter behavior:

```bash
export MD_ALLOW_UNSAFE_TORCH_LOAD=0
```

## 9. Sampling CLI

Txt2img:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task txt2img \
  --prompt "a cinematic photo of a red fox in snow" \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/fox.png
```

Img2img:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task img2img \
  --prompt "same scene, watercolor style" \
  --init-image input.png \
  --strength 0.6 \
  --steps 28 \
  --cfg 4.5 \
  --out samples/img2img.png
```

Inpaint:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task inpaint \
  --prompt "replace the background with a neon city" \
  --init-image input.png \
  --mask mask.png \
  --steps 28 \
  --cfg 4.5 \
  --out samples/inpaint.png
```

Control:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task control \
  --prompt "edge guided cat" \
  --control-image control.png \
  --control-type image \
  --control-strength 0.75 \
  --steps 28 \
  --out samples/control.png
```

Strict task/input matrix:

```text
txt2img -> no init_image, mask, control_image
img2img -> requires init_image; no mask/control_image
inpaint -> requires init_image + mask; no control_image
control -> requires control_image; no init_image/mask
```

`control_img2img` / `control_inpaint` are not implicit modes. Add explicit task names before combining those inputs.

Device default is `auto`:

```bash
python -m sample.cli --ckpt runs/.../checkpoints/latest.pt --prompt "a cat" --device auto --out samples/cat.png
```

EMA flags:

```bash
python -m sample.cli --ckpt runs/.../checkpoints/latest.pt --prompt "a cat" --use-ema --out samples/ema.png
python -m sample.cli --ckpt runs/.../checkpoints/latest.pt --prompt "a cat" --no-ema --out samples/raw.png
```

Latent-only output requires `.pt` or `.pth`:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "smoke" \
  --latent-only \
  --device cpu \
  --steps 2 \
  --n 1 \
  --out samples/latent.pt
```

Image output requires `.png`, `.jpg`, `.jpeg` or `.webp`.

Sampling loads checkpoint on CPU first, builds/loads model on CPU, then moves model to target device to reduce VRAM spikes. CPU sampling forces text/VAE dtype to `float32`.

Metadata sidecar is written next to every output:

```text
samples/txt2img.png
samples/txt2img.json
```

## 10. Evaluation

Prompt bank:

```bash
python -m train.eval_cli --prompt-set core --count-per-set 3 --print
```

Fixed seed eval grids:

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

Sweeps:

```bash
python -m train.eval_cli --ckpt runs/.../checkpoints/latest.pt --out-dir runs/... --prompt-set core --cfg-sweep 1.0 2.5 4.5 7.0
python -m train.eval_cli --ckpt runs/.../checkpoints/latest.pt --out-dir runs/... --prompt-set core --step-sweep 8 16 28 40 --sampler-sweep flow_euler flow_heun --shift-sweep 1.0 2.0 3.0 4.0
```

Resolution eval:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --resolution 768
```

Output:

```text
runs/.../eval/eval_768/step_000100/
  core_grid.png
  metadata.json
  events.jsonl
```

## 11. WebUI

Local backend + frontend:

```bash
export WEBUI_AUTH_TOKEN="secret"
python -m main --host 127.0.0.1 --port 8000 --frontend --frontend-host 127.0.0.1 --frontend-port 5173
```

Open:

```text
http://127.0.0.1:5173
```

Backend only:

```bash
python -m main --host 127.0.0.1 --port 8000 --no-frontend
```

Static frontend mode:

```bash
cd webui/frontend
npm ci
npm run build
cd ../..
python -m main --host 127.0.0.1 --port 8000 --frontend
```

Public bind requires auth:

```bash
export WEBUI_AUTH_TOKEN="secret"
python -m main --host 0.0.0.0 --port 8000 --frontend --frontend-host 0.0.0.0
```

For remote dev frontend, use relative API/Vite proxy or explicit API base:

```bash
python -m main \
  --host 0.0.0.0 \
  --frontend-host 0.0.0.0 \
  --api-base http://SERVER_IP:8000
```

Security model:

- `/runs` is not mounted publicly.
- Read endpoints require token when `WEBUI_AUTH_TOKEN` is set.
- Uploads are not exposed through predictable `/api/files/_uploads/...` paths.
- File previews/downloads use HMAC-signed file tokens.
- `WEBUI_FILE_TOKEN_SECRET` signs file tokens. If not set, a local `.webui_file_token_secret` is generated.
- Do not commit `.webui_file_token_secret`.

Recommended env:

```bash
export WEBUI_AUTH_TOKEN="secret"
export WEBUI_FILE_TOKEN_SECRET="separate-long-random-secret"
export WEBUI_MAX_UPLOAD_MB=32
export WEBUI_ALLOWED_PATHS="/path/to/data:/path/to/extra/outputs"
```

WebUI sample jobs run as subprocesses in normal repo mode. Stop endpoints are type-aware:

```text
/api/train/stop   -> train only
/api/sample/stop  -> sample only
/api/latents/stop -> latent_cache only
```

Sample metrics are saved to:

```text
webui_runs/<run_id>/metrics/sample.jsonl
```

Latent prepare metrics are saved to:

```text
webui_runs/<run_id>/metrics/latent_prepare.jsonl
```

### WebUI API highlights

```text
GET  /api/status
GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/logs/{stdout|stderr}?raw=true
GET  /api/runs/{run_id}/metrics
GET  /api/artifacts?run_id=...&source=all
GET  /api/generated-samples?run_id=...
GET  /api/train-samples?run_id=...
GET  /api/files/{relative-preview-path}
GET  /api/files/by-path/{signed-token}
GET  /api/files/download/{signed-token}
```

Artifacts have backend-generated URLs:

```json
{
  "path": "...",
  "name": "sample.png",
  "source": "webui_sample",
  "run_id": "...",
  "previewable": true,
  "url": "/api/files/...",
  "download_url": "/api/files/download/..."
}
```

Frontend must not parse filesystem paths or assume `webui_runs/` in URLs.

## 12. Distributed / Accelerate

Dry-run:

```bash
python -m train.cli --profile distributed_smoke --dry-run
```

Accelerate:

```bash
accelerate config
accelerate launch -m train.cli --config config/train_distributed_smoke.yaml
```

Checkpoint и events пишутся только rank 0. Метрики агрегируются через distributed context.

FSDP пока не включён в trainer. Есть template и документация:

```text
config/train_fsdp_template.yaml
docs/distributed_and_fsdp.md
```

Проверка template:

```bash
python -m train.cli --profile fsdp_template --dry-run
```

## 13. Тесты и проверки

Быстрый suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Полный локальный check:

```bash
bash scripts/check_project.sh
```

Frontend:

```bash
cd webui/frontend
npm ci
npm run build
cd ../..
```

Compile check:

```bash
python -m compileall -q .
```

## 14. Рекомендуемый порядок нового запуска

1. Проверить dataset:

```bash
python - <<'PY'
from pathlib import Path
root = Path('data/dataset')
print('images:', len(list((root/'images').glob('*'))))
print('metadata.jsonl:', (root/'metadata.jsonl').exists())
print('cache:', root/'.cache')
PY
```

2. Проверить config:

```bash
python -m train.cli --config config/train.yaml --dry-run
```

3. Подготовить text cache:

```bash
python -m scripts.prepare_text_cache --config config/train.yaml
```

4. Подготовить latent cache:

```bash
python -m scripts.prepare_latents --config config/train.yaml
```

5. Проверить cache:

```bash
python -m scripts.validate_cache --config config/train.yaml
```

6. Smoke train:

```bash
python -m train.cli --profile smoke
```

7. Overfit/dev/base:

```bash
python -m train.cli --profile overfit
python -m train.cli --profile dev
python -m train.cli --profile base
```

8. Sample checkpoint:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "test prompt" \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/test.png
```

9. Eval grids:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --seed 42
```

## 15. Типичные ошибки

### `Text cache missing key`

Dataset index изменился, а text cache старый:

```bash
rm -rf data/dataset/.cache/text
python -m scripts.prepare_text_cache --config config/train.yaml
```

### `text cache dataset_hash mismatch`

Изменились prompts/captions/tags или `text_field/text_fields`. Пересобери text cache.

### `latent cache manifest status is not complete`

Cache preparation был прерван. Удали partial cache или запусти repair/rebuild:

```bash
python -m scripts.prepare_training_cache --config config/train.yaml --repair --rebuild
```

### `Latent cache is stale` / `latent shape mismatch`

Изменился VAE, `image_size`, aspect buckets, `latent_downsample_factor`, latent dtype или dataset. Пересобери latents:

```bash
python -m scripts.prepare_latents --config config/train.yaml --overwrite
```

### `Checkpoint incompatible: ... mismatch`

Checkpoint создан для другой архитектуры/config. Используй совместимый config или другой checkpoint.

### `task=... does not allow ...`

Sampler API запрещает неявные mixed modes. Используй task/input matrix из раздела Sampling CLI.

### `latent_only outputs must use .pt or .pth extension`

Для `--latent-only` output должен быть `.pt` или `.pth`.

### `text_cache=false is only allowed when allow_on_the_fly_text=true`

Для real training text cache должен быть включён. Debug-only:

```yaml
text_cache: false
allow_on_the_fly_text: true
```

### CUDA OOM

Уменьшай:

```yaml
training:
  batch_size: 1
  grad_accum_steps: 32
model:
  gradient_checkpointing: true
```

Также можно использовать меньший профиль: `smoke`, `overfit`, `dev`, `milestone_a`.

## 16. Entry points

После `pip install -e .` доступны:

```bash
md-train --help
md-sample --help
md-prepare-latents --help
md-prepare-text-cache --help
md-prepare-training-cache --help
md-eval --help
md-cache-validate --help
md-webui --help
```

Module commands:

```bash
python -m train.cli --help
python -m sample.cli --help
python -m scripts.prepare_latents --help
python -m scripts.prepare_text_cache --help
python -m scripts.prepare_training_cache --help
python -m train.eval_cli --help
python -m scripts.validate_cache --help
python -m main --help
```

## 17. Runtime files that must not be committed

Do not commit generated data, cache, outputs or local secrets:

```text
.venv/
webui/frontend/node_modules/
webui/frontend/.vite/
.webui_file_token_secret
data/dataset/
samples/
runs/
webui_runs/
vae_sd_mse/
*.egg-info/
```

If `webui/frontend/dist` is intentionally shipped with a wheel/package, add it explicitly:

```bash
git add -f webui/frontend/dist
```
