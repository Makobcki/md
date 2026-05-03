# MMDiT Rectified Flow image model

Проект обучает и запускает **latent MMDiT Rectified Flow** модель для image generation. Поддерживаемая линия модели — **MMDiT RF-only**: latent space, frozen CLIP/T5 text encoders, joint text/image attention, flow-matching objective, Flow Euler/Heun sampling. Старый U-Net/DDPM/DDIM/DPM workflow не используется.

## 1. Что нужно установить

Рекомендуемое окружение:

- Linux;
- Python 3.11+;
- NVIDIA GPU для реального обучения/инференса;
- CUDA-сборка PyTorch, подходящая под драйвер;
- `git`, `python-venv`, `build-essential`;
- `npm` только для WebUI frontend.

Создание окружения:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Установка PyTorch под свою CUDA/CPU сборку. Пример для CUDA-сборки нужно брать под текущий драйвер с сайта PyTorch. После этого установи проект:

```bash
python -m pip install -e ".[all]"
```

Минимальная установка без WebUI/dev extras:

```bash
python -m pip install -e ".[ml]"
```

Для WebUI:

```bash
python -m pip install -e ".[web,ml]"
cd webui/frontend
npm install
cd ../..
```

Для тестов:

```bash
python -m pip install -e ".[all,dev]"
```

Если text encoders скачиваются с HuggingFace, желательно задать token:

```bash
export HF_TOKEN=...
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

Поддерживаемые conditioning modes:

- `txt2img`: target noisy latent + text tokens;
- `img2img`: target noisy latent + source latent tokens + text tokens;
- `inpaint`: target noisy latent + source latent tokens + mask tokens + text tokens;
- `control`: control latent token stream для дальнейшего ControlNet/IP-Adapter-style развития.

## 3. Структура датасета

Минимальная структура:

```text
data/dataset/
  images/
    <hash>.png
    <hash>.jpg
  metadata.jsonl
```

Каждая строка `metadata.jsonl` — JSON object. Минимально:

```json
{ "md5": "hash", "file_name": "hash.png", "prompt": "a cat on a table" }
```

Также поддерживаются per-image JSON files через `meta_dir`:

```text
data/dataset/
  images/hash.png
  meta/hash.json
```

Пример `meta/hash.json`:

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

### 3.1. Обучение на prompt вместо caption

Для обучения именно на prompt-полях:

```yaml
text_field: prompt
text_fields:
  - prompt
  - caption
  - text
```

или nested-вариант:

```yaml
dataset:
  text_field: prompt
  text_fields:
    - prompt
    - caption
    - text
```

`caption_field` остаётся совместимым со старым caption-flow, но для новых запусков с prompt-first лучше использовать `text_field/text_fields`.

Dataset index сохраняет:

```json
{
  "text": "resolved training text",
  "text_source": "prompt",
  "caption": "resolved training text"
}
```

`caption` заполняется тем же текстом для обратной совместимости.

### 3.2. Если промпты лежат в `prompts/<hash>.txt`

Если структура такая:

```text
data/dataset/
  images/hash.png
  prompts/hash.txt
```

лучше один раз собрать `metadata.jsonl` из sidecar-файлов:

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
    rows.append({
        "md5": img.stem,
        "file_name": img.name,
        "prompt": prompt,
    })

out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
print(f"wrote {len(rows)} rows to {out}")
PY
```

После этого в config указать:

```yaml
data_root: ./data/dataset
image_dir: images
meta_dir: ""
tags_dir: ""
text_field: prompt
text_fields: [prompt, caption, text]
```

`tags_dir: prompts` можно использовать только если эти `.txt` нужны как tags sidecars, но для exact prompt training правильнее `metadata.jsonl` с полем `prompt`.

## 4. VAE и text encoders

Для реального latent training нужен VAE, совместимый с `diffusers.AutoencoderKL.from_pretrained`:

```yaml
vae:
  pretrained: ./vae_sd_mse
  freeze: true
  scaling_factor: 0.18215
```

`./vae_sd_mse` должен быть локальной папкой diffusers VAE или именем модели HuggingFace.

Text encoders задаются в config:

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

Для тестов и smoke sample есть fake/latent-only пути, но real training использует подготовленные text/latent caches.

## 5. Config для твоей структуры `data/dataset`

Если данные лежат так:

```text
data/dataset/
  images/<hash>.png
  metadata.jsonl
  prompts/<hash>.txt   # optional, если уже сконвертировано в metadata.jsonl, можно не использовать
```

то в `config/train.yaml` или отдельном config указать:

```yaml
architecture: mmdit_rf
mode: latent
image_size: 512
latent_channels: 4
latent_downsample_factor: 8
latent_patch_size: 2

objective: rectified_flow
prediction_type: flow_velocity

data_root: ./data/dataset
image_dir: images
meta_dir: ""
tags_dir: ""
text_field: prompt
text_fields: [prompt, caption, text]
require_512: true
min_tag_count: 0

cache:
  latent_cache: true
  text_cache: true
  auto_prepare: true
  validate_on_start: true
  strict: true
  rebuild_if_stale: false
  sharded: true
  dtype: bf16
```

Если текущий config использует `data_root: ./data/dataset/pixso_512`, то либо переместить dataset туда, либо поменяй `data_root` на `path/to/dataset`.

## 6. Проверка проекта

Быстрые тесты:

```bash
pytest -q
```

Локальная CI-style проверка:

```bash
bash scripts/check_project.sh
```

Dry-run профили без запуска обучения:

```bash
python -m train.cli --profile smoke --dry-run
python -m train.cli --profile overfit --dry-run
python -m train.cli --profile dev --dry-run
python -m train.cli --profile base --dry-run
python -m train.cli --profile milestone_a --dry-run
python -m train.cli --profile milestone_b --dry-run
python -m train.cli --profile milestone_c --dry-run
```

## 7. Подготовка cache

Training умеет auto-prepare, если в config:

```yaml
cache:
  auto_prepare: true
```

Но для больших датасетов лучше подготовить cache вручную.

### 7.1. Text cache

```bash
python -m scripts.prepare_text_cache --config config/train.yaml
```

Будет создано:

```text
data/dataset/.cache/text/
  metadata.json
  manifest.json
  index.jsonl
  empty_prompt.safetensors
  shards/text_00000.safetensors
  ...
```

Проверить text cache:

```bash
python -m scripts.validate_cache --config config/train.yaml --text
```

### 7.2. Latent cache

```bash
python -m scripts.prepare_latents --config config/train.yaml
```

Будет создано:

```text
data/dataset/.cache/latents/
  metadata.json
  shard_index.jsonl
  shards/*.pt или *.safetensors
```

### 7.3. Unified training cache

Одна команда для text + latents + manifest:

```bash
python -m scripts.prepare_training_cache --config config/train.yaml
```

Repair/rebuild modes:

```bash
python -m scripts.prepare_training_cache --config config/train.yaml --repair
python -m scripts.prepare_training_cache --config config/train.yaml --repair --force
python -m scripts.prepare_training_cache --config config/train.yaml --repair --rebuild
```

Если переключаешься с captions на prompts, пересобери text cache:

```bash
rm -rf data/dataset/.cache/text
python -m scripts.prepare_text_cache --config config/train.yaml
```

Если меняешь VAE, image size, latent dtype или latent shape, пересобери latent cache:

```bash
rm -rf data/dataset/.cache/latents
python -m scripts.prepare_latents --config config/train.yaml
```

## 8. Обучение

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

Явный config:

```bash
python -m train.cli --config config/train.yaml
```

Resume:

```bash
python -m train.cli --config config/train.yaml --resume runs/.../checkpoints/latest.pt
```

Результаты пишутся в run directory вида:

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

Train events содержат:

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

Validation events:

```json
{
  "type": "eval",
  "step": 1000,
  "val_loss": 0.101,
  "val_loss_t_bin_00_01": 0.09
}
```

## 9. Sampling CLI

Text-to-image:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "1girl, blue hair, white dress" \
  --neg_prompt "low quality, blurry" \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --shift 3.0 \
  --seed 42 \
  --out samples/txt2img.png
```

Euler sampler:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "a cat on a table" \
  --sampler flow_euler \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/euler.png
```

Image-to-image:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task img2img \
  --prompt "same character, winter outfit" \
  --init-image input.png \
  --strength 0.55 \
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
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
  --sampler flow_heun \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/inpaint.png
```

Control smoke path:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --task control \
  --prompt "edge guided cat" \
  --control-image control.png \
  --control-strength 0.75 \
  --sampler flow_heun \
  --steps 28 \
  --out samples/control.png
```

EMA flags:

```bash
python -m sample.cli --ckpt runs/.../checkpoints/latest.pt --prompt "a cat" --use-ema --out samples/ema.png
python -m sample.cli --ckpt runs/.../checkpoints/latest.pt --prompt "a cat" --no-ema --out samples/raw.png
```

Latent-only smoke sample без VAE decode:

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

Fake VAE smoke sample:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "smoke" \
  --fake-vae \
  --device cpu \
  --steps 2 \
  --n 1 \
  --out samples/fake.png
```

Рядом с output всегда пишется JSON metadata sidecar:

```text
samples/txt2img.png
samples/txt2img.json
```

Metadata содержит checkpoint path/step, prompt, negative prompt, sampler, steps, cfg, seed, shift, latent shape, model config, VAE config и text encoder config.

## 10. Evaluation

Посмотреть prompt bank:

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

CFG sweep:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --cfg-sweep 1.0 2.5 4.5 7.0
```

Step/sampler/shift sweep:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --step-sweep 8 16 28 40 \
  --sampler-sweep flow_euler flow_heun \
  --shift-sweep 1.0 2.0 3.0 4.0
```

Resolution eval:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --resolution 768
```

Output layout:

```text
runs/.../eval/eval_768/step_000100/
  core_grid.png
  metadata.json
  events.jsonl
```

## 11. WebUI

Backend + frontend:

```bash
python -m main --host 127.0.0.1 --port 8000 --frontend --frontend-host 127.0.0.1 --frontend-port 5173
```

fontend:

```text
http://127.0.0.1:5173
```

Backend only:

```bash
python -m main --host 127.0.0.1 --port 8000 --no-frontend
```

WebUI sample path напрямую использует `sample.api.run_sample`, а не subprocess `sample.cli`. Поддерживаются поля:

- `task`: `txt2img`, `img2img`, `inpaint`, `control`;
- `sampler`: `flow_euler`, `flow_heun`;
- `steps`, `cfg`, `shift`, `seed`, `n`;
- `prompt`, `neg_prompt`;
- `init-image`, `strength`;
- `mask`;
- `control-image`, `control-strength`;
- `latent-only`, `fake-vae`, `use-ema`.

По умолчанию WebUI ограничивает пути repo root, run dir и configured `out_dir`. Для дополнительных директорий:

```bash
export WEBUI_ALLOWED_PATHS="/path/to/data:/path/to/samples"
```

Если нужен token:

```bash
export WEBUI_AUTH_TOKEN="secret"
```

## 12. Distributed / Accelerate

Dry-run:

```bash
python -m train.cli --profile distributed_smoke --dry-run
```

Реальный запуск через Accelerate:

```bash
accelerate config
accelerate launch -m train.cli --config config/train_distributed_smoke.yaml
```

Checkpoint и events пишутся только rank 0. Метрики агрегируются через distributed context.

FSDP пока намеренно не включён в trainer. Есть template и документация:

```text
config/train_fsdp_template.yaml
docs/distributed_and_fsdp.md
```

Проверка template:

```bash
python -m train.cli --profile fsdp_template --dry-run
```

## 13. Рекомендуемый порядок нового запуска

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

7. Overfit:

```bash
python -m train.cli --profile overfit
```

8. Dev/base training:

```bash
python -m train.cli --profile dev
python -m train.cli --profile base
```

9. Sample checkpoint:

```bash
python -m sample.cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --prompt "test prompt" \
  --steps 28 \
  --cfg 4.5 \
  --seed 42 \
  --out samples/test.png
```

10. Eval grids:

```bash
python -m train.eval_cli \
  --ckpt runs/.../checkpoints/latest.pt \
  --out-dir runs/... \
  --prompt-set core \
  --seed 42
```

## 14. Типичные ошибки

### `Text cache missing key`

Dataset index изменился, а text cache старый. Пересборка:

```bash
rm -rf data/dataset/.cache/text
python -m scripts.prepare_text_cache --config config/train.yaml
```

### `text cache dataset_hash mismatch`

Изменились prompts/captions/tags или `text_field/text_fields`. Пересборка text cache.

### `Latent cache is stale` или `latent shape mismatch`

Изменился VAE, `image_size`, `latent_downsample_factor`, `latent_patch_size`, dtype или dataset. Пересборка latents:

```bash
python -m scripts.prepare_latents --config config/train.yaml --overwrite
```

### `Checkpoint incompatible: hidden_dim mismatch`

Checkpoint создан для другой архитектуры/config. Используй совместимый config или другой checkpoint.

### `task=inpaint requires --mask`

Для inpaint обязательно нужны `--init-image` и `--mask`.

### `task=img2img requires --init-image`

Для img2img нужен `--init-image`.

### `text_cache=false is only allowed when allow_on_the_fly_text=true`

Для real training text cache должен быть включён. Debug-only режим:

```yaml
text_cache: false
allow_on_the_fly_text: true
```

### CUDA OOM

Уменьшать:

```yaml
training:
  batch_size: 1
  grad_accum_steps: 32
model:
  gradient_checkpointing: true
```

Также можно использовать меньший профиль: `smoke`, `overfit`, `dev`, `milestone_a`.

## 15. Полезные entry points

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

Эквивалентные module commands:

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
