
# md — Minimal Diffusion (DDPM / DDIM)

Минимальный, **чистый и полностью оффлайн** проект для обучения и сэмплинга изображений
на основе **DDPM (v-prediction) + DDIM**, оптимизированный под **8 GB VRAM**.

Проект предназначен для:
- обучения **с нуля** (без pretrained весов),
- исследования архитектуры и диффузии,
- дальнейшего расширения (LoRA, text-cond, ControlNet и т.д.).

---

## Возможности

- 🧠 **U-Net** с residual blocks и spatial self-attention  
- 🔁 **v-prediction** (как в современных diffusion-моделях)  
- ⚖️ **Min-SNR loss weighting** — стабильное обучение  
- 🌊 **DDIM sampling** (быстро и детерминированно)  
- 🧮 **EMA** для качественного инференса  
- 🚀 Оптимизирован под **8 GB VRAM** (AMP, SDPA, grad-checkpoint)  
- 📦 Полностью **оффлайн**, без внешних зависимостей и API  

---

## Структура проекта

md
├─ train.py            # обучение
├─ sample.py           # инференс (DDIM)
├─ sanity_check.py     # быстрый тест пайплайна
├─ train.yaml          # конфигурация обучения
├─ data/
│  └─ raw/             # датасет
├─ ddpm/
│  ├─ model.py         # U-Net + attention
│  ├─ diffusion.py     # DDPM (v-pred)
│  ├─ ddim.py          # DDIM sampler
│  ├─ data.py          # загрузка и препроцессинг
│  └─ utils.py         # EMA, ckpt, seed, metadata
└─ runs/
└─ ...              # чекпоинты и логи


---

## Установка

Минимальные зависимости:

```bash
pip install torch torchvision pyyaml pillow numpy tqdm
````

Рекомендуется:

* Python **3.10+**
* PyTorch **≥ 2.1**
* CUDA **12+**

---

## Подготовка датасета

Ожидаемая структура:


data/raw/
└─ people/
   ├─ train/
   │  ├─ 0001.jpg
   │  ├─ 0002.png
   │  └─ ...
   ├─ val/
   └─ test/


* изображения RGB,
* произвольные разрешения (будут приведены к `image_size`),
* значения нормализуются в `[-1, 1]`.

---

## Быстрая проверка (обязательно)

Перед полноценным обучением:

```bash
python sanity_check.py --image_size 64 --steps 2
```

Это проверит:

* датасет,
* модель,
* diffusion,
* backward,
* сохранение чекпоинта.

---

## Обучение

### Запуск

```bash
python train.py --config train.yaml
```

### Что происходит

* используется **v-prediction**
* loss = MSE(v̂, v) с **Min-SNR weighting**
* применяется **EMA**
* чекпоинты сохраняются в `runs/.../`

### Логи

В процессе обучения создаются:

* `runs/.../config.yaml` — snapshot конфига
* `runs/.../run_meta.yaml` — информация об окружении
* `runs/.../train_log.jsonl` — loss, скорость, VRAM, ETA

---

## Инференс (DDIM)

Пример:

```bash
python sample.py \
  --ckpt ./runs/ddpm_people_384/ckpt_final.pt \
  --out ./samples/grid.png \
  --n 8 \
  --steps 200
```

Рекомендации:

* `steps=50–200` — оптимально для проверки качества
* всегда используйте **EMA-веса** (по умолчанию включено)

---

## Конфигурация (`train.yaml`)

Ключевые параметры:

```yaml
image_size: 384
batch_size: 2
grad_accum_steps: 4     # эффективный batch = 8
amp: true
grad_checkpoint: true

timesteps: 1000
min_snr_gamma: 5.0

attn_resolutions: [48]
```

---

## Рекомендации под 8 GB VRAM

* запуск из **TTY** (без графической сессии),
* `batch_size ↓`, `grad_accum_steps ↑`,
* `amp: true`,
* `attn_resolutions: [48]` (не ниже),
* `PYTORCH_ALLOC_CONF=expandable_segments:True`.

---

## Частые проблемы

### Генерируется только шум

Проверь:

* что используется **EMA** при сэмплинге,
* корректную формулу Min-SNR для v-pred,
* что датасет не содержит паддинговых рамок,
* что loss действительно уменьшается.

### CUDA out of memory

* закрой все процессы на GPU (`nvidia-smi`),
* уменьши `num_workers`,
* снизь `batch_size`,
* включи `grad_checkpoint`.

---

## Web UI (опционально)

Backend:

```bash
uvicorn webui.backend.app:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd webui/frontend
npm install
npm run dev
```

---

## Лицензия

Проект предоставляется **как есть**, для исследований и обучения.

```
```
