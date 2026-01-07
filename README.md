# md
diffusion model
Минимальный DDPM/DDIM проект для обучения и сэмплинга изображений.

## Структура
- `train.py` — точка входа для обучения.
- `sample.py` — точка входа для инференса (DDIM).
- `ddpm/` — логика модели/диффузии/датасета.
- `train.yaml` — конфиг обучения.

## Установка
```bash
pip install torch torchvision pyyaml pillow numpy tqdm
```

## Запуск
**Обучение (одна команда):**
```bash
python train.py --config train.yaml
```

**Инференс (одна команда):**
```bash
python sample.py --ckpt ./runs/ddpm_people_384/ckpt_final.pt --out ./samples/grid.png --n 8 --steps 200
```

## Логи и метрики
После старта обучения создаётся:
- `runs/.../config.yaml` — сохранённый конфиг,
- `runs/.../run_meta.yaml` — инфо об окружении,
- `runs/.../train_log.jsonl` — логи с `loss`, `img_per_sec`, `peak_mem_mb`.

## Web UI
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt

# backend
uvicorn webui.backend.app:app --host 127.0.0.1 --port 8000
```

В другом терминале:
```bash
cd webui/frontend
npm i
npm run dev
```
