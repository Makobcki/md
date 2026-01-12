# md — минимальный diffusion-проект (DDPM/DDIM, v-pred)

Проект для обучения и сэмплинга изображений с условием по тексту (BPE-токенайзер),
ориентирован на понятный код и локальный запуск. Включает CLI и WebUI.

## Установка

### Python окружение

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### WebUI (backend + frontend)

Backend зависимости:

```bash
pip install -r requirements-web.txt
```

Frontend зависимости:

```bash
cd webui/frontend
npm install
```

## Обучение

### Запуск

```bash
python scripts/train.py --config config/train.yaml
```

### Возобновление

```bash
python scripts/train.py --config config/train.yaml --resume ./runs/.../ckpt_stop_0000500.pt
```

### Важные параметры

- `config/train.yaml` — основной конфиг.
- `log_every` — период логирования. **Loss в логах и WebUI — среднее за последние `log_every` шагов.**
- `compile` — включает `torch.compile` (если доступен).
- `resume_ckpt` — путь к чекпоинту для резюма (CLI-флаг `--resume` переопределяет, поддерживает `latest` и номер шага).
- `ckpt_keep_last` — сколько последних чекпоинтов `ckpt_*.pt` хранить (CLI-флаг `--ckpt-keep-last` переопределяет).

### Выходные файлы

В `out_dir` (из `config/train.yaml`) создаются:

- `config_snapshot.yaml` — снимок конфига запуска.
- `run_meta.yaml` — информация об окружении.
- `metrics/events.jsonl` — события и метрики.
- `ckpt_*.pt`, `ckpt_stop_*.pt`, `ckpt_final.pt` — чекпоинты.

### Совместимость чекпоинтов и `torch.compile`

Чекпоинты, сохранённые в режиме `torch.compile`, совместимы с запуском без компиляции и наоборот.
Если структура модели не совпадает, загрузка завершится с понятной ошибкой о несоответствии ключей.

## Сэмплинг

### Запуск

```bash
python scripts/sample.py \
  --ckpt ./runs/.../ckpt_final.pt \
  --out ./samples/grid.png \
  --n 4 \
  --steps 50 \
  --prompt "1girl" \
  --cfg 5 \
  --sampler heun \
  --seed 42
```

### Важные параметры

- `--steps N` означает **ровно N итераций** внешнего цикла для всех самплеров.
- `--sampler`: `ddim`, `ddpm`, `euler`, `heun`, `dpm_solver`.
- `--cfg` — classifier-free guidance (1.0 = без усиления).
- `--seed` — фиксирует детерминированность.
- `--out` — путь к выходному изображению; директория создаётся автоматически.

## WebUI

### Запуск backend

```bash
python main.py --host 127.0.0.1 --port 8000
```

### Запуск frontend

```bash
cd webui/frontend
npm run dev
```

### Логи и метрики WebUI

Для каждого запуска создаётся директория:

```
webui_runs/<run_id>/
  logs/
    train.log
    train.err.log
    sample.log
    sample.err.log
  metrics/
    train_metrics.jsonl
    events.jsonl
  samples/
```

Метрика `loss` в UI отображается как среднее значение за последний интервал `log_every`.

## Профилирование и телеметрия

### Встроенные тайминги

В `scripts/train.py` в `metric`-событиях публикуются агрегированные тайминги секций (CPU и GPU):
`data_fetch`, `fwd_bwd`, `opt_step`, `step_total`.
Используйте `metrics/events.jsonl` для анализа.

### CPU профилирование (cProfile)

```bash
python -m cProfile -o /tmp/train.prof scripts/train.py --config config/train.yaml
python -m pstats /tmp/train.prof
```

### py-spy (если установлена)

```bash
py-spy record -o /tmp/train.svg -- python scripts/train.py --config config/train.yaml
```

## Примечания и troubleshooting

- **torch.compile**: при первом запуске возможна длительная компиляция и больший расход памяти.
- **Недостаток VRAM**: уменьшите `batch_size`, увеличьте `grad_accum_steps`, включите `amp`.
- **Сэмплинг в 0 шагов**: CLI ожидает `--steps >= 1`, иначе шаги не выполняются.
- **Чекпоинт не грузится**: проверьте, что архитектура (из `config_snapshot.yaml`) соответствует текущему конфигу.
