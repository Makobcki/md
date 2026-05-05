# WebUI

The WebUI provides a frontend and backend for sampling.

---

## Install WebUI dependencies

```bash
python -m pip install -e ".[web,ml]"

cd webui/frontend
npm install
cd ../..
```

---

## Run backend + frontend

```bash
python -m main \
  --host 127.0.0.1 \
  --port 8000 \
  --frontend \
  --frontend-host 127.0.0.1 \
  --frontend-port 5173
```

Frontend URL:

```text
http://127.0.0.1:5173
```

---

## Backend only

```bash
python -m main \
  --host 127.0.0.1 \
  --port 8000 \
  --no-frontend
```

---

## Sampling implementation

The WebUI sampling path calls:

```text
sample.api.run_sample
```

It does not call `sample.cli` as a subprocess.

---

## Supported WebUI fields

Supported task fields:

- `task`: `txt2img`, `img2img`, `inpaint`, `control`;
- `sampler`: `flow_euler`, `flow_heun`;
- `steps`;
- `cfg`;
- `shift`;
- `seed`;
- `n`;
- `prompt`;
- `neg_prompt`;
- `init-image`;
- `strength`;
- `mask`;
- `control-image`;
- `control-strength`;
- `latent-only`;
- `fake-vae`;
- `use-ema`.

---

## Path restrictions

By default, WebUI restricts file access to:

- repository root;
- run directory;
- configured `out_dir`.

Allow extra paths:

```bash
export WEBUI_ALLOWED_PATHS="/path/to/data:/path/to/samples"
```

Use absolute paths.

---

## Authentication token

Set a token:

```bash
export WEBUI_AUTH_TOKEN="secret"
```

Do not expose WebUI to the public internet without authentication.

---

## Common WebUI command

```bash
export WEBUI_AUTH_TOKEN="secret"
export WEBUI_ALLOWED_PATHS="/path/to/data:/path/to/samples"

python -m main \
  --host 127.0.0.1 \
  --port 8000 \
  --frontend \
  --frontend-host 127.0.0.1 \
  --frontend-port 5173
```
