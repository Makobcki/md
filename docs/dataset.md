# Dataset format

The training dataset consists of images and metadata.

---

## Minimal layout

```text
data/dataset/
  images/
    <hash>.png
    <hash>.jpg
  metadata.jsonl
```

Each line in `metadata.jsonl` must be a JSON object.

Minimal row:

```json
{ "md5": "hash", "file_name": "hash.png", "prompt": "a cat on a table" }
```

---

## Supported image fields

The image path field may be named:

- `file_name`
- `filename`
- `image`
- `img`
- `path`
- `image_path`

If no image field is present, the loader searches for:

```text
images/<md5>.*
```

---

## Sidecar metadata files

Per-image JSON files are supported through `meta_dir`.

Example layout:

```text
data/dataset/
  images/
    hash.png
  meta/
    hash.json
```

Example `meta/hash.json`:

```json
{
  "md5": "hash",
  "file_name": "hash.png",
  "prompt": "a cat on a table",
  "caption": "cat sitting on a wooden table",
  "tags": ["cat", "table"]
}
```

Config:

```yaml
data_root: ./data/dataset
image_dir: images
meta_dir: meta
```

---

## Prompt-first training

For training on `prompt` fields, use:

```yaml
text_field: prompt
text_fields:
  - prompt
  - caption
  - text
```

Nested config variant:

```yaml
dataset:
  text_field: prompt
  text_fields:
    - prompt
    - caption
    - text
```

Use only the variant supported by your current config schema. Do not define both unless the loader explicitly supports both.

`caption_field` may remain supported for older caption-first configs, but new prompt-first runs should prefer `text_field` and `text_fields`.

The dataset index stores resolved text as:

```json
{
  "text": "resolved training text",
  "text_source": "prompt",
  "caption": "resolved training text"
}
```

The `caption` field is filled with the same resolved text for backward compatibility.

---

## Prompts stored as `prompts/<hash>.txt`

If the dataset is structured like this:

```text
data/dataset/
  images/
    hash.png
  prompts/
    hash.txt
```

convert prompt sidecars to `metadata.jsonl` once:

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

out.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    encoding="utf-8",
)

print(f"wrote {len(rows)} rows to {out}")
PY
```

Recommended config after conversion:

```yaml
data_root: ./data/dataset
image_dir: images
meta_dir: ""
tags_dir: ""

text_field: prompt
text_fields: [prompt, caption, text]
```

Use `tags_dir: prompts` only when those `.txt` files are intended as tag sidecars. For exact prompt training, prefer `metadata.jsonl` with a `prompt` field.

---

## Dataset check

Basic check:

```bash
python - <<'PY'
from pathlib import Path

root = Path("data/dataset")
images = root / "images"
metadata = root / "metadata.jsonl"

print("root:", root.resolve())
print("images exists:", images.exists())
print("images count:", len([p for p in images.glob("*") if p.is_file()]))
print("metadata.jsonl exists:", metadata.exists())
print("cache dir:", root / ".cache")
PY
```

---

## Recommended config for `data/dataset`

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

If the current config points to:

```yaml
data_root: ./data/dataset/pixso_512
```

then either move the dataset there or change `data_root` to the actual dataset path.
