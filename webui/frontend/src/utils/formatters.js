const LABELS = {
  ckpt: "Модель (Checkpoint)",
  prompt: "Промпт",
  neg_prompt: "Негативный промпт",
  cfg: "Guidance scale",
  guidance: "Guidance scale",
  steps: "Шаги",
  n: "Количество изображений",
  seed: "Seed",
  width: "Ширина",
  height: "Высота",
  sampler: "Sampler",
  task: "Задача",
  shift: "Sampling shift",
  strength: "Img2img strength",
  mask: "Inpaint mask",
  "init-image": "Init image",
  init_image: "Init image",
  "control-image": "Control image",
  control_image: "Control image",
  "control-strength": "Control strength",
  control_strength: "Control strength",
  "control-type": "Control type",
  control_type: "Control type",
  "latent-only": "Latent-only",
  latent_only: "Latent-only",
  "fake-vae": "Fake VAE",
  fake_vae: "Fake VAE",
  "use-ema": "Use EMA",
  use_ema: "Use EMA",
  device: "Устройство",
  out: "Папка вывода",
  config: "Конфиг",
  overwrite: "Перезаписать кэш",
  limit: "Лимит файлов",
  "batch-size": "Размер батча",
  batch_size: "Размер батча",
  "num-workers": "Workers",
  num_workers: "Workers",
  "prefetch-factor": "Prefetch factor",
  prefetch_factor: "Prefetch factor",
  "pin-memory": "Pin memory",
  pin_memory: "Pin memory",
  "latent-dtype": "Latent dtype",
  latent_dtype: "Latent dtype",
  "autocast-dtype": "Autocast dtype",
  autocast_dtype: "Autocast dtype",
  "queue-size": "Размер очереди",
  queue_size: "Размер очереди",
  "writer-threads": "Writer threads",
  writer_threads: "Writer threads",
  "shard-size": "Размер shard",
  shard_size: "Размер shard",
  "stats-every-sec": "Интервал статистики",
  stats_every_sec: "Интервал статистики",
  "decode-backend": "Decode backend",
  decode_backend: "Decode backend",
};

const RUN_TYPES = {
  train: "Training",
  sample: "Generation",
  latent_cache: "Latent cache",
};

const MONTHS = [
  "янв",
  "фев",
  "мар",
  "апр",
  "мая",
  "июн",
  "июл",
  "авг",
  "сен",
  "окт",
  "ноя",
  "дек",
];

export function argLabel(name) {
  if (LABELS[name]) return LABELS[name];
  return String(name)
    .replace(/^--?/, "")
    .replace(/[-_]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function argTooltip(spec) {
  const flags = spec.flags?.join(", ");
  return [flags, spec.help].filter(Boolean).join("\n");
}

export function formatRunType(type) {
  return RUN_TYPES[type] || type || "-";
}

export function parseRunDate(run) {
  const direct = run?.created_at ? new Date(run.created_at) : null;
  if (direct && !Number.isNaN(direct.getTime())) return direct;

  const match = String(run?.run_id || "").match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match;
  const parsed = new Date(
    Number(year),
    Number(month) - 1,
    Number(day),
    Number(hour),
    Number(minute),
    Number(second)
  );
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

export function formatDate(value) {
  if (!value) return "-";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return `${date.getDate()} ${MONTHS[date.getMonth()]}, ${String(date.getHours()).padStart(2, "0")}:${String(
    date.getMinutes()
  ).padStart(2, "0")}`;
}

export function formatRelativeDate(value) {
  if (!value) return "-";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  const diffMs = Date.now() - date.getTime();
  const future = diffMs < 0;
  const abs = Math.abs(diffMs);
  const minute = 60 * 1000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (abs < minute) return future ? "скоро" : "только что";
  if (abs < hour) {
    const value = Math.round(abs / minute);
    return future ? `через ${value} мин` : `${value} мин назад`;
  }
  if (abs < day) {
    const value = Math.round(abs / hour);
    return future ? `через ${value} ч` : `${value} ч назад`;
  }
  return formatDate(date);
}

export function isToday(value) {
  if (!value) return false;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return false;
  const now = new Date();
  return (
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  );
}

export function formatRunId(runId) {
  const value = String(runId || "");
  const match = value.match(/^(\d{8}_\d{6})_([a-f0-9]+)$/i);
  if (match) return `Run ${match[2]}`;
  if (value.length > 12) return `Run ${value.slice(-8)}`;
  return value || "-";
}

export function lastUsefulLogLine(content) {
  return String(content || "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .reverse()
    .find((line) => !line.startsWith("{")) || "";
}
