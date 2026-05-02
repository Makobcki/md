import React from "react";
import { argLabel, argTooltip } from "../utils/formatters.js";

const SLIDER_LIMITS = {
  steps: { min: 1, max: 500, step: 1 },
  n: { min: 1, max: 64, step: 1 },
  width: { min: 64, max: 2048, step: 64 },
  height: { min: 64, max: 2048, step: 64 },
  cfg: { min: 0, max: 30, step: 0.1 },
  guidance: { min: 0, max: 30, step: 0.1 },
  seed: { min: 0, max: 9999, step: 1 },
  limit: { min: 1, max: 100000, step: 100 },
  "batch-size": { min: 1, max: 512, step: 1 },
  batch_size: { min: 1, max: 512, step: 1 },
  "num-workers": { min: 0, max: 64, step: 1 },
  num_workers: { min: 0, max: 64, step: 1 },
  "prefetch-factor": { min: 1, max: 32, step: 1 },
  prefetch_factor: { min: 1, max: 32, step: 1 },
  "queue-size": { min: 1, max: 4096, step: 16 },
  queue_size: { min: 1, max: 4096, step: 16 },
  "writer-threads": { min: 1, max: 16, step: 1 },
  writer_threads: { min: 1, max: 16, step: 1 },
  "shard-size": { min: 0, max: 16384, step: 256 },
  shard_size: { min: 0, max: 16384, step: 256 },
  "stats-every-sec": { min: 0.5, max: 3600, step: 0.5 },
  stats_every_sec: { min: 0.5, max: 3600, step: 0.5 },
};

function sliderLimits(spec) {
  if (SLIDER_LIMITS[spec.name]) return SLIDER_LIMITS[spec.name];
  if (spec.type === "int") return { min: 0, max: 100, step: 1 };
  if (spec.type === "float") return { min: 0, max: 100, step: 0.1 };
  return null;
}

function fileLabel(value) {
  return String(value || "").split(/[\\/]/).pop() || value;
}

export default function ArgField({
  spec,
  value,
  onChange,
  checkpoints = [],
  disabled = false,
  variant = "card",
  multiline = false,
  rows = 3,
}) {
  const title = argTooltip(spec);
  const label = argLabel(spec.name);
  const choices = spec.name === "device" ? ["auto", "cuda", "cpu"] : spec.choices;

  const setValue = (next) => onChange(spec.name, next);

  let control;
  if (spec.name === "ckpt" && checkpoints.length > 0) {
    control = (
      <select value={value || ""} onChange={(event) => setValue(event.target.value)} disabled={disabled}>
        {checkpoints.map((item) => (
          <option key={item} value={item}>
            {fileLabel(item)}
          </option>
        ))}
      </select>
    );
  } else if (choices && choices.length > 0) {
    control = (
      <select value={value || ""} onChange={(event) => setValue(event.target.value)} disabled={disabled}>
        {choices.map((choice) => (
          <option key={choice} value={choice}>
            {choice}
          </option>
        ))}
      </select>
    );
  } else if (spec.type === "bool" || typeof spec.default === "boolean") {
    control = (
      <label className="switch-row">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => setValue(event.target.checked)}
          disabled={disabled}
        />
        <span>{Boolean(value) ? "Включено" : "Выключено"}</span>
      </label>
    );
  } else if (spec.type === "int" || spec.type === "float") {
    const limits = sliderLimits(spec);
    const numeric = Number(value);
    const sliderValue = Number.isFinite(numeric)
      ? Math.min(limits.max, Math.max(limits.min, numeric))
      : spec.default ?? limits.min;
    const slider = (
      <input
        type="range"
        min={limits.min}
        max={limits.max}
        step={limits.step}
        value={sliderValue}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
      />
    );
    control = ["steps", "n"].includes(spec.name) ? (
      <div className="number-control slider-only">
        {slider}
        <span>{value ?? sliderValue}</span>
      </div>
    ) : (
      <div className="number-control">
        {slider}
        <input
          type="number"
          value={value ?? ""}
          min={limits.min}
          max={limits.max}
          step={limits.step}
          onChange={(event) => setValue(event.target.value)}
          disabled={disabled}
        />
      </div>
    );
  } else {
    control = multiline ? (
      <textarea
        value={value ?? ""}
        rows={rows}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
      />
    ) : (
      <input
        type="text"
        value={value ?? ""}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
      />
    );
  }

  return (
    <div className={`${variant === "flat" ? "field-row" : "card soft field-card"}`}>
      <label title={title}>
        {label}
        {title && <span className="info-dot">i</span>}
      </label>
      {control}
      {spec.help && <div className="muted field-help">{spec.help}</div>}
    </div>
  );
}
