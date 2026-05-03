export const METRIC_EVENT_TYPES = new Set(["metric", "progress", "train", "eval", "sample"]);

export function isMetricEvent(event) {
  return event && METRIC_EVENT_TYPES.has(event.type);
}

export function metricLoss(event) {
  if (!event) return undefined;
  if (Number.isFinite(event.loss)) return event.loss;
  if (Number.isFinite(event.train_loss)) return event.train_loss;
  if (Number.isFinite(event.val_loss)) return event.val_loss;
  return undefined;
}

export function metricThroughput(event) {
  if (!event) return undefined;
  if (Number.isFinite(event.samples_per_sec)) return event.samples_per_sec;
  if (Number.isFinite(event.img_per_sec)) return event.img_per_sec;
  if (Number.isFinite(event.items_per_sec)) return event.items_per_sec;
  return undefined;
}

export function metricStep(event) {
  if (!event) return undefined;
  if (Number.isFinite(event.step)) return event.step;
  if (Number.isFinite(event.global_step)) return event.global_step;
  if (Number.isFinite(event.processed)) return event.processed;
  return undefined;
}

export function metricMaxSteps(event) {
  if (!event) return undefined;
  if (Number.isFinite(event.max_steps)) return event.max_steps;
  if (Number.isFinite(event.total_steps)) return event.total_steps;
  return undefined;
}

export function formatStep(event) {
  const step = metricStep(event);
  if (!Number.isFinite(step)) return "—";
  const maxSteps = metricMaxSteps(event);
  if (Number.isFinite(maxSteps) && maxSteps > 0) {
    return `${step}/${maxSteps}`;
  }
  return String(step);
}

export function metricKey(event) {
  if (!event) return "";
  const step = metricStep(event);
  const loss = metricLoss(event);
  const valLoss = Number.isFinite(event.val_loss) ? event.val_loss : "";
  const saved = event.saved ?? "";
  return [event.type || "event", Number.isFinite(step) ? step : "", Number.isFinite(loss) ? loss : "", valLoss, saved, event.path || ""].join(":");
}

export function mergeMetricEvents(prev, next, maxItems = 1000) {
  const out = [];
  const seen = new Set();
  for (const event of [...(prev || []), ...(next || [])]) {
    if (!isMetricEvent(event)) continue;
    const key = metricKey(event);
    if (key && seen.has(key)) continue;
    if (key) seen.add(key);
    out.push(event);
  }
  return out.slice(-maxItems);
}

export function metricChartData(events) {
  return (events || [])
    .map((event) => ({ step: metricStep(event), loss: metricLoss(event), type: event.type }))
    .filter((item) => Number.isFinite(item.step) && Number.isFinite(item.loss));
}

export function latestMetricWithLoss(events) {
  const items = events || [];
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (Number.isFinite(metricLoss(items[i])) || Number.isFinite(metricStep(items[i]))) {
      return items[i];
    }
  }
  return null;
}


export function metricSecondsPerStep(event) {
  if (!event) return undefined;
  if (Number.isFinite(event.sec_per_step)) return event.sec_per_step;
  if (Number.isFinite(event.s_per_step)) return event.s_per_step;
  if (Number.isFinite(event.steps_per_sec) && event.steps_per_sec > 0) {
    return 1.0 / event.steps_per_sec;
  }
  return undefined;
}

export function formatDurationSeconds(value) {
  if (!Number.isFinite(value)) return "—";
  const total = Math.max(0, Math.round(value));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours > 0) {
    return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  }
  return `${seconds}s`;
}

export function metricElapsed(event) {
  if (!event) return "—";
  if (typeof event.elapsed === "string" && event.elapsed.length > 0) return event.elapsed;
  return formatDurationSeconds(event.elapsed_sec);
}

export function metricEta(event) {
  if (!event) return "—";
  if (typeof event.eta === "string" && event.eta.length > 0) return event.eta;
  return formatDurationSeconds(event.eta_sec);
}
