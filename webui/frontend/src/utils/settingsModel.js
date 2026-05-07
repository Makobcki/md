export const SAMPLE_SETTINGS_KEY = "md-diffusion:webui:sample-settings";
export const LATENT_SETTINGS_KEY = "md-diffusion:webui:latent-settings";

const EMPTY_OBJECT = Object.freeze({});

function safeStorage() {
  if (typeof window === "undefined" || !window.localStorage) return null;
  return window.localStorage;
}

function knownSpecNames(specs = []) {
  return new Set((specs || []).map((spec) => spec.name));
}

export function readStoredSettings(key) {
  const storage = safeStorage();
  if (!storage) return {};
  try {
    const parsed = JSON.parse(storage.getItem(key) || "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export function writeStoredSettings(key, value = EMPTY_OBJECT) {
  const storage = safeStorage();
  if (!storage) return;
  storage.setItem(key, JSON.stringify(value || {}));
}

export function clearStoredSettings(key) {
  const storage = safeStorage();
  if (!storage) return;
  storage.removeItem(key);
}

export function buildDefaultArgs(specs = [], { checkpoints = [] } = {}) {
  const initial = {};
  (specs || []).forEach((spec) => {
    if (spec.name === "ckpt" && checkpoints.length > 0) {
      initial[spec.name] = checkpoints[0];
    } else if (spec.default !== null && spec.default !== undefined) {
      initial[spec.name] = spec.default;
    } else if (spec.type === "bool") {
      initial[spec.name] = false;
    } else {
      initial[spec.name] = "";
    }
  });
  return initial;
}

export function mergeStoredArgs(defaults = EMPTY_OBJECT, stored = EMPTY_OBJECT, specs = []) {
  const names = knownSpecNames(specs);
  const merged = { ...defaults };
  Object.entries(stored || {}).forEach(([key, value]) => {
    if (names.has(key)) {
      merged[key] = value;
    }
  });
  return merged;
}

export function buildArgsWithStoredSettings(specs = [], key, options = {}) {
  const defaults = buildDefaultArgs(specs, options);
  return mergeStoredArgs(defaults, readStoredSettings(key), specs);
}

export function resetArgsToDefaults(specs = [], key, options = {}) {
  clearStoredSettings(key);
  return buildDefaultArgs(specs, options);
}

export function fileName(value) {
  return String(value || "").split(/[\\/]/).pop() || value || "—";
}

export function summarizeSampleArgs(args = {}) {
  return [
    { label: "Family", value: args.family || "mmdit" },
    { label: "Task", value: args.task || "txt2img" },
    { label: "Checkpoint", value: fileName(args.ckpt) },
    { label: "Sampler", value: args.sampler || "—" },
    { label: "Steps", value: args.steps ?? "—" },
    { label: "Batch", value: args.n ?? "—" },
    { label: "Seed", value: args.seed === "" || args.seed === undefined ? "random" : args.seed },
  ];
}

export function summarizeLatentArgs(args = {}) {
  return [
    { label: "Config", value: fileName(args.config) },
    { label: "Batch", value: args["batch-size"] ?? args.batch_size ?? "—" },
    { label: "Workers", value: args["num-workers"] ?? args.num_workers ?? "—" },
    { label: "Shard", value: args["shard-size"] ?? args.shard_size ?? "—" },
    { label: "Device", value: args.device || "auto" },
    { label: "DType", value: args["latent-dtype"] || args.latent_dtype || "—" },
  ];
}
