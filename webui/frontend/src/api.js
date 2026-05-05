const DEFAULT_API_BASE = window.location.origin;
const API_BASE = (import.meta?.env?.VITE_API_BASE || DEFAULT_API_BASE).replace(/\/+$/, "");
export const API_ORIGIN = new URL(API_BASE).origin;

export function getAuthToken() {
  return import.meta?.env?.VITE_AUTH_TOKEN || window.localStorage?.getItem("WEBUI_AUTH_TOKEN") || "";
}

export const AUTH_TOKEN = getAuthToken();

export function setAuthToken(token) {
  if (token) {
    window.localStorage?.setItem("WEBUI_AUTH_TOKEN", token);
  } else {
    window.localStorage?.removeItem("WEBUI_AUTH_TOKEN");
  }
}

export function absoluteFileUrl(item) {
  if (!item) return null;
  if (typeof item === "object" && item.url) {
    return String(item.url).startsWith("/") ? `${API_ORIGIN}${item.url}` : String(item.url);
  }
  return null;
}

export function absoluteDownloadUrl(item) {
  if (!item) return null;
  if (typeof item === "object" && item.download_url) {
    return String(item.download_url).startsWith("/") ? `${API_ORIGIN}${item.download_url}` : String(item.download_url);
  }
  return absoluteFileUrl(item);
}

function authHeaders() {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function errorMessage(detail, fallback) {
  if (Array.isArray(detail?.detail)) {
    return detail.detail
      .map((entry) => `${(entry.loc || []).join(".") || "request"}: ${entry.msg || JSON.stringify(entry)}`)
      .join("\n");
  }
  if (typeof detail?.detail === "string") return detail.detail;
  if (detail?.detail) return JSON.stringify(detail.detail);
  return fallback;
}

export async function fetchJson(path, options = {}) {
  const headers = {
    "Content-Type": "application/json",
    ...authHeaders(),
    ...(options.headers || {}),
  };
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(errorMessage(detail, res.statusText));
  }
  return res.json();
}

export async function uploadImage(file, kind = "image") {
  const form = new FormData();
  form.append("file", file);
  form.append("kind", kind);
  const res = await fetch(`${API_BASE}/api/uploads/image`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(errorMessage(detail, res.statusText));
  }
  return res.json();
}

const OPTIONAL_EMPTY_FIELDS = new Set(["shift", "seed", "limit", "width", "height"]);

function cleanArgs(args = {}) {
  const payload = { ...args };
  Object.entries(payload).forEach(([key, value]) => {
    if (value === "" && OPTIONAL_EMPTY_FIELDS.has(key)) {
      delete payload[key];
    }
  });
  const task = payload.task || "txt2img";
  if (task === "txt2img") {
    delete payload["init-image"];
    delete payload.mask;
    delete payload["control-image"];
  } else if (task === "img2img") {
    delete payload.mask;
    delete payload["control-image"];
  } else if (task === "inpaint") {
    delete payload["control-image"];
  } else if (task === "control") {
    delete payload["init-image"];
    delete payload.mask;
  }
  return payload;
}

export const api = {
  getStatus: () => fetchJson("/api/status"),
  listRuns: () => fetchJson("/api/runs"),
  getRun: (id) => fetchJson(`/api/runs/${id}`),
  getRunConfig: (id) => fetchJson(`/api/runs/${id}/config`),
  getRunLog: (id, stream, params = {}) => {
    const query = new URLSearchParams();
    if (params.raw) query.set("raw", "true");
    if (Number.isFinite(params.limit)) query.set("limit", String(params.limit));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return fetchJson(`/api/runs/${id}/logs/${stream}${suffix}`);
  },
  getRunMetrics: (id, params = {}) => {
    const query = new URLSearchParams();
    if (Number.isFinite(params.offset)) query.set("offset", String(params.offset));
    if (Number.isFinite(params.limit)) query.set("limit", String(params.limit));
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return fetchJson(`/api/runs/${id}/metrics${suffix}`);
  },
  getConfig: () => fetchJson("/api/config"),
  updateConfig: (content) =>
    fetchJson("/api/config", { method: "PUT", body: JSON.stringify({ content }) }),
  listCheckpoints: () => fetchJson("/api/checkpoints"),
  getCheckpointInfo: (path) =>
    fetchJson(`/api/checkpoints/info?path=${encodeURIComponent(path)}`),
  getOutDirSummary: () => fetchJson("/api/out_dir/summary"),
  getAuthStatus: () => fetchJson("/api/auth/status"),
  listSamples: (runId = "") => {
    const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    return fetchJson(`/api/generated-samples${suffix}`);
  },
  listTrainSamples: (runId = "") => {
    const suffix = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    return fetchJson(`/api/train-samples${suffix}`);
  },
  listArtifacts: ({ runId = "", source = "all" } = {}) => {
    const query = new URLSearchParams();
    if (runId) query.set("run_id", runId);
    if (source) query.set("source", source);
    const suffix = query.toString() ? `?${query.toString()}` : "";
    return fetchJson(`/api/artifacts${suffix}`);
  },
  getSampleArgs: () => fetchJson("/api/sample/args"),
  getLatentArgs: () => fetchJson("/api/latents/args"),
  startTrain: (payload = {}) =>
    fetchJson("/api/train/start", { method: "POST", body: JSON.stringify(payload) }),
  stopTrain: () => fetchJson("/api/train/stop", { method: "POST" }),
  startSample: (args) =>
    fetchJson("/api/sample/start", { method: "POST", body: JSON.stringify({ args: cleanArgs(args) }) }),
  stopSample: () => fetchJson("/api/sample/stop", { method: "POST" }),
  startLatentCache: (args) =>
    fetchJson("/api/latents/start", { method: "POST", body: JSON.stringify({ args: cleanArgs(args) }) }),
  stopLatentCache: () => fetchJson("/api/latents/stop", { method: "POST" }),
  uploadImage,
};

export function wsUrl(path) {
  const url = new URL(API_BASE);
  const protocol = url.protocol === "https:" ? "wss" : "ws";
  const ws = new URL(path, `${protocol}://${url.host}`);
  const token = getAuthToken();
  if (token) {
    ws.searchParams.set("token", token);
  }
  return ws.toString();
}
