const DEFAULT_API_BASE =
  window.location.port === "5173" ? "http://127.0.0.1:8000" : window.location.origin;
const API_BASE = (import.meta?.env?.VITE_API_BASE || DEFAULT_API_BASE).replace(/\/+$/, "");
export const API_ORIGIN = new URL(API_BASE).origin;

export async function fetchJson(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
    },
    ...options,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || res.statusText);
  }
  return res.json();
}

export const api = {
  getStatus: () => fetchJson("/api/status"),
  listRuns: () => fetchJson("/api/runs"),
  getRun: (id) => fetchJson(`/api/runs/${id}`),
  getRunConfig: (id) => fetchJson(`/api/runs/${id}/config`),
  getRunLog: (id, stream) => fetchJson(`/api/runs/${id}/logs/${stream}`),
  getRunMetrics: (id) => fetchJson(`/api/runs/${id}/metrics`),
  getConfig: () => fetchJson("/api/config"),
  updateConfig: (content) =>
    fetchJson("/api/config", { method: "PUT", body: JSON.stringify({ content }) }),
  listCheckpoints: () => fetchJson("/api/checkpoints"),
  getCheckpointInfo: (path) =>
    fetchJson(`/api/checkpoints/info?path=${encodeURIComponent(path)}`),
  getOutDirSummary: () => fetchJson("/api/out_dir/summary"),
  listSamples: () => fetchJson("/api/samples"),
  getSampleArgs: () => fetchJson("/api/sample/args"),
  getLatentArgs: () => fetchJson("/api/latents/args"),
  startTrain: (payload = {}) =>
    fetchJson("/api/train/start", { method: "POST", body: JSON.stringify(payload) }),
  stopTrain: () => fetchJson("/api/train/stop", { method: "POST" }),
  startSample: (args) =>
    fetchJson("/api/sample/start", { method: "POST", body: JSON.stringify({ args }) }),
  stopSample: () => fetchJson("/api/sample/stop", { method: "POST" }),
  startLatentCache: (args) =>
    fetchJson("/api/latents/start", { method: "POST", body: JSON.stringify({ args }) }),
  stopLatentCache: () => fetchJson("/api/latents/stop", { method: "POST" }),
};

export function wsUrl(path) {
  const url = new URL(API_BASE);
  const protocol = url.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${url.host}${path}`;
}
