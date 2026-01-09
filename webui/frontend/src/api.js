const API_BASE = "http://127.0.0.1:8000";

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
  listSamples: () => fetchJson("/api/samples"),
  getSampleArgs: () => fetchJson("/api/sample/args"),
  startTrain: () => fetchJson("/api/train/start", { method: "POST" }),
  stopTrain: () => fetchJson("/api/train/stop", { method: "POST" }),
  startSample: (args) =>
    fetchJson("/api/sample/start", { method: "POST", body: JSON.stringify({ args }) }),
  stopSample: () => fetchJson("/api/sample/stop", { method: "POST" }),
};

export function wsUrl(path) {
  return `ws://127.0.0.1:8000${path}`;
}
