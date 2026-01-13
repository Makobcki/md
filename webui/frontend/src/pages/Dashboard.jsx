import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, API_ORIGIN, wsUrl } from "../api.js";
import LineChart from "../components/LineChart.jsx";

const statusClass = (status) => {
  if (!status) return "";
  return `status-pill ${status}`;
};

const toRunsUrl = (path) => {
  if (!path) return null;
  const marker = "webui_runs/";
  const idx = path.indexOf(marker);
  if (idx === -1) return null;
  return `${API_ORIGIN}/runs/${path.slice(idx + marker.length)}`;
};

export default function Dashboard() {
  const [runs, setRuns] = useState([]);
  const [status, setStatus] = useState({ active: false });
  const [metrics, setMetrics] = useState([]);
  const [samples, setSamples] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        const [runsData, statusData, samplesData] = await Promise.all([
          api.listRuns(),
          api.getStatus(),
          api.listSamples(),
        ]);
        setRuns(runsData);
        setStatus(statusData);
        setSamples(samplesData.items || []);
      } catch (err) {
        setError(err.message);
      }
    };
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!status.active) return;
    const runId = status.run.run_id;
    const ws = new WebSocket(wsUrl(`/ws/metrics/${runId}`));
    ws.onmessage = (event) => {
      try {
        const metric = JSON.parse(event.data);
        if (metric.type === "metric") {
          setMetrics((prev) => [...prev.slice(-200), metric]);
        }
      } catch (err) {
        console.warn(err);
      }
    };
    return () => ws.close();
  }, [status.active, status.run?.run_id]);

  const handleStop = async () => {
    if (!status.active) return;
    setError("");
    try {
      if (status.run.run_type === "train") {
        await api.stopTrain();
      } else if (status.run.run_type === "sample") {
        await api.stopSample();
      } else if (status.run.run_type === "latent_cache") {
        await api.stopLatentCache();
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const lastMetric = metrics[metrics.length - 1];
  const progressValue = lastMetric?.step ?? lastMetric?.processed ?? 0;
  const progressMax = lastMetric?.max_steps || lastMetric?.total || 0;

  const recentSamples = useMemo(() => {
    return [...samples].reverse().slice(0, 8);
  }, [samples]);

  return (
    <div className="page">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Active Tasks</h2>
          {error && <span className="muted">{error}</span>}
        </div>
        {status.active ? (
          <div className="grid">
            <div className="card soft">
              <div className="row">
                <span className={statusClass(status.run.status)}>{status.run.status}</span>
                <strong>{status.run.run_type}</strong>
                <span className="muted">{status.run.run_id}</span>
              </div>
              <div className="row">
                <button onClick={handleStop} className="danger">
                  Stop
                </button>
                <button className="secondary" disabled>
                  Pause
                </button>
                <Link className="muted" to={`/runs/${status.run.run_id}`}>
                  Open run
                </Link>
              </div>
              {progressMax ? (
                <div className="progress">
                  <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
                </div>
              ) : (
                <div className="muted">Progress unavailable</div>
              )}
              <div className="grid">
                <div>
                  <div className="muted">Step</div>
                  <div>{lastMetric?.step ?? "-"}</div>
                </div>
                <div>
                  <div className="muted">ETA</div>
                  <div>{lastMetric?.eta_h ? `${lastMetric.eta_h.toFixed(2)}h` : "-"}</div>
                </div>
                <div>
                  <div className="muted">Speed</div>
                  <div>{lastMetric?.img_per_sec ? `${lastMetric.img_per_sec.toFixed(2)} it/s` : "-"}</div>
                </div>
                <div>
                  <div className="muted">VRAM</div>
                  <div>{lastMetric?.peak_mem_mb ? `${lastMetric.peak_mem_mb.toFixed(0)} MB` : "-"}</div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="muted">Нет активных задач.</div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Training Metrics</h2>
        </div>
        {status.active && status.run.run_type === "train" ? (
          <LineChart data={metrics.map((m) => ({ step: m.step, loss: m.loss }))} />
        ) : (
          <div className="muted">Нет активного обучения.</div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Recent Generations</h2>
          <Link className="muted" to="/generate">
            Open Generate
          </Link>
        </div>
        {recentSamples.length === 0 ? (
          <div className="muted">Нет сгенерированных изображений.</div>
        ) : (
          <div className="gallery-grid">
            {recentSamples.map((item) => {
              const url = toRunsUrl(item);
              return url ? (
                <div key={item} className="image-card">
                  <img src={url} alt="sample" />
                  <div className="image-meta">
                    <span className="badge">{item.split("/").pop()}</span>
                    <Link className="muted" to="/generate">
                      Use in Generate
                    </Link>
                  </div>
                </div>
              ) : null;
            })}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Runs</h2>
          <Link className="muted" to="/files">
            View files/logs
          </Link>
        </div>
        <div className="grid">
          {runs.map((run) => (
            <div key={run.run_id} className="card soft">
              <div className="row">
                <span className={statusClass(run.status)}>{run.status}</span>
                <strong>{run.run_type}</strong>
              </div>
              <div className="muted">{run.created_at}</div>
              <div className="muted">{run.run_id}</div>
              <Link className="muted" to={`/runs/${run.run_id}`}>
                Details
              </Link>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
