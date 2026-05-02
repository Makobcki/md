import React, { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, API_ORIGIN, wsUrl } from "../api.js";
import LineChart from "../components/LineChart.jsx";
import StatusPill from "../components/StatusPill.jsx";
import {
  formatDate,
  formatRelativeDate,
  formatRunId,
  formatRunType,
  isToday,
  lastUsefulLogLine,
  parseRunDate,
} from "../utils/formatters.js";

const toRunsUrl = (path) => {
  if (!path) return null;
  const marker = "webui_runs/";
  const idx = path.indexOf(marker);
  if (idx === -1) return null;
  return `${API_ORIGIN}/runs/${path.slice(idx + marker.length)}`;
};

export default function Dashboard() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [status, setStatus] = useState({ active: false });
  const [metrics, setMetrics] = useState([]);
  const [samples, setSamples] = useState([]);
  const [error, setError] = useState("");
  const [runFilters, setRunFilters] = useState({ failedOnly: false, todayOnly: false });
  const [sortDir, setSortDir] = useState("desc");
  const [errorHints, setErrorHints] = useState({});

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

  const activeRun = status.active ? status.run : null;

  useEffect(() => {
    const failedRuns = runs.filter((run) => run.status === "failed" && !errorHints[run.run_id]);
    if (failedRuns.length === 0) return;
    let cancelled = false;
    Promise.all(
      failedRuns.slice(0, 12).map(async (run) => {
        try {
          const stderr = await api.getRunLog(run.run_id, "stderr");
          const stdout = await api.getRunLog(run.run_id, "stdout");
          return [run.run_id, lastUsefulLogLine(stderr.content) || lastUsefulLogLine(stdout.content)];
        } catch {
          return [run.run_id, ""];
        }
      })
    ).then((items) => {
      if (cancelled) return;
      setErrorHints((prev) => ({
        ...prev,
        ...Object.fromEntries(items.filter(([, hint]) => hint)),
      }));
    });
    return () => {
      cancelled = true;
    };
  }, [runs, errorHints]);

  useEffect(() => {
    if (!activeRun?.run_id) return;
    const runId = activeRun.run_id;
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
  }, [activeRun?.run_id]);

  const handleStop = async () => {
    if (!activeRun) return;
    setError("");
    try {
      if (activeRun.run_type === "train") {
        await api.stopTrain();
      } else if (activeRun.run_type === "sample") {
        await api.stopSample();
      } else if (activeRun.run_type === "latent_cache") {
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

  const visibleRuns = useMemo(() => {
    return [...runs]
      .filter((run) => (runFilters.failedOnly ? run.status === "failed" : true))
      .filter((run) => (runFilters.todayOnly ? isToday(parseRunDate(run)) : true))
      .sort((a, b) => {
        const left = parseRunDate(a)?.getTime() || 0;
        const right = parseRunDate(b)?.getTime() || 0;
        return sortDir === "desc" ? right - left : left - right;
      });
  }, [runs, runFilters, sortDir]);

  return (
    <div className="page">
      <h1 className="page-title">Dashboard</h1>
      {(activeRun || status.active) && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Active Task</h2>
            {error && <span className="muted">{error}</span>}
          </div>
          {activeRun ? (
            <div className="grid">
              <div className="card soft">
                <div className="row">
                  <StatusPill status={activeRun.status} />
                  <strong>{formatRunType(activeRun.run_type)}</strong>
                  <span className="muted">{formatRunId(activeRun.run_id)}</span>
                </div>
                <div className="row">
                  <button onClick={handleStop} className="danger">
                    Stop
                  </button>
                  <button className="secondary" disabled>
                    Pause
                  </button>
                  <Link className="muted" to={`/runs/${activeRun.run_id}`}>
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
              {activeRun.run_type === "train" && (
                <div className="card soft">
                  <div className="card-header">
                    <h2 className="card-title">Training Metrics</h2>
                  </div>
                  <LineChart data={metrics.map((m) => ({ step: m.step, loss: m.loss }))} />
                </div>
              )}
            </div>
          ) : (
            <div className="muted">Job запускается...</div>
          )}
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Recent Generations</h2>
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
          <div className="row">
            <label className="chip-control">
              <input
                type="checkbox"
                checked={runFilters.failedOnly}
                onChange={(event) =>
                  setRunFilters((prev) => ({ ...prev, failedOnly: event.target.checked }))
                }
              />
              Только ошибки
            </label>
            <label className="chip-control">
              <input
                type="checkbox"
                checked={runFilters.todayOnly}
                onChange={(event) =>
                  setRunFilters((prev) => ({ ...prev, todayOnly: event.target.checked }))
                }
              />
              За сегодня
            </label>
            <button
              className="ghost"
              type="button"
              onClick={() => setSortDir((prev) => (prev === "desc" ? "asc" : "desc"))}
            >
              {sortDir === "desc" ? "Сначала новые" : "Сначала старые"}
            </button>
          </div>
        </div>
        <table className="table runs-table">
          <thead>
            <tr>
              <th>Status</th>
              <th>Type</th>
              <th>Created</th>
              <th>Run</th>
            </tr>
          </thead>
          <tbody>
            {visibleRuns.map((run) => {
              const created = parseRunDate(run);
              return (
                <tr
                  key={run.run_id}
                  className="clickable-row"
                  onClick={() => navigate(`/runs/${run.run_id}`)}
                >
                  <td>
                    <StatusPill status={run.status} title={errorHints[run.run_id] || run.status} />
                  </td>
                  <td>{formatRunType(run.run_type)}</td>
                  <td title={created ? formatDate(created) : run.created_at}>
                    {created ? formatRelativeDate(created) : "-"}
                  </td>
                  <td className="muted" title={run.run_id}>
                    {formatRunId(run.run_id)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
