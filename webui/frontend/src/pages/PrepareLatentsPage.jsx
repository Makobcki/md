import React, { useEffect, useMemo, useState } from "react";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";

const statusClass = (status) => `status-pill ${status || ""}`;

export default function PrepareLatentsPage() {
  const [argSpecs, setArgSpecs] = useState([]);
  const [args, setArgs] = useState({});
  const [status, setStatus] = useState({ active: false });
  const [runId, setRunId] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [command, setCommand] = useState([]);
  const [error, setError] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);

  const logKey = runId ? `latents:logs:${runId}` : "latents:logs:idle";
  const { lines: logLines, appendLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const argsData = await api.getLatentArgs();
      setArgSpecs(argsData.items || []);

      const initial = {};
      (argsData.items || []).forEach((spec) => {
        if (spec.default !== null && spec.default !== undefined) {
          initial[spec.name] = spec.default;
        } else if (spec.type === "bool") {
          initial[spec.name] = false;
        } else {
          initial[spec.name] = "";
        }
      });
      setArgs(initial);
    };
    load();
  }, []);

  useEffect(() => {
    const poll = async () => {
      const stat = await api.getStatus();
      setStatus(stat);
      if (stat.active && stat.run.run_type === "latent_cache") {
        setRunId(stat.run.run_id);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(wsUrl(`/ws/logs/${runId}`));
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "log") {
        appendLines(`[${payload.stream}] ${payload.line}`);
      }
    };
    return () => ws.close();
  }, [runId, appendLines]);

  useEffect(() => {
    if (!runId) return;
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
  }, [runId]);

  const handleChange = (name, value) => {
    setArgs((prev) => ({ ...prev, [name]: value }));
  };

  const handleStart = async (overrideArgs = null) => {
    setError("");
    try {
      const resp = await api.startLatentCache(overrideArgs || args);
      setRunId(resp.run_id);
      setCommand(resp.command || []);
      clearLogs();
      setMetrics([]);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRebuild = async () => {
    const nextArgs = { ...args, overwrite: true };
    setArgs(nextArgs);
    await handleStart(nextArgs);
  };

  const handleStop = async () => {
    setError("");
    try {
      await api.stopLatentCache();
    } catch (err) {
      setError(err.message);
    }
  };

  const lastMetric = metrics[metrics.length - 1];
  const progressMax = lastMetric?.max_steps || 0;
  const progressValue = lastMetric?.processed ?? 0;

  const renderField = (spec) => {
    if (spec.choices && spec.choices.length > 0) {
      return (
        <select
          value={args[spec.name] || ""}
          onChange={(event) => handleChange(spec.name, event.target.value)}
        >
          {spec.choices.map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </select>
      );
    }

    if (spec.type === "int" || spec.type === "float") {
      return (
        <input
          type="number"
          value={args[spec.name]}
          onChange={(event) => handleChange(spec.name, event.target.value)}
        />
      );
    }

    if (spec.type === "bool" || typeof spec.default === "boolean") {
      return (
        <input
          type="checkbox"
          checked={Boolean(args[spec.name])}
          onChange={(event) => handleChange(spec.name, event.target.checked)}
        />
      );
    }

    return (
      <input
        type="text"
        value={args[spec.name] ?? ""}
        onChange={(event) => handleChange(spec.name, event.target.value)}
      />
    );
  };

  const overview = useMemo(() => {
    if (!lastMetric) return [];
    return [
      { label: "processed", value: lastMetric.processed ?? "-" },
      { label: "saved", value: lastMetric.saved ?? "-" },
      { label: "errors", value: lastMetric.errors ?? "-" },
      {
        label: "items/s",
        value: lastMetric.items_per_sec ? lastMetric.items_per_sec.toFixed(2) : "-",
      },
    ];
  }, [lastMetric]);

  return (
    <div className="page">
      <h1 className="page-title">Prepare Latents</h1>
      <div className="two-col">
        <div className="page">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Control</h2>
              <span className={statusClass(status.active ? "running" : "stopped")}>
                {status.active ? "running" : "stopped"}
              </span>
            </div>
            {error && <div className="muted">{error}</div>}
            {status.active && status.run.run_type !== "latent_cache" && (
              <div className="muted">Другой job уже выполняется: {status.run.run_type}</div>
            )}
            <div className="row">
              <button onClick={handleStart} disabled={status.active}>
                Start
              </button>
              <button className="secondary" onClick={handleRebuild} disabled={status.active}>
                Rebuild cache
              </button>
              <button className="danger" onClick={handleStop} disabled={!status.active}>
                Stop
              </button>
            </div>
            {status.active && progressMax ? (
              <div className="progress">
                <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
              </div>
            ) : null}
            {command.length > 0 && <div className="muted">Command: {command.join(" ")}</div>}
            <div className="grid">
              {overview.map((item) => (
                <div key={item.label}>
                  <div className="muted">{item.label}</div>
                  <div>{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Cache Settings</h2>
              <span className="muted">dataset/.cache</span>
            </div>
            <div className="grid">
              {argSpecs.map((spec) => (
                <div key={spec.name} className="card soft">
                  <div className="muted">{spec.flags.join(", ")}</div>
                  <label>{spec.name}</label>
                  {renderField(spec)}
                  <div className="muted">{spec.help}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="page">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Live Logs</h2>
              <div className="row">
                <label className="row">
                  <input
                    type="checkbox"
                    checked={autoScroll}
                    onChange={(event) => setAutoScroll(event.target.checked)}
                  />
                  Auto-scroll
                </label>
                <button className="ghost" onClick={clearLogs}>
                  Clear
                </button>
              </div>
            </div>
            <LogViewer lines={logLines} autoScroll={autoScroll} />
            <div className="muted">Buffer: {logLines.length} / 10k lines</div>
          </div>
        </div>
      </div>
    </div>
  );
}
