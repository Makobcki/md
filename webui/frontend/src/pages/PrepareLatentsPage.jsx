import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";
import useRunLogStream from "../hooks/useRunLogStream.js";
import MetricTile from "../components/MetricTile.jsx";
import PageHeader from "../components/PageHeader.jsx";
import StatusPill from "../components/StatusPill.jsx";
import { isMetricEvent, mergeMetricEvents } from "../utils/metrics.js";
import {
  buildArgsWithStoredSettings,
  LATENT_SETTINGS_KEY,
  summarizeLatentArgs,
} from "../utils/settingsModel.js";


export default function PrepareLatentsPage() {
  const [args, setArgs] = useState({});
  const [status, setStatus] = useState({ active: false });
  const [runId, setRunId] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [command, setCommand] = useState([]);
  const [error, setError] = useState("");

  const logKey = runId ? `latents:logs:${runId}` : "latents:logs:idle";
  const { lines: logLines, appendLines, replaceLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const argsData = await api.getLatentArgs();
      const initial = buildArgsWithStoredSettings(argsData.items || [], LATENT_SETTINGS_KEY);
      setArgs(initial);
    };
    load();
  }, []);

  useEffect(() => {
    const poll = async () => {
      const stat = await api.getStatus();
      setStatus(stat);
      if (stat.active && stat.run?.run_type === "latent_cache") {
        setRunId(stat.run.run_id);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (runId) replaceLines([]);
  }, [runId, replaceLines]);

  useRunLogStream(runId, {
    backlog: 2000,
    onLog: (payload) => appendLines(`[${payload.stream}] ${payload.line}`),
    onError: (err) => console.warn("failed to stream logs", err),
  });

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(wsUrl(`/ws/metrics/${runId}`));
    ws.onmessage = (event) => {
      try {
        const metric = JSON.parse(event.data);
        if (isMetricEvent(metric)) {
          setMetrics((prev) => mergeMetricEvents(prev, [metric], 500));
        }
      } catch (err) {
        console.warn(err);
      }
    };
    return () => ws.close();
  }, [runId]);


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
  const activeRun = status.active ? status.run : null;
  const activeRunType = activeRun?.run_type;
  const isLatentActive = activeRunType === "latent_cache";

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

  const settingsSummary = useMemo(() => summarizeLatentArgs(args), [args]);


  return (
    <div className="page fit-page latents-page">
      <PageHeader
        eyebrow="Dataset cache"
        title="Latents"
        description="Build cache shards."
        meta={<StatusPill status={isLatentActive ? activeRun?.status || "running" : "stopped"} />}
      />
      <div className="split latents-page-grid">
        <div className="stack-page latents-main-column">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Control</h2>
              <StatusPill status={isLatentActive ? activeRun?.status || "running" : "stopped"} />
            </div>
            {error && <div className="muted">{error}</div>}
            {status.active && !activeRun && (
              <div className="muted">Job запускается...</div>
            )}
            {activeRun && activeRunType !== "latent_cache" && (
              <div className="muted">Другой job уже выполняется: {activeRunType}</div>
            )}
            <div className="row">
              <button onClick={handleStart} disabled={status.active}>
                Start
              </button>
              <button className="warning" onClick={handleRebuild} disabled={status.active}>
                Rebuild cache
              </button>
              <button className="danger" onClick={handleStop} disabled={!isLatentActive}>
                Stop
              </button>
            </div>
            {isLatentActive && progressMax ? (
              <div className="progress">
                <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
              </div>
            ) : null}
            {command.length > 0 && <div className="muted">Latent cache task prepared.</div>}
            <div className="compact-metrics">
              {overview.map((item) => (
                <MetricTile key={item.label} label={item.label} value={item.value} />
              ))}
            </div>
          </div>

          <div className="generation-context-card">
            <div className="card-header">
              <div>
                <h2 className="card-title">Cache profile</h2>
                <div className="muted">Параметры подготовки латентов вынесены в отдельное меню настроек.</div>
              </div>
              <Link className="secondary" to="/settings">
                Settings
              </Link>
            </div>
            <div className="settings-summary-grid compact">
              {settingsSummary.map((item) => (
                <div key={`${item.label}:${item.value}`} className="settings-summary-item">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="stack-page latents-side-column">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Live Logs</h2>
              <div className="row">
                <button className="ghost" onClick={clearLogs}>
                  Clear
                </button>
              </div>
            </div>
            <LogViewer lines={logLines} />
            <div className="muted">Buffer: {logLines.length} / 10k lines</div>
          </div>
        </div>
      </div>
    </div>
  );
}
