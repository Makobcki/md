import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";
import YamlEditor from "../components/YamlEditor.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";

const statusClass = (status) => `status-pill ${status || ""}`;

export default function TrainPage() {
  const [config, setConfig] = useState("");
  const [lastSaved, setLastSaved] = useState("");
  const [status, setStatus] = useState({ active: false });
  const [metrics, setMetrics] = useState([]);
  const [runId, setRunId] = useState(null);
  const [command, setCommand] = useState([]);
  const [error, setError] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [resumeCkpt, setResumeCkpt] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [saving, setSaving] = useState(false);

  const saveTimeoutRef = useRef(null);
  const logKey = runId ? `train:logs:${runId}` : "train:logs:idle";
  const { lines: logLines, appendLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const [cfg, ckpts] = await Promise.all([api.getConfig(), api.listCheckpoints()]);
      setConfig(cfg.content);
      setLastSaved(cfg.content);
      setCheckpoints(ckpts.items || []);
    };
    load();
    const poll = async () => {
      const stat = await api.getStatus();
      setStatus(stat);
      if (stat.active && stat.run.run_type === "train") {
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

  const isDirty = useMemo(() => config !== lastSaved, [config, lastSaved]);

  useEffect(() => {
    if (!isDirty) return;
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    saveTimeoutRef.current = setTimeout(() => {
      handleSave(true);
    }, 1200);
    return () => clearTimeout(saveTimeoutRef.current);
  }, [config]);

  const handleSave = async (silent = false) => {
    if (saving) return;
    setSaving(true);
    if (!silent) {
      setError("");
    }
    try {
      await api.updateConfig(config);
      setLastSaved(config);
    } catch (err) {
      if (!silent) {
        setError(err.message);
      }
    } finally {
      setSaving(false);
    }
  };

  const handleStart = async () => {
    setError("");
    if (isDirty) {
      await handleSave(true);
    }
    try {
      const payload = resumeCkpt ? { resume: resumeCkpt } : {};
      const resp = await api.startTrain(payload);
      setRunId(resp.run_id);
      setCommand(resp.command);
      clearLogs();
      setMetrics([]);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStop = async () => {
    setError("");
    try {
      await api.stopTrain();
    } catch (err) {
      setError(err.message);
    }
  };

  const lastMetric = metrics[metrics.length - 1];
  const progressMax = lastMetric?.max_steps || 0;
  const progressValue = lastMetric?.step ?? 0;

  return (
    <div className="page">
      <h1 className="page-title">Train</h1>
      <div className="two-col">
        <div className="page">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Config Editor</h2>
              <span className="muted">{isDirty ? "Unsaved changes" : "Saved"}</span>
            </div>
            <YamlEditor value={config} onChange={setConfig} onSave={() => handleSave(false)} />
            <div className="row" style={{ marginTop: "12px" }}>
              <button onClick={() => handleSave(false)} disabled={saving}>
                Save
              </button>
              {saving && <span className="muted">Saving...</span>}
            </div>
          </div>
        </div>

        <div className="page">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Training Control</h2>
              <span className={statusClass(status.active ? "running" : "stopped")}>
                {status.active ? "running" : "stopped"}
              </span>
            </div>
            {error && <div className="muted">{error}</div>}
            {status.active && status.run.run_type !== "train" && (
              <div className="muted">Другой job уже выполняется: {status.run.run_type}</div>
            )}
            <div className="row">
              <button onClick={handleStart} disabled={status.active}>
                Start
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
            <div className="row">
              <label>Resume checkpoint</label>
              <select
                value={resumeCkpt}
                onChange={(event) => setResumeCkpt(event.target.value)}
                disabled={status.active}
              >
                <option value="">(none)</option>
                {checkpoints.map((ckpt) => (
                  <option key={ckpt} value={ckpt}>
                    {ckpt}
                  </option>
                ))}
              </select>
            </div>
            {command.length > 0 && <div className="muted">Command: {command.join(" ")}</div>}
            {lastMetric && (
              <div className="grid">
                <div>
                  <div className="muted">elapsed</div>
                  <div>{Math.round(lastMetric.elapsed_sec)}s</div>
                </div>
                <div>
                  <div className="muted">ETA</div>
                  <div>{lastMetric.eta_h?.toFixed(2)}h</div>
                </div>
                <div>
                  <div className="muted">s/step</div>
                  <div>{lastMetric.sec_per_step?.toFixed(3)}</div>
                </div>
                <div>
                  <div className="muted">img/s</div>
                  <div>{lastMetric.img_per_sec?.toFixed(2)}</div>
                </div>
                <div>
                  <div className="muted">VRAM</div>
                  <div>{lastMetric.peak_mem_mb?.toFixed(0)} MB</div>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Loss vs Step</h2>
              <span className="muted">avg log_every</span>
            </div>
            <LineChart data={metrics.map((m) => ({ step: m.step, loss: m.loss }))} />
          </div>

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

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Samples</h2>
            </div>
            <div className="row">
              <a href="/train/samples" className="muted">
                Open samples gallery
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
