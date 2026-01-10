import React, { useEffect, useState } from "react";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";

export default function TrainPage() {
  const [config, setConfig] = useState("");
  const [status, setStatus] = useState({ active: false });
  const [logs, setLogs] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [runId, setRunId] = useState(null);
  const [command, setCommand] = useState([]);
  const [error, setError] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [resumeCkpt, setResumeCkpt] = useState("");

  useEffect(() => {
    const load = async () => {
      const [cfg, ckpts] = await Promise.all([api.getConfig(), api.listCheckpoints()]);
      setConfig(cfg.content);
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
        setLogs((prev) => [...prev.slice(-200), `[${payload.stream}] ${payload.line}`]);
      }
    };
    return () => ws.close();
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(wsUrl(`/ws/metrics/${runId}`));
    ws.onmessage = (event) => {
      try {
        const metric = JSON.parse(event.data);
        if (metric.type === "metric") {
          setMetrics((prev) => [...prev.slice(-200), metric]);
        }
      } catch (e) {
        console.warn(e);
      }
    };
    return () => ws.close();
  }, [runId]);

  const handleSave = async () => {
    setError("");
    try {
      await api.updateConfig(config);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStart = async () => {
    setError("");
    try {
      const payload = resumeCkpt ? { resume: resumeCkpt } : {};
      const resp = await api.startTrain(payload);
      setRunId(resp.run_id);
      setCommand(resp.command);
      setLogs([]);
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
    <div className="container">
      <div className="card">
        <h2>Train</h2>
        {error && <div className="muted">{error}</div>}
        {status.active && status.run.run_type !== "train" && (
          <div className="muted">Другой job уже выполняется: {status.run.run_type}</div>
        )}
        <div className="row">
          <button onClick={handleStart} disabled={status.active}>Start</button>
          <button className="danger" onClick={handleStop} disabled={!status.active}>Stop</button>
          <span className="status-pill">{status.active ? "running" : "stopped"}</span>
          {lastMetric && (
            <>
              <span>elapsed: {Math.round(lastMetric.elapsed_sec)}s</span>
              <span>ETA: {lastMetric.eta_h?.toFixed(2)}h</span>
              <span>s/step: {lastMetric.sec_per_step?.toFixed(3)}</span>
              <span>img/s: {lastMetric.img_per_sec?.toFixed(2)}</span>
              <span>VRAM: {lastMetric.peak_mem_mb?.toFixed(0)} MB</span>
            </>
          )}
        </div>
        {status.active && (
          <progress
            style={{ width: "100%" }}
            value={progressMax ? progressValue : undefined}
            max={progressMax || undefined}
          />
        )}
        <div className="row">
          <label>Resume checkpoint</label>
          <select
            value={resumeCkpt}
            onChange={(e) => setResumeCkpt(e.target.value)}
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
        {command.length > 0 && (
          <div className="muted">Command: {command.join(" ")}</div>
        )}
      </div>

      <div className="card">
        <h3>train.yaml</h3>
        <textarea
          value={config}
          onChange={(e) => setConfig(e.target.value)}
          rows={18}
          style={{ width: "100%" }}
        />
        <div className="row">
          <button onClick={handleSave}>Save</button>
        </div>
      </div>

      <div className="card">
        <h3>Loss vs Step</h3>
        <LineChart data={metrics.map((m) => ({ step: m.step, loss: m.loss }))} />
      </div>

      <div className="card">
        <h3>Live logs</h3>
        <LogViewer lines={logs} />
      </div>
    </div>
  );
}
