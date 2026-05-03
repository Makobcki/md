import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";
import YamlEditor from "../components/YamlEditor.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";
import StatusPill from "../components/StatusPill.jsx";
import {
  formatStep,
  isMetricEvent,
  latestMetricWithLoss,
  mergeMetricEvents,
  metricChartData,
  metricElapsed,
  metricEta,
  metricLoss,
  metricSecondsPerStep,
  metricStep,
  metricThroughput,
} from "../utils/metrics.js";

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
  const [saving, setSaving] = useState(false);

  const saveTimeoutRef = useRef(null);
  const metricOffsetRef = useRef(0);
  const logKey = runId ? `train:logs:${runId}` : "train:logs:idle";
  const { lines: logLines, appendLines, replaceLines, clear: clearLogs } = useLogBuffer(logKey, {
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
      if (stat.active && stat.run?.run_type === "train") {
        setRunId(stat.run.run_id);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    const loadLogs = async () => {
      try {
        const [stdout, stderr] = await Promise.all([
          api.getRunLog(runId, "stdout"),
          api.getRunLog(runId, "stderr"),
        ]);
        if (cancelled) return;
        replaceLines([
          ...stdout.content.split("\n").filter(Boolean).map((line) => `[stdout] ${line}`),
          ...stderr.content.split("\n").filter(Boolean).map((line) => `[stderr] ${line}`),
        ]);
      } catch (err) {
        console.warn("failed to load log tail", err);
      }
    };
    loadLogs();
    const timer = setInterval(loadLogs, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId, replaceLines]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    metricOffsetRef.current = 0;
    setMetrics([]);
    const loadMetrics = async () => {
      try {
        const data = await api.getRunMetrics(runId, {
          offset: metricOffsetRef.current,
          limit: 2000,
        });
        if (cancelled) return;
        const items = data.items || [];
        if (Number.isFinite(data.next_offset)) {
          metricOffsetRef.current = data.next_offset;
        }
        if (items.length > 0) {
          setMetrics((prev) => mergeMetricEvents(prev, items, 1000));
        }
      } catch (err) {
        console.warn("failed to load metrics", err);
      }
    };
    loadMetrics();
    const timer = setInterval(loadMetrics, 500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId]);

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(wsUrl(`/ws/logs/${runId}?backlog=0`));
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
        if (isMetricEvent(metric)) {
          setMetrics((prev) => mergeMetricEvents(prev, [metric], 1000));
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
      metricOffsetRef.current = 0;
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

  const lastMetric = latestMetricWithLoss(metrics);
  const chartData = metricChartData(metrics);
  const progressMax = lastMetric?.max_steps || 0;
  const progressValue = metricStep(lastMetric) ?? 0;
  const activeRun = status.active ? status.run : null;
  const activeRunType = activeRun?.run_type;

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
            <div className="train-config-editor">
              <YamlEditor value={config} onChange={setConfig} onSave={() => handleSave(false)} />
            </div>
            <div className="row" style={{ marginTop: "12px" }}>
              <button onClick={() => handleSave(false)} disabled={saving}>
                Save
              </button>
              {saving && <span className="muted">Saving...</span>}
            </div>
          </div>
        </div>

        <div className="page">
          <div className="card training-control-card">
            <div className="card-header">
              <h2 className="card-title">Training Control</h2>
              <StatusPill status={status.active ? "running" : "stopped"} />
            </div>
            {error && <div className="muted">{error}</div>}
            {status.active && !activeRun && (
              <div className="muted">Job запускается...</div>
            )}
            {activeRun && activeRunType !== "train" && (
              <div className="muted">Другой job уже выполняется: {activeRunType}</div>
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
              <>
                <div className="progress">
                  <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
                </div>
                <div className="muted">step {formatStep(lastMetric)}</div>
              </>
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
            {command.length > 0 && <div className="muted">Training command prepared.</div>}
            {lastMetric && (
              <div className="grid train-metrics-grid">
                <div>
                  <div className="muted">step</div>
                  <div>{formatStep(lastMetric)}</div>
                </div>
                <div>
                  <div className="muted">elapsed</div>
                  <div>{metricElapsed(lastMetric)}</div>
                </div>
                <div>
                  <div className="muted">ETA</div>
                  <div>{metricEta(lastMetric)}</div>
                </div>
                <div>
                  <div className="muted">s/step</div>
                  <div>{Number.isFinite(metricSecondsPerStep(lastMetric)) ? metricSecondsPerStep(lastMetric).toFixed(3) : "—"}</div>
                </div>
                <div>
                  <div className="muted">samples/s</div>
                  <div>{Number.isFinite(metricThroughput(lastMetric)) ? metricThroughput(lastMetric).toFixed(2) : "—"}</div>
                </div>
                <div>
                  <div className="muted">loss</div>
                  <div>{Number.isFinite(metricLoss(lastMetric)) ? metricLoss(lastMetric).toFixed(4) : "—"}</div>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Loss vs Step</h2>
              <span className="muted">{lastMetric ? `step ${formatStep(lastMetric)}` : "avg log_every"}</span>
            </div>
            <LineChart data={chartData} />
          </div>

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
