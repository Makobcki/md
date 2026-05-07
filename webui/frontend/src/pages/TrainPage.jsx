import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";
import MetricTile from "../components/MetricTile.jsx";
import PageHeader from "../components/PageHeader.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";
import useRunLogStream from "../hooks/useRunLogStream.js";
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
  const [status, setStatus] = useState({ active: false });
  const [metrics, setMetrics] = useState([]);
  const [runId, setRunId] = useState(null);
  const [command, setCommand] = useState([]);
  const [error, setError] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [resumeCkpt, setResumeCkpt] = useState("");

  const metricOffsetRef = useRef(0);
  const logKey = runId ? `train:logs:${runId}` : "train:logs:idle";
  const { lines: logLines, appendLines, replaceLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const ckpts = await api.listCheckpoints();
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
    if (runId) replaceLines([]);
  }, [runId, replaceLines]);

  useEffect(() => {
    if (!runId) return undefined;
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

  useRunLogStream(runId, {
    backlog: 2000,
    onLog: (payload) => appendLines(`[${payload.stream}] ${payload.line}`),
    onError: (err) => console.warn("failed to stream logs", err),
  });

  useEffect(() => {
    if (!runId) return undefined;
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

  const handleStart = async () => {
    setError("");
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
  const isTrainActive = activeRunType === "train";
  const trainingSummary = useMemo(
    () => [
      { label: "Config", value: "Settings → Config" },
      { label: "Resume", value: resumeCkpt || "(none)" },
      { label: "Run", value: runId || "—" },
      { label: "State", value: activeRunType === "train" ? activeRun?.status || "running" : "idle" },
    ],
    [activeRun?.status, activeRunType, resumeCkpt, runId]
  );

  return (
    <div className="page fit-page train-page">
      <PageHeader
        eyebrow="Training"
        title="Train"
        description="Start, resume and monitor."
        meta={<StatusPill status={isTrainActive ? activeRun?.status || "running" : "stopped"} />}
      />
      <div className="train-workspace">
        {error ? <div className="alert error">{error}</div> : null}
        {status.active && !activeRun ? <div className="alert">Job запускается...</div> : null}
        {activeRun && activeRunType !== "train" ? (
          <div className="alert">Другой job уже выполняется: {activeRunType}</div>
        ) : null}

        <div className="train-stage">
          <section className="train-control-square">
            <div className="train-square-head">
              <div>
                <h2>Control</h2>
                <p>Config lives in Settings.</p>
              </div>
              <StatusPill status={isTrainActive ? activeRun?.status || "running" : "stopped"} />
            </div>

            <div className="settings-summary-grid compact train-summary-grid">
              {trainingSummary.map((item) => (
                <div key={`${item.label}:${item.value}`} className="settings-summary-item">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>

            {isTrainActive && progressMax ? (
              <div className="train-progress-block">
                <div className="progress">
                  <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
                </div>
                <div className="muted">step {formatStep(lastMetric)}</div>
              </div>
            ) : (
              <div className="empty-state train-idle-state">
                <strong>No training run</strong>
                <span>Select an optional checkpoint and start training.</span>
              </div>
            )}

            {lastMetric ? (
              <div className="compact-metrics train-metrics-grid">
                <MetricTile label="step" value={formatStep(lastMetric)} />
                <MetricTile label="elapsed" value={metricElapsed(lastMetric)} />
                <MetricTile label="ETA" value={metricEta(lastMetric)} />
                <MetricTile
                  label="s/step"
                  value={Number.isFinite(metricSecondsPerStep(lastMetric)) ? metricSecondsPerStep(lastMetric).toFixed(3) : "—"}
                />
                <MetricTile
                  label="samples/s"
                  value={Number.isFinite(metricThroughput(lastMetric)) ? metricThroughput(lastMetric).toFixed(2) : "—"}
                />
                <MetricTile
                  label="loss"
                  value={Number.isFinite(metricLoss(lastMetric)) ? metricLoss(lastMetric).toFixed(4) : "—"}
                />
              </div>
            ) : null}

            {command.length > 0 ? <div className="muted">Training command prepared.</div> : null}
          </section>

          <section className="train-monitor-square">
            <div className="train-square-head">
              <div>
                <h2>Monitor</h2>
                <p>{lastMetric ? `step ${formatStep(lastMetric)}` : "Waiting for metrics."}</p>
              </div>
              <Link to="/train/samples" className="text-link">
                Samples
              </Link>
            </div>
            <div className="train-chart-panel">
              <LineChart data={chartData} />
            </div>
            <div className="train-log-panel">
              <div className="train-log-head">
                <span>Live logs</span>
                <button className="ghost" onClick={clearLogs}>
                  Clear
                </button>
              </div>
              <LogViewer lines={logLines} />
              <div className="muted">Buffer: {logLines.length} / 10k lines</div>
            </div>
          </section>
        </div>

        <section className="train-compose-panel">
          <Link className="secondary" to="/settings">
            Settings
          </Link>
          <label className="train-resume-field">
            <span>Resume</span>
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
          </label>
          <button onClick={handleStart} disabled={status.active}>
            Start
          </button>
          <button className="danger" onClick={handleStop} disabled={!isTrainActive}>
            Stop
          </button>
        </section>
      </div>
    </div>
  );
}
