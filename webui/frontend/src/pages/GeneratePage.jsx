import React, { useEffect, useMemo, useState } from "react";
import { api, wsUrl, API_ORIGIN } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";

const minimalFields = [
  "prompt",
  "neg_prompt",
  "neg",
  "ckpt",
  "steps",
  "seed",
  "width",
  "height",
  "guidance",
];

const toRunsUrl = (path) => {
  if (!path) return null;
  const marker = "webui_runs/";
  const idx = path.indexOf(marker);
  if (idx === -1) return null;
  return `${API_ORIGIN}/runs/${path.slice(idx + marker.length)}`;
};

const statusClass = (status) => `status-pill ${status || ""}`;

export default function GeneratePage() {
  const [argSpecs, setArgSpecs] = useState([]);
  const [args, setArgs] = useState({});
  const [checkpoints, setCheckpoints] = useState([]);
  const [status, setStatus] = useState({ active: false });
  const [runId, setRunId] = useState(null);
  const [command, setCommand] = useState([]);
  const [output, setOutput] = useState(null);
  const [error, setError] = useState("");
  const [gallery, setGallery] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [imageOnly, setImageOnly] = useState(false);
  const [textConditioningAvailable, setTextConditioningAvailable] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);

  const logKey = runId ? `generate:logs:${runId}` : "generate:logs:idle";
  const { lines: logLines, appendLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const [argsData, ckptData, samples] = await Promise.all([
        api.getSampleArgs(),
        api.listCheckpoints(),
        api.listSamples(),
      ]);
      setArgSpecs(argsData.items || []);
      setCheckpoints(ckptData.items || []);
      setGallery(samples.items || []);

      const initial = {};
      (argsData.items || []).forEach((spec) => {
        if (spec.name === "ckpt" && ckptData.items?.length > 0) {
          initial[spec.name] = ckptData.items[0];
        } else if (spec.default !== null && spec.default !== undefined) {
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
    const ckpt = args.ckpt;
    if (!ckpt) {
      setTextConditioningAvailable(true);
      return;
    }
    api
      .getCheckpointInfo(ckpt)
      .then((info) => {
        const available = info.use_text_conditioning !== false;
        setTextConditioningAvailable(available);
        if (!available) {
          setImageOnly(true);
        }
      })
      .catch(() => setTextConditioningAvailable(true));
  }, [args.ckpt]);

  useEffect(() => {
    const refresh = async () => {
      const samples = await api.listSamples();
      setGallery(samples.items || []);
    };
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const poll = async () => {
      const stat = await api.getStatus();
      setStatus(stat);
      if (stat.active && stat.run.run_type === "sample") {
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

  const handleStart = async () => {
    setError("");
    try {
      const payload = { ...args };
      if (imageOnly || !textConditioningAvailable) {
        payload.prompt = "";
        payload.neg_prompt = "";
        payload.neg = "";
      }
      const resp = await api.startSample(payload);
      setRunId(resp.run_id);
      setCommand(resp.command);
      setOutput(resp.output);
      clearLogs();
      setMetrics([]);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStop = async () => {
    setError("");
    try {
      await api.stopSample();
    } catch (err) {
      setError(err.message);
    }
  };

  const commandText = useMemo(() => command.join(" "), [command]);
  const lastMetric = metrics[metrics.length - 1];
  const progressMax = lastMetric?.max_steps || 0;
  const progressValue = lastMetric?.step ?? 0;

  const minimalSpecs = useMemo(
    () => argSpecs.filter((spec) => minimalFields.includes(spec.name)),
    [argSpecs]
  );
  const advancedSpecs = useMemo(
    () => argSpecs.filter((spec) => !minimalFields.includes(spec.name)),
    [argSpecs]
  );

  const renderField = (spec) => {
    const disabled = (imageOnly || !textConditioningAvailable) &&
      ["prompt", "neg_prompt", "neg"].includes(spec.name);

    if (spec.name === "ckpt" && checkpoints.length > 0) {
      return (
        <select
          value={args[spec.name] || ""}
          onChange={(event) => handleChange(spec.name, event.target.value)}
        >
          {checkpoints.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      );
    }

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
        disabled={disabled}
      />
    );
  };

  return (
    <div className="page">
      <h1 className="page-title">Generate</h1>
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
            {status.active && status.run.run_type !== "sample" && (
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
            {status.active && status.run.run_type === "sample" && progressMax ? (
              <div className="progress">
                <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
              </div>
            ) : null}
            {commandText && <div className="muted">Command: {commandText}</div>}
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Minimal Settings</h2>
              <div className="toggle-group">
                <button
                  className={!imageOnly ? "active" : ""}
                  onClick={() => setImageOnly(false)}
                  disabled={!textConditioningAvailable}
                >
                  Text2Image
                </button>
                <button className={imageOnly ? "active" : ""} onClick={() => setImageOnly(true)}>
                  Image-only
                </button>
              </div>
            </div>
            {!textConditioningAvailable && (
              <div className="muted">Checkpoint without text conditioning. Prompt disabled.</div>
            )}
            <div className="grid">
              {minimalSpecs.map((spec) => (
                <div key={spec.name} className="card soft">
                  <div className="muted">{spec.flags.join(", ")}</div>
                  <label>{spec.name}</label>
                  {renderField(spec)}
                  <div className="muted">{spec.help}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <details>
              <summary className="card-title">Advanced Settings</summary>
              <div className="grid" style={{ marginTop: "12px" }}>
                {advancedSpecs.map((spec) => (
                  <div key={spec.name} className="card soft">
                    <div className="muted">{spec.flags.join(", ")}</div>
                    <label>{spec.name}</label>
                    {renderField(spec)}
                    <div className="muted">{spec.help}</div>
                  </div>
                ))}
              </div>
            </details>
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

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Latest Output</h2>
            </div>
            {output && toRunsUrl(output) ? (
              <div className="image-card">
                <img src={toRunsUrl(output)} alt="sample" />
                <div className="image-meta">
                  <a href={toRunsUrl(output)} target="_blank" rel="noreferrer">
                    Open full
                  </a>
                  <a href={toRunsUrl(output)} download>
                    Download
                  </a>
                </div>
              </div>
            ) : (
              <div className="muted">No output yet</div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Gallery</h2>
            </div>
            <div className="gallery-grid">
              {[...gallery].reverse().map((item) => {
                const url = toRunsUrl(item);
                return url ? (
                  <div key={item} className="image-card">
                    <img src={url} alt="sample" />
                    <div className="image-meta">
                      <span className="badge">{item.split("/").pop()}</span>
                      <div className="row">
                        <a href={url} target="_blank" rel="noreferrer">
                          Open full
                        </a>
                        <a href={url} download>
                          Download
                        </a>
                      </div>
                    </div>
                  </div>
                ) : null;
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
