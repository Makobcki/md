import React, { useEffect, useMemo, useState } from "react";
import { api, wsUrl, API_ORIGIN } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";

export default function GeneratePage() {
  const [argSpecs, setArgSpecs] = useState([]);
  const [args, setArgs] = useState({});
  const [checkpoints, setCheckpoints] = useState([]);
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState({ active: false });
  const [runId, setRunId] = useState(null);
  const [command, setCommand] = useState([]);
  const [output, setOutput] = useState(null);
  const [error, setError] = useState("");
  const [gallery, setGallery] = useState([]);
  const [metrics, setMetrics] = useState([]);

  useEffect(() => {
    const load = async () => {
      const [argsData, ckptData, samples] = await Promise.all([
        api.getSampleArgs(),
        api.listCheckpoints(),
        api.listSamples(),
      ]);
      setArgSpecs(argsData.items);
      setCheckpoints(ckptData.items);
      setGallery(samples.items);

      const initial = {};
      argsData.items.forEach((spec) => {
        if (spec.name === "ckpt" && ckptData.items.length > 0) {
          initial[spec.name] = ckptData.items[0];
        } else if (spec.default !== null && spec.default !== undefined) {
          initial[spec.name] = spec.default;
        } else {
          initial[spec.name] = "";
        }
      });
      setArgs(initial);
    };
    load();
  }, []);

  useEffect(() => {
    const refresh = async () => {
      const samples = await api.listSamples();
      setGallery(samples.items);
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

  const handleChange = (name, value) => {
    setArgs((prev) => ({ ...prev, [name]: value }));
  };

  const handleStart = async () => {
    setError("");
    try {
      const resp = await api.startSample(args);
      setRunId(resp.run_id);
      setCommand(resp.command);
      setOutput(resp.output);
      setLogs([]);
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

  const toRunsUrl = (path) => {
    if (!path) return null;
    const marker = "webui_runs/";
    const idx = path.indexOf(marker);
    if (idx === -1) return null;
    return `${API_ORIGIN}/runs/${path.slice(idx + marker.length)}`;
  };

  return (
    <div className="container">
      <div className="card">
        <h2>Generate / Sample</h2>
        {error && <div className="muted">{error}</div>}
        {status.active && status.run.run_type !== "sample" && (
          <div className="muted">Другой job уже выполняется: {status.run.run_type}</div>
        )}
        <div className="row">
          <button onClick={handleStart} disabled={status.active}>Start</button>
          <button className="danger" onClick={handleStop} disabled={!status.active}>Stop</button>
          <span className="status-pill">{status.active ? "running" : "stopped"}</span>
        </div>
        {status.active && status.run.run_type === "sample" && (
          <progress
            style={{ width: "100%" }}
            value={progressMax ? progressValue : undefined}
            max={progressMax || undefined}
          />
        )}
        {commandText && <div className="muted">Command: {commandText}</div>}
      </div>

      <div className="card">
        <h3>Arguments</h3>
        <div className="grid">
          {argSpecs.map((spec) => (
            <div key={spec.name} className="card">
              <div className="muted">{spec.flags.join(", ")}</div>
              <label>{spec.name}</label>
              {spec.name === "ckpt" && checkpoints.length > 0 ? (
                <select
                  value={args[spec.name] || ""}
                  onChange={(e) => handleChange(spec.name, e.target.value)}
                >
                  {checkpoints.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              ) : spec.name === "ckpt" ? (
                <input
                  type="text"
                  value={args[spec.name]}
                  onChange={(e) => handleChange(spec.name, e.target.value)}
                />
              ) : spec.type === "int" || spec.type === "float" ? (
                <input
                  type="number"
                  value={args[spec.name]}
                  onChange={(e) => handleChange(spec.name, e.target.value)}
                />
              ) : (
                <input
                  type="text"
                  value={args[spec.name]}
                  onChange={(e) => handleChange(spec.name, e.target.value)}
                />
              )}
              <div className="muted">{spec.help}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <h3>Live logs</h3>
        <LogViewer lines={logs} />
      </div>

      <div className="card">
        <h3>Latest output</h3>
        {output && toRunsUrl(output) ? (
          <div className="gallery">
            <img src={toRunsUrl(output)} alt="sample" />
          </div>
        ) : (
          <div className="muted">No output yet</div>
        )}
      </div>

      <div className="card">
        <h3>Gallery</h3>
        <div className="grid gallery">
          {gallery.map((item) => (
            <img key={item} src={toRunsUrl(item)} alt="sample" />
          ))}
        </div>
      </div>
    </div>
  );
}
