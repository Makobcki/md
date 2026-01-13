import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, API_ORIGIN } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";

const toRunsUrl = (path) => {
  if (!path) return null;
  const marker = "webui_runs/";
  const idx = path.indexOf(marker);
  if (idx === -1) return null;
  return `${API_ORIGIN}/runs/${path.slice(idx + marker.length)}`;
};

export default function RunDetails() {
  const { runId } = useParams();
  const [run, setRun] = useState(null);
  const [config, setConfig] = useState("");
  const [stdout, setStdout] = useState([]);
  const [stderr, setStderr] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [samples, setSamples] = useState([]);

  useEffect(() => {
    const load = async () => {
      const runData = await api.getRun(runId);
      setRun(runData);
      if (runData.config_snapshot) {
        const cfg = await api.getRunConfig(runId);
        setConfig(cfg.content);
      }
      const outLog = await api.getRunLog(runId, "stdout");
      const errLog = await api.getRunLog(runId, "stderr");
      setStdout(outLog.content.split("\n").filter(Boolean));
      setStderr(errLog.content.split("\n").filter(Boolean));
      const metricsData = await api.getRunMetrics(runId);
      setMetrics(metricsData.items || []);
      const ckptData = await api.listCheckpoints();
      setCheckpoints(ckptData.items || []);
      const sampleData = await api.listSamples();
      setSamples(
        (sampleData.items || []).filter((path) => path.includes(`/webui_runs/${runId}/samples/`))
      );
    };
    load();
  }, [runId]);

  if (!run) {
    return <div className="card">Loading...</div>;
  }

  return (
    <div className="page">
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Run {runId}</h2>
          <span className={`status-pill ${run.status}`}>{run.status}</span>
        </div>
        <div className="row">
          <span>{run.run_type}</span>
          <span className="muted">{run.created_at}</span>
        </div>
        <div className="muted">Command: {run.command.join(" ")}</div>
        {run.output_path && <div className="muted">Output: {run.output_path}</div>}
      </div>

      {config && (
        <div className="card">
          <h3 className="card-title">Config snapshot</h3>
          <textarea value={config} rows={16} readOnly style={{ width: "100%" }} />
        </div>
      )}

      {metrics.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Loss vs Step</h3>
            <span className="muted">avg log_every</span>
          </div>
          <LineChart data={metrics.map((m) => ({ step: m.step, loss: m.loss }))} />
        </div>
      )}

      {checkpoints.length > 0 && (
        <div className="card">
          <h3 className="card-title">Checkpoints</h3>
          <ul>
            {checkpoints.map((ckpt) => (
              <li key={ckpt} className="muted">
                {ckpt}
              </li>
            ))}
          </ul>
        </div>
      )}

      {samples.length > 0 && (
        <div className="card">
          <h3 className="card-title">Samples</h3>
          <div className="gallery-grid">
            {samples.map((item) => {
              const url = toRunsUrl(item);
              return url ? (
                <div key={item} className="image-card">
                  <img src={url} alt="sample" />
                  <div className="image-meta">
                    <a href={url} target="_blank" rel="noreferrer">
                      Open full
                    </a>
                  </div>
                </div>
              ) : null;
            })}
          </div>
        </div>
      )}

      <div className="grid">
        <div className="card">
          <h3 className="card-title">stdout</h3>
          <LogViewer lines={stdout} autoScroll={false} />
        </div>
        <div className="card">
          <h3 className="card-title">stderr</h3>
          <LogViewer lines={stderr} autoScroll={false} />
        </div>
      </div>
    </div>
  );
}
