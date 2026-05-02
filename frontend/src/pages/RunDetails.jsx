import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, API_ORIGIN } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";
import StatusPill from "../components/StatusPill.jsx";
import { formatDate, formatRunId, formatRunType, parseRunDate } from "../utils/formatters.js";

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
          <h2 className="card-title" title={runId}>
            {formatRunId(runId)}
          </h2>
          <StatusPill status={run.status} />
        </div>
        <div className="row">
          <span>{formatRunType(run.run_type)}</span>
          <span className="muted" title={run.created_at}>
            {formatDate(parseRunDate(run))}
          </span>
        </div>
        <div className="muted">Run settings captured.</div>
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
          <LogViewer lines={stdout} />
        </div>
        <div className="card">
          <h3 className="card-title">stderr</h3>
          <LogViewer lines={stderr} />
        </div>
      </div>
    </div>
  );
}
