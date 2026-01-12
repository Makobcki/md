import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";

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
      setStdout(outLog.content.split("\n"));
      setStderr(errLog.content.split("\n"));
      const metricsData = await api.getRunMetrics(runId);
      setMetrics(metricsData.items || []);
      const ckptData = await api.listCheckpoints();
      setCheckpoints(ckptData.items || []);
      const sampleData = await api.listSamples();
      setSamples(
        (sampleData.items || []).filter((path) =>
          path.includes(`/webui_runs/${runId}/samples/`)
        )
      );
    };
    load();
  }, [runId]);

  if (!run) {
    return <div className="card">Loading...</div>;
  }

  return (
    <div className="container">
      <div className="card">
        <h2>Run {runId}</h2>
        <div className="row">
          <span className="status-pill">{run.status}</span>
          <span>{run.run_type}</span>
          <span className="muted">{run.created_at}</span>
        </div>
        <div className="muted">Command: {run.command.join(" ")}</div>
        {run.output_path && <div className="muted">Output: {run.output_path}</div>}
      </div>

      {config && (
        <div className="card">
          <h3>Config snapshot</h3>
          <textarea value={config} rows={16} readOnly style={{ width: "100%" }} />
        </div>
      )}

      {metrics.length > 0 && (
        <div className="card">
          <h3>Loss vs Step (avg log_every)</h3>
          <div className="muted">Среднее значение loss за последний интервал логирования.</div>
          <LineChart data={metrics.map((m) => ({ step: m.step, loss: m.loss }))} />
        </div>
      )}

      {checkpoints.length > 0 && (
        <div className="card">
          <h3>Checkpoints</h3>
          <ul>
            {checkpoints.map((ckpt) => (
              <li key={ckpt} className="muted">{ckpt}</li>
            ))}
          </ul>
        </div>
      )}

      {samples.length > 0 && (
        <div className="card">
          <h3>Samples</h3>
          <div className="grid gallery">
            {samples.map((item) => {
              const marker = "webui_runs/";
              const idx = item.indexOf(marker);
              const url = idx === -1 ? null : `http://127.0.0.1:8000/runs/${item.slice(idx + marker.length)}`;
              return url ? <img key={item} src={url} alt="sample" /> : null;
            })}
          </div>
        </div>
      )}

      <div className="grid">
        <div className="card">
          <h3>stdout</h3>
          <LogViewer lines={stdout} />
        </div>
        <div className="card">
          <h3>stderr</h3>
          <LogViewer lines={stderr} />
        </div>
      </div>
    </div>
  );
}
