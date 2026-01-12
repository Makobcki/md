import React, { useEffect, useState } from "react";
import { api } from "../api.js";

export default function FilesPage() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        const data = await api.getOutDirSummary();
        setSummary(data);
      } catch (err) {
        setError(err.message);
      }
    };
    load();
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, []);

  if (!summary) {
    return <div className="container">{error ? error : "Loading..."}</div>;
  }

  return (
    <div className="container">
      <div className="card">
        <h2>Files / Logs</h2>
        <div className="muted">out_dir: {summary.out_dir}</div>
      </div>

      <div className="grid">
        <div className="card">
        <h3>metrics/events.jsonl (tail)</h3>
          <pre className="log-box">{summary.train_log.tail || "No log yet"}</pre>
        </div>
        <div className="card">
          <h3>config_snapshot.yaml</h3>
          <pre className="log-box">{summary.config_snapshot.content || "No config snapshot"}</pre>
        </div>
        <div className="card">
          <h3>run_meta.yaml</h3>
          <pre className="log-box">{summary.run_meta.content || "No run meta yet"}</pre>
        </div>
      </div>

      <div className="card">
        <h3>Checkpoints</h3>
        {summary.checkpoints.length === 0 ? (
          <div className="muted">No checkpoints yet</div>
        ) : (
          <ul>
            {summary.checkpoints.map((ckpt) => (
              <li key={ckpt}>{ckpt}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
