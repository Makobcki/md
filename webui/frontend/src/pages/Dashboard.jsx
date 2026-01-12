import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

export default function Dashboard() {
  const [runs, setRuns] = useState([]);
  const [status, setStatus] = useState({ active: false });

  useEffect(() => {
    const load = async () => {
      const [runsData, statusData] = await Promise.all([api.listRuns(), api.getStatus()]);
      setRuns(runsData);
      setStatus(statusData);
    };
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="container">
      <div className="card">
        <h2>Текущий статус</h2>
        {status.active ? (
          <div className="row">
            <span className="status-pill">Running</span>
            <span>{status.run.run_type}</span>
            <Link to={`/runs/${status.run.run_id}`}>Open run</Link>
          </div>
        ) : (
          <span className="muted">Нет активных задач</span>
        )}
      </div>

      <div className="card">
        <h2>Runs</h2>
        <div className="grid">
          {runs.map((run) => (
            <div key={run.run_id} className="card">
              <div className="row">
                <span className="status-pill">{run.status}</span>
                <strong>{run.run_type}</strong>
              </div>
              <div className="muted">{run.created_at}</div>
              <div className="muted">{run.run_id}</div>
              <Link to={`/runs/${run.run_id}`}>Details</Link>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
