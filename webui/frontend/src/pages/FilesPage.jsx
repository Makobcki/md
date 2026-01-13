import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";

const parseLogLine = (line) => {
  const trimmed = line.trim();
  if (!trimmed.startsWith("{")) {
    return { raw: line, level: "info", timestamp: null };
  }
  try {
    const obj = JSON.parse(trimmed);
    const level = obj.level || obj.lvl || obj.severity || "info";
    const timestamp = obj.timestamp || obj.time || obj.ts || null;
    return { raw: line, level: String(level).toLowerCase(), timestamp };
  } catch (err) {
    return { raw: line, level: "info", timestamp: null };
  }
};

export default function FilesPage() {
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [stdout, setStdout] = useState([]);
  const [stderr, setStderr] = useState([]);
  const [stream, setStream] = useState("stdout");
  const [level, setLevel] = useState("all");
  const [query, setQuery] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [timeStart, setTimeStart] = useState("");
  const [timeEnd, setTimeEnd] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        const data = await api.listRuns();
        setRuns(data);
        if (!selectedRunId && data.length > 0) {
          setSelectedRunId(data[0].run_id);
        }
      } catch (err) {
        setError(err.message);
      }
    };
    load();
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, [selectedRunId]);

  useEffect(() => {
    if (!selectedRunId) return;
    const loadLogs = async () => {
      try {
        const [outLog, errLog] = await Promise.all([
          api.getRunLog(selectedRunId, "stdout"),
          api.getRunLog(selectedRunId, "stderr"),
        ]);
        setStdout(outLog.content.split("\n").filter(Boolean));
        setStderr(errLog.content.split("\n").filter(Boolean));
      } catch (err) {
        setError(err.message);
      }
    };
    loadLogs();
  }, [selectedRunId]);

  const filteredLines = useMemo(() => {
    const source = stream === "stderr" ? stderr : stdout;
    const parsed = source.map(parseLogLine);
    return parsed
      .filter((item) => (level === "all" ? true : item.level === level))
      .filter((item) => (query ? item.raw.toLowerCase().includes(query.toLowerCase()) : true))
      .filter((item) => {
        if (!timeStart && !timeEnd) return true;
        if (!item.timestamp) return false;
        const timestamp = new Date(item.timestamp).getTime();
        if (Number.isNaN(timestamp)) return false;
        if (timeStart && timestamp < new Date(timeStart).getTime()) return false;
        if (timeEnd && timestamp > new Date(timeEnd).getTime()) return false;
        return true;
      })
      .map((item) => item.raw);
  }, [stdout, stderr, stream, level, query, timeStart, timeEnd]);

  return (
    <div className="page">
      <h1 className="page-title">Files / Logs</h1>
      {error && <div className="muted">{error}</div>}
      <div className="split">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Runs</h2>
          </div>
          {runs.length === 0 ? (
            <div className="muted">No runs yet.</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Type</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr
                    key={run.run_id}
                    onClick={() => setSelectedRunId(run.run_id)}
                    style={{ cursor: "pointer", background: run.run_id === selectedRunId ? "#0b1224" : "transparent" }}
                  >
                    <td>
                      <span className={`status-pill ${run.status}`}>{run.status}</span>
                    </td>
                    <td>{run.run_type}</td>
                    <td className="muted">{run.created_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="page">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Logs Viewer</h2>
              {selectedRunId && (
                <Link className="muted" to={`/runs/${selectedRunId}`}>
                  Open run
                </Link>
              )}
            </div>
            <div className="log-controls">
              <label className="row">
                Stream
                <select value={stream} onChange={(event) => setStream(event.target.value)}>
                  <option value="stdout">stdout</option>
                  <option value="stderr">stderr</option>
                </select>
              </label>
              <label className="row">
                Level
                <select value={level} onChange={(event) => setLevel(event.target.value)}>
                  <option value="all">all</option>
                  <option value="info">info</option>
                  <option value="warn">warn</option>
                  <option value="error">error</option>
                </select>
              </label>
              <label className="row">
                Search
                <input value={query} onChange={(event) => setQuery(event.target.value)} />
              </label>
              <label className="row">
                From
                <input
                  type="datetime-local"
                  value={timeStart}
                  onChange={(event) => setTimeStart(event.target.value)}
                />
              </label>
              <label className="row">
                To
                <input
                  type="datetime-local"
                  value={timeEnd}
                  onChange={(event) => setTimeEnd(event.target.value)}
                />
              </label>
              <label className="row">
                <input
                  type="checkbox"
                  checked={autoScroll}
                  onChange={(event) => setAutoScroll(event.target.checked)}
                />
                Auto-scroll
              </label>
            </div>
            <LogViewer lines={filteredLines} autoScroll={autoScroll} />
            <div className="muted">Showing {filteredLines.length} lines</div>
          </div>
        </div>
      </div>
    </div>
  );
}
