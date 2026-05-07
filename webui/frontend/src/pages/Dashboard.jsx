import React, { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, absoluteFileUrl, wsUrl } from "../api.js";
import LineChart from "../components/LineChart.jsx";
import MetricTile from "../components/MetricTile.jsx";
import PageHeader from "../components/PageHeader.jsx";
import StatusPill from "../components/StatusPill.jsx";
import {
  formatDate,
  formatRelativeDate,
  formatRunId,
  formatRunType,
  parseRunDate,
} from "../utils/formatters.js";
import {
  isMetricEvent,
  mergeMetricEvents,
  metricChartData,
  metricStep,
  metricThroughput,
} from "../utils/metrics.js";
import {
  buildRunInventory,
  recentPreviewItems,
  sortAndFilterRuns,
  workflowCards,
} from "../utils/uiModel.js";

export default function Dashboard() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [status, setStatus] = useState({ active: false });
  const [metrics, setMetrics] = useState([]);
  const [samples, setSamples] = useState([]);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");
  const [runFilters, setRunFilters] = useState({ failedOnly: false, todayOnly: false });
  const [sortDir, setSortDir] = useState("desc");

  useEffect(() => {
    const load = async () => {
      try {
        const [runsData, statusData, samplesData, summaryData] = await Promise.all([
          api.listRuns(),
          api.getStatus(),
          api.listSamples(),
          api.getOutDirSummary().catch(() => null),
        ]);
        setRuns(runsData);
        setStatus(statusData);
        setSamples(samplesData.items || []);
        setSummary(summaryData);
        setError("");
      } catch (err) {
        setError(err.message);
      }
    };
    load();
    const timer = setInterval(load, 3000);
    return () => clearInterval(timer);
  }, []);

  const activeRun = status.active ? status.run : null;

  useEffect(() => {
    if (!activeRun?.run_id) return undefined;
    const ws = new WebSocket(wsUrl(`/ws/metrics/${activeRun.run_id}`));
    ws.onmessage = (event) => {
      try {
        const metric = JSON.parse(event.data);
        if (isMetricEvent(metric)) {
          setMetrics((prev) => mergeMetricEvents(prev, [metric], 500));
        }
      } catch (err) {
        console.warn(err);
      }
    };
    return () => ws.close();
  }, [activeRun?.run_id]);

  const handleStop = async () => {
    if (!activeRun) return;
    setError("");
    try {
      if (activeRun.run_type === "train") {
        await api.stopTrain();
      } else if (activeRun.run_type === "sample") {
        await api.stopSample();
      } else if (activeRun.run_type === "latent_cache") {
        await api.stopLatentCache();
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const lastMetric = metrics[metrics.length - 1];
  const progressValue = metricStep(lastMetric) ?? lastMetric?.processed ?? 0;
  const progressMax = lastMetric?.max_steps || lastMetric?.total || 0;
  const recentSamples = useMemo(() => recentPreviewItems(samples, 10), [samples]);
  const inventory = useMemo(() => buildRunInventory(runs), [runs]);
  const visibleRuns = useMemo(
    () =>
      sortAndFilterRuns(runs, {
        ...runFilters,
        sortDir,
      }),
    [runs, runFilters, sortDir]
  );

  return (
    <div className="page dashboard-page">
      <PageHeader
        eyebrow="Operations"
        title="Dashboard"
        description="Single control surface for generation, training, cache preparation, artifacts and run history."
        meta={<StatusPill status={activeRun?.status || "stopped"} />}
      />

      {error ? <div className="alert error">{error}</div> : null}

      <section className="overview-grid">
        <MetricTile label="Runs" value={inventory.total} detail="total recorded jobs" />
        <MetricTile label="Active" value={inventory.running} detail="currently running" tone="success" />
        <MetricTile label="Failed" value={inventory.failed} detail="needs inspection" tone={inventory.failed ? "danger" : ""} />
        <MetricTile
          label="Artifacts"
          value={summary?.sample_count ?? recentSamples.length}
          detail={summary?.out_dir ? "from configured out_dir" : "recent previews"}
        />
      </section>

      <section className="dashboard-grid">
        <div className="panel active-run-panel">
          <div className="panel-header">
            <div>
              <h2>Runtime</h2>
              <p>Current job state and latest live metrics.</p>
            </div>
            {activeRun ? (
              <button type="button" className="danger" onClick={handleStop}>
                Stop
              </button>
            ) : null}
          </div>
          {activeRun ? (
            <div className="active-run-body">
              <div className="run-identity">
                <StatusPill status={activeRun.status} />
                <div>
                  <strong>{formatRunType(activeRun.run_type)}</strong>
                  <span title={activeRun.run_id}>{formatRunId(activeRun.run_id)}</span>
                </div>
              </div>
              {progressMax ? (
                <div className="progress">
                  <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
                </div>
              ) : (
                <div className="muted">Progress will appear after the first metric event.</div>
              )}
              <div className="compact-metrics">
                <MetricTile label="Step" value={metricStep(lastMetric) ?? "-"} />
                <MetricTile label="ETA" value={lastMetric?.eta_h ? `${lastMetric.eta_h.toFixed(2)}h` : "-"} />
                <MetricTile
                  label="Speed"
                  value={Number.isFinite(metricThroughput(lastMetric)) ? `${metricThroughput(lastMetric).toFixed(2)}/s` : "-"}
                />
                <MetricTile label="VRAM" value={lastMetric?.peak_mem_mb ? `${lastMetric.peak_mem_mb.toFixed(0)} MB` : "-"} />
              </div>
              {activeRun.run_type === "train" ? <LineChart data={metricChartData(metrics)} /> : null}
              <Link className="text-link" to={`/runs/${activeRun.run_id}`}>
                Open run details
              </Link>
            </div>
          ) : (
            <div className="empty-state">
              <strong>No job running</strong>
              <span>Start generation, training or latent preparation from workflow cards.</span>
            </div>
          )}
        </div>

        <div className="workflow-grid">
          {workflowCards.map((card) => (
            <Link key={card.key} className="workflow-card" to={card.to}>
              <div>
                <h2>{card.title}</h2>
                <p>{card.description}</p>
              </div>
              <span>{card.action}</span>
            </Link>
          ))}
        </div>
      </section>

      <section className="content-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <h2>Recent Generations</h2>
              <p>Latest previewable images from sample artifacts.</p>
            </div>
            <Link className="text-link" to="/generate">
              Generate
            </Link>
          </div>
          {recentSamples.length === 0 ? (
            <div className="empty-state">No previewable generated images yet.</div>
          ) : (
            <div className="gallery-grid dense">
              {recentSamples.map((item) => {
                const url = absoluteFileUrl(item);
                return url ? (
                  <a key={item.path || item.url || item} className="image-card" href={url} target="_blank" rel="noreferrer">
                    <img src={url} alt="sample" />
                    <div className="image-meta">
                      <span className="badge">{String(item.path || item).split("/").pop()}</span>
                    </div>
                  </a>
                ) : null;
              })}
            </div>
          )}
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h2>Runs</h2>
              <p>
                {inventory.training} training, {inventory.generation} generation, {inventory.latentCache} cache.
              </p>
            </div>
            <div className="row">
              <label className="chip-control">
                <input
                  type="checkbox"
                  checked={runFilters.failedOnly}
                  onChange={(event) =>
                    setRunFilters((prev) => ({ ...prev, failedOnly: event.target.checked }))
                  }
                />
                Failed
              </label>
              <label className="chip-control">
                <input
                  type="checkbox"
                  checked={runFilters.todayOnly}
                  onChange={(event) =>
                    setRunFilters((prev) => ({ ...prev, todayOnly: event.target.checked }))
                  }
                />
                Today
              </label>
              <button
                className="ghost"
                type="button"
                onClick={() => setSortDir((prev) => (prev === "desc" ? "asc" : "desc"))}
              >
                {sortDir === "desc" ? "Newest" : "Oldest"}
              </button>
            </div>
          </div>
          <div className="table-scroll">
            <table className="table runs-table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Type</th>
                  <th>Created</th>
                  <th>Run</th>
                </tr>
              </thead>
              <tbody>
                {visibleRuns.map((run) => {
                  const created = parseRunDate(run);
                  return (
                    <tr
                      key={run.run_id}
                      className="clickable-row"
                      onClick={() => navigate(`/runs/${run.run_id}`)}
                    >
                      <td>
                        <StatusPill status={run.status} title={run.status} />
                      </td>
                      <td>{formatRunType(run.run_type)}</td>
                      <td title={created ? formatDate(created) : run.created_at}>
                        {created ? formatRelativeDate(created) : "-"}
                      </td>
                      <td className="muted" title={run.run_id}>
                        {formatRunId(run.run_id)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </div>
  );
}
