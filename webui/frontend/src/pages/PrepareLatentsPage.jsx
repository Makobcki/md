import React, { useEffect, useMemo, useState } from "react";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";
import ArgField from "../components/ArgField.jsx";
import StatusPill from "../components/StatusPill.jsx";
import { isMetricEvent, mergeMetricEvents } from "../utils/metrics.js";

const settingGroups = [
  {
    title: "Файловая система / Кэш",
    names: ["config", "limit", "shard-size", "decode-backend"],
  },
  {
    title: "Производительность",
    names: [
      "batch-size",
      "num-workers",
      "prefetch-factor",
      "pin-memory",
      "queue-size",
      "writer-threads",
      "stats-every-sec",
    ],
  },
  {
    title: "GPU",
    names: ["device", "latent-dtype", "autocast-dtype"],
  },
];

export default function PrepareLatentsPage() {
  const [argSpecs, setArgSpecs] = useState([]);
  const [args, setArgs] = useState({});
  const [status, setStatus] = useState({ active: false });
  const [runId, setRunId] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [command, setCommand] = useState([]);
  const [error, setError] = useState("");

  const logKey = runId ? `latents:logs:${runId}` : "latents:logs:idle";
  const { lines: logLines, appendLines, replaceLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const argsData = await api.getLatentArgs();
      setArgSpecs(argsData.items || []);

      const initial = {};
      (argsData.items || []).forEach((spec) => {
        if (spec.default !== null && spec.default !== undefined) {
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
    const poll = async () => {
      const stat = await api.getStatus();
      setStatus(stat);
      if (stat.active && stat.run?.run_type === "latent_cache") {
        setRunId(stat.run.run_id);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    const loadLogs = async () => {
      try {
        const [stdout, stderr] = await Promise.all([
          api.getRunLog(runId, "stdout"),
          api.getRunLog(runId, "stderr"),
        ]);
        if (cancelled) return;
        replaceLines([
          ...stdout.content.split("\n").filter(Boolean).map((line) => `[stdout] ${line}`),
          ...stderr.content.split("\n").filter(Boolean).map((line) => `[stderr] ${line}`),
        ]);
      } catch (err) {
        console.warn("failed to load log tail", err);
      }
    };
    loadLogs();
    const timer = setInterval(loadLogs, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId, replaceLines]);

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(wsUrl(`/ws/logs/${runId}?backlog=0`));
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
        if (isMetricEvent(metric)) {
          setMetrics((prev) => mergeMetricEvents(prev, [metric], 500));
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

  const handleStart = async (overrideArgs = null) => {
    setError("");
    try {
      const resp = await api.startLatentCache(overrideArgs || args);
      setRunId(resp.run_id);
      setCommand(resp.command || []);
      clearLogs();
      setMetrics([]);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleRebuild = async () => {
    const nextArgs = { ...args, overwrite: true };
    setArgs(nextArgs);
    await handleStart(nextArgs);
  };

  const handleStop = async () => {
    setError("");
    try {
      await api.stopLatentCache();
    } catch (err) {
      setError(err.message);
    }
  };

  const lastMetric = metrics[metrics.length - 1];
  const progressMax = lastMetric?.max_steps || 0;
  const progressValue = lastMetric?.processed ?? 0;
  const activeRun = status.active ? status.run : null;
  const activeRunType = activeRun?.run_type;

  const overview = useMemo(() => {
    if (!lastMetric) return [];
    return [
      { label: "processed", value: lastMetric.processed ?? "-" },
      { label: "saved", value: lastMetric.saved ?? "-" },
      { label: "errors", value: lastMetric.errors ?? "-" },
      {
        label: "items/s",
        value: lastMetric.items_per_sec ? lastMetric.items_per_sec.toFixed(2) : "-",
      },
    ];
  }, [lastMetric]);

  const groupedSpecs = useMemo(() => {
    const visibleSpecs = argSpecs.filter((spec) => spec.name !== "overwrite");
    const byName = new Map(visibleSpecs.map((spec) => [spec.name, spec]));
    const used = new Set();
    const groups = settingGroups
      .map((group) => ({
        ...group,
        specs: group.names.map((name) => byName.get(name)).filter(Boolean),
      }))
      .filter((group) => group.specs.length > 0);
    groups.forEach((group) => group.specs.forEach((spec) => used.add(spec.name)));
    const other = visibleSpecs.filter((spec) => !used.has(spec.name));
    if (other.length > 0) {
      groups.push({ title: "Прочее", specs: other });
    }
    return groups;
  }, [argSpecs]);

  return (
    <div className="page">
      <h1 className="page-title">Prepare Latents</h1>
      <div className="two-col">
        <div className="page">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Control</h2>
              <StatusPill status={status.active ? "running" : "stopped"} />
            </div>
            {error && <div className="muted">{error}</div>}
            {status.active && !activeRun && (
              <div className="muted">Job запускается...</div>
            )}
            {activeRun && activeRunType !== "latent_cache" && (
              <div className="muted">Другой job уже выполняется: {activeRunType}</div>
            )}
            <div className="row">
              <button onClick={handleStart} disabled={status.active}>
                Start
              </button>
              <button className="warning" onClick={handleRebuild} disabled={status.active}>
                Rebuild cache
              </button>
              <button className="danger" onClick={handleStop} disabled={!status.active}>
                Stop
              </button>
            </div>
            {status.active && progressMax ? (
              <div className="progress">
                <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
              </div>
            ) : null}
            {command.length > 0 && <div className="muted">Latent cache task prepared.</div>}
            <div className="grid">
              {overview.map((item) => (
                <div key={item.label}>
                  <div className="muted">{item.label}</div>
                  <div>{item.value}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="settings-panel">
            <div className="settings-groups">
              {groupedSpecs.map((group) => (
                <section key={group.title} className="settings-section">
                  <h3>{group.title}</h3>
                  <div className="flat-grid">
                    {group.specs.map((spec) => (
                      <ArgField
                        key={spec.name}
                        spec={spec}
                        value={args[spec.name]}
                        onChange={handleChange}
                        variant="flat"
                      />
                    ))}
                  </div>
                </section>
              ))}
            </div>
          </div>
        </div>

        <div className="page">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Live Logs</h2>
              <div className="row">
                <button className="ghost" onClick={clearLogs}>
                  Clear
                </button>
              </div>
            </div>
            <LogViewer lines={logLines} />
            <div className="muted">Buffer: {logLines.length} / 10k lines</div>
          </div>
        </div>
      </div>
    </div>
  );
}
