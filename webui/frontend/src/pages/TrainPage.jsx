import React, { useEffect, useMemo, useRef, useState } from "react";
import { api, wsUrl } from "../api.js";
import LogViewer from "../components/LogViewer.jsx";
import LineChart from "../components/LineChart.jsx";
import MetricTile from "../components/MetricTile.jsx";
import PageHeader from "../components/PageHeader.jsx";
import YamlEditor from "../components/YamlEditor.jsx";
import useLogBuffer from "../hooks/useLogBuffer.js";
import useRunLogStream from "../hooks/useRunLogStream.js";
import StatusPill from "../components/StatusPill.jsx";
import {
  formatStep,
  isMetricEvent,
  latestMetricWithLoss,
  mergeMetricEvents,
  metricChartData,
  metricElapsed,
  metricEta,
  metricLoss,
  metricSecondsPerStep,
  metricStep,
  metricThroughput,
} from "../utils/metrics.js";
import { extractConfigUses } from "../utils/uiModel.js";

const presetKindOrder = ["model", "training", "data", "text", "sampler", "webui"];

const presetKindLabels = {
  model: "Model",
  training: "Training",
  data: "Data",
  text: "Text",
  sampler: "Sampler",
  webui: "WebUI",
};

function insertPresetUse(content, kind, name) {
  const section = kind === "sampler" ? "sampling" : kind;
  const lines = String(content || "").split("\n");
  const useLine = `    use "${name}"`;
  let depth = 0;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const startsSection = depth === 1 && new RegExp(`^\\s*${section}\\s*\\{`).test(line);
    if (startsSection) {
      lines.splice(index + 1, 0, useLine);
      return lines.join("\n");
    }
    depth += (line.match(/\{/g) || []).length;
    depth -= (line.match(/\}/g) || []).length;
  }

  const insertAt = Math.max(1, lines.findIndex((line) => /^\s*}\s*$/.test(line)));
  const block = [`  ${section} {`, `    use "${name}"`, "  }", ""];
  if (insertAt > 0) {
    lines.splice(insertAt, 0, ...block);
    return lines.join("\n");
  }
  return `${content.trimEnd()}\n\n${block.join("\n")}`;
}

function PresetLibrary({
  presets,
  selectedKind,
  selectedName,
  onSelectKind,
  onSelectPreset,
  activeUses,
  onInsert,
}) {
  const groups = presets?.groups || {};
  const kinds = presetKindOrder.filter((kind) => (groups[kind] || []).length > 0);
  const items = groups[selectedKind] || [];
  const selected = items.find((item) => item.name === selectedName) || items[0] || null;
  const isActive = (item) => {
    const activeKind = item.kind === "sampler" ? "sampling" : item.kind;
    return (activeUses[activeKind] || activeUses[item.kind] || []).includes(item.name);
  };

  return (
    <div className="preset-library">
      <div className="preset-kind-tabs" role="tablist" aria-label="Preset groups">
        {kinds.map((kind) => (
          <button
            key={kind}
            type="button"
            className={kind === selectedKind ? "active" : ""}
            onClick={() => onSelectKind(kind)}
          >
            {presetKindLabels[kind] || kind}
          </button>
        ))}
      </div>

      <div className="preset-browser">
        <div className="preset-list">
          {items.map((item) => {
            const active = isActive(item);
            return (
              <button
                key={`${item.kind}:${item.name}`}
                type="button"
                className={`preset-list-item ${selected?.name === item.name ? "selected" : ""}`}
                onClick={() => onSelectPreset(item.name)}
              >
                <span>
                  <strong>{item.name}</strong>
                  <small>{item.summary || item.relative_path}</small>
                </span>
                {active ? <em>active</em> : null}
              </button>
            );
          })}
        </div>

        <div className="preset-detail">
          {selected ? (
            <>
              <div className="preset-detail-head">
                <div>
                  <h3>{selected.name}</h3>
                  <p>{selected.summary || selected.relative_path}</p>
                </div>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => onInsert(selected)}
                  disabled={isActive(selected)}
                >
                  Insert use
                </button>
              </div>
              <div className="preset-meta-row">
                <span className="badge">{selected.kind}</span>
                <span className="badge">v{selected.version ?? "?"}</span>
                <span className="badge">{selected.relative_path}</span>
              </div>
              <pre className="preset-preview">{selected.content}</pre>
            </>
          ) : (
            <div className="empty-state">No presets in this group.</div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function TrainPage() {
  const [config, setConfig] = useState("");
  const [lastSaved, setLastSaved] = useState("");
  const [status, setStatus] = useState({ active: false });
  const [metrics, setMetrics] = useState([]);
  const [runId, setRunId] = useState(null);
  const [command, setCommand] = useState([]);
  const [error, setError] = useState("");
  const [checkpoints, setCheckpoints] = useState([]);
  const [resumeCkpt, setResumeCkpt] = useState("");
  const [saving, setSaving] = useState(false);
  const [presets, setPresets] = useState({ groups: {}, active: {} });
  const [selectedPresetKind, setSelectedPresetKind] = useState("model");
  const [selectedPresetName, setSelectedPresetName] = useState("");

  const saveTimeoutRef = useRef(null);
  const metricOffsetRef = useRef(0);
  const logKey = runId ? `train:logs:${runId}` : "train:logs:idle";
  const { lines: logLines, appendLines, replaceLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const [cfg, ckpts, presetData] = await Promise.all([
        api.getConfig(),
        api.listCheckpoints(),
        api.getConfigPresets(),
      ]);
      setConfig(cfg.content);
      setLastSaved(cfg.content);
      setCheckpoints(ckpts.items || []);
      setPresets(presetData);
      const firstKind = presetKindOrder.find((kind) => (presetData.groups?.[kind] || []).length > 0);
      if (firstKind) {
        setSelectedPresetKind(firstKind);
        setSelectedPresetName(presetData.groups[firstKind][0]?.name || "");
      }
    };
    load();
    const poll = async () => {
      const stat = await api.getStatus();
      setStatus(stat);
      if (stat.active && stat.run?.run_type === "train") {
        setRunId(stat.run.run_id);
      }
    };
    poll();
    const timer = setInterval(poll, 3000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (runId) replaceLines([]);
  }, [runId, replaceLines]);

  useEffect(() => {
    if (!runId) return;
    let cancelled = false;
    metricOffsetRef.current = 0;
    setMetrics([]);
    const loadMetrics = async () => {
      try {
        const data = await api.getRunMetrics(runId, {
          offset: metricOffsetRef.current,
          limit: 2000,
        });
        if (cancelled) return;
        const items = data.items || [];
        if (Number.isFinite(data.next_offset)) {
          metricOffsetRef.current = data.next_offset;
        }
        if (items.length > 0) {
          setMetrics((prev) => mergeMetricEvents(prev, items, 1000));
        }
      } catch (err) {
        console.warn("failed to load metrics", err);
      }
    };
    loadMetrics();
    const timer = setInterval(loadMetrics, 500);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [runId]);

  useRunLogStream(runId, {
    backlog: 2000,
    onLog: (payload) => appendLines(`[${payload.stream}] ${payload.line}`),
    onError: (err) => console.warn("failed to stream logs", err),
  });

  useEffect(() => {
    if (!runId) return;
    const ws = new WebSocket(wsUrl(`/ws/metrics/${runId}`));
    ws.onmessage = (event) => {
      try {
        const metric = JSON.parse(event.data);
        if (isMetricEvent(metric)) {
          setMetrics((prev) => mergeMetricEvents(prev, [metric], 1000));
        }
      } catch (err) {
        console.warn(err);
      }
    };
    return () => ws.close();
  }, [runId]);

  const isDirty = useMemo(() => config !== lastSaved, [config, lastSaved]);

  useEffect(() => {
    if (!isDirty) return;
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    saveTimeoutRef.current = setTimeout(() => {
      handleSave(true);
    }, 1200);
    return () => clearTimeout(saveTimeoutRef.current);
  }, [config]);

  const handleSave = async (silent = false) => {
    if (saving) return false;
    setSaving(true);
    if (!silent) {
      setError("");
    }
    try {
      await api.updateConfig(config);
      setLastSaved(config);
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setSaving(false);
    }
  };

  const handleStart = async () => {
    setError("");
    if (isDirty) {
      const saved = await handleSave(true);
      if (!saved) return;
    }
    try {
      const payload = resumeCkpt ? { resume: resumeCkpt } : {};
      const resp = await api.startTrain(payload);
      setRunId(resp.run_id);
      setCommand(resp.command);
      metricOffsetRef.current = 0;
      clearLogs();
      setMetrics([]);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleStop = async () => {
    setError("");
    try {
      await api.stopTrain();
    } catch (err) {
      setError(err.message);
    }
  };

  const lastMetric = latestMetricWithLoss(metrics);
  const chartData = metricChartData(metrics);
  const progressMax = lastMetric?.max_steps || 0;
  const progressValue = metricStep(lastMetric) ?? 0;
  const activeRun = status.active ? status.run : null;
  const activeRunType = activeRun?.run_type;
  const activeUses = useMemo(() => extractConfigUses(config), [config]);
  const activeUseEntries = useMemo(
    () =>
      Object.entries(activeUses).flatMap(([kind, names]) =>
        names.map((name) => ({ kind, name }))
      ),
    [activeUses]
  );

  const handleInsertPreset = (preset) => {
    setConfig((prev) => insertPresetUse(prev, preset.kind, preset.name));
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow="Training"
        title="Train"
        description="Edit the active config, start or resume training, and monitor loss, throughput and logs."
        meta={<StatusPill status={status.active ? "running" : "stopped"} />}
      />
      <div className="two-col train-page-grid">
        <div className="page">
          <div className="card train-config-card">
            <div className="card-header">
              <div>
                <h2 className="card-title">Config Editor</h2>
                <div className="muted">Primary config plus section-scoped preset library.</div>
              </div>
              <span className={isDirty ? "badge dirty" : "badge"}>{isDirty ? "Unsaved" : "Saved"}</span>
            </div>
            <div className="active-presets-row">
              {activeUseEntries.length > 0 ? (
                activeUseEntries.map((item) => (
                  <span key={`${item.kind}:${item.name}`} className="badge active-preset">
                    {item.kind}: {item.name}
                  </span>
                ))
              ) : (
                <span className="muted">No direct preset use lines detected.</span>
              )}
            </div>
            <div className="train-editor-layout">
              <div className="train-config-editor">
                <YamlEditor value={config} onChange={setConfig} onSave={() => handleSave(false)} />
              </div>
              <PresetLibrary
                presets={presets}
                selectedKind={selectedPresetKind}
                selectedName={selectedPresetName}
                onSelectKind={(kind) => {
                  setSelectedPresetKind(kind);
                  setSelectedPresetName(presets.groups?.[kind]?.[0]?.name || "");
                }}
                onSelectPreset={setSelectedPresetName}
                activeUses={activeUses}
                onInsert={handleInsertPreset}
              />
            </div>
            <div className="row" style={{ marginTop: "12px" }}>
              <button onClick={() => handleSave(false)} disabled={saving}>
                Save
              </button>
              {saving && <span className="muted">Saving...</span>}
            </div>
          </div>
        </div>

        <div className="page">
          <div className="card training-control-card">
            <div className="card-header">
              <h2 className="card-title">Training Control</h2>
              <StatusPill status={status.active ? "running" : "stopped"} />
            </div>
            {error && <div className="muted">{error}</div>}
            {status.active && !activeRun && (
              <div className="muted">Job запускается...</div>
            )}
            {activeRun && activeRunType !== "train" && (
              <div className="muted">Другой job уже выполняется: {activeRunType}</div>
            )}
            <div className="row">
              <button onClick={handleStart} disabled={status.active}>
                Start
              </button>
              <button className="danger" onClick={handleStop} disabled={!status.active}>
                Stop
              </button>
            </div>
            {status.active && progressMax ? (
              <>
                <div className="progress">
                  <span style={{ width: `${Math.min(100, (progressValue / progressMax) * 100)}%` }} />
                </div>
                <div className="muted">step {formatStep(lastMetric)}</div>
              </>
            ) : null}
            <div className="row">
              <label>Resume checkpoint</label>
              <select
                value={resumeCkpt}
                onChange={(event) => setResumeCkpt(event.target.value)}
                disabled={status.active}
              >
                <option value="">(none)</option>
                {checkpoints.map((ckpt) => (
                  <option key={ckpt} value={ckpt}>
                    {ckpt}
                  </option>
                ))}
              </select>
            </div>
            {command.length > 0 && <div className="muted">Training command prepared.</div>}
            {lastMetric && (
              <div className="compact-metrics train-metrics-grid">
                <MetricTile label="step" value={formatStep(lastMetric)} />
                <MetricTile label="elapsed" value={metricElapsed(lastMetric)} />
                <MetricTile label="ETA" value={metricEta(lastMetric)} />
                <MetricTile
                  label="s/step"
                  value={Number.isFinite(metricSecondsPerStep(lastMetric)) ? metricSecondsPerStep(lastMetric).toFixed(3) : "—"}
                />
                <MetricTile
                  label="samples/s"
                  value={Number.isFinite(metricThroughput(lastMetric)) ? metricThroughput(lastMetric).toFixed(2) : "—"}
                />
                <MetricTile
                  label="loss"
                  value={Number.isFinite(metricLoss(lastMetric)) ? metricLoss(lastMetric).toFixed(4) : "—"}
                />
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Loss vs Step</h2>
              <span className="muted">{lastMetric ? `step ${formatStep(lastMetric)}` : "avg log_every"}</span>
            </div>
            <LineChart data={chartData} />
          </div>

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

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Samples</h2>
            </div>
            <div className="row">
              <a href="/train/samples" className="muted">
                Open samples gallery
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
