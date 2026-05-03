import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, wsUrl, API_ORIGIN } from "../api.js";
import useLogBuffer from "../hooks/useLogBuffer.js";
import ArgField from "../components/ArgField.jsx";

const quickFields = ["ckpt", "steps", "n", "seed"];
const promptFields = ["prompt", "neg_prompt"];

const toRunsUrl = (path) => {
  if (!path) return null;
  const marker = "webui_runs/";
  const idx = path.indexOf(marker);
  if (idx === -1) return null;
  return `${API_ORIGIN}/runs/${path.slice(idx + marker.length)}`;
};

export default function GeneratePage() {
  const [argSpecs, setArgSpecs] = useState([]);
  const [args, setArgs] = useState({});
  const [checkpoints, setCheckpoints] = useState([]);
  const [status, setStatus] = useState({ active: false });
  const [runId, setRunId] = useState(null);
  const [output, setOutput] = useState(null);
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState([]);
  const [textConditioningAvailable, setTextConditioningAvailable] = useState(true);
  const wasGeneratingRef = useRef(false);
  const promptRef = useRef(null);
  const negativeRef = useRef(null);

  const logKey = runId ? `generate:logs:${runId}` : "generate:logs:idle";
  const { appendLines, replaceLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const [argsData, ckptData] = await Promise.all([
        api.getSampleArgs(),
        api.listCheckpoints(),
      ]);
      setArgSpecs(argsData.items || []);
      setCheckpoints(ckptData.items || []);

      const initial = {};
      (argsData.items || []).forEach((spec) => {
        if (spec.name === "ckpt" && ckptData.items?.length > 0) {
          initial[spec.name] = ckptData.items[0];
        } else if (spec.default !== null && spec.default !== undefined) {
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
    const ckpt = args.ckpt;
    if (!ckpt) {
      setTextConditioningAvailable(true);
      return;
    }
    api
      .getCheckpointInfo(ckpt)
      .then((info) => setTextConditioningAvailable(info.use_text_conditioning !== false))
      .catch(() => setTextConditioningAvailable(true));
  }, [args.ckpt]);

  const refreshSamples = async ({ keepOutput = false } = {}) => {
    const samples = await api.listSamples();
    const items = samples.items || [];
    if (items.length > 0 && !keepOutput) {
      setOutput([...items].reverse()[0]);
    }
  };

  useEffect(() => {
    refreshSamples().catch((err) => console.warn("failed to refresh samples", err));
    const timer = setInterval(() => {
      refreshSamples({ keepOutput: status.active }).catch((err) =>
        console.warn("failed to refresh samples", err)
      );
    }, 5000);
    return () => clearInterval(timer);
  }, [status.active]);

  useEffect(() => {
    const poll = async () => {
      const stat = await api.getStatus();
      setStatus(stat);
      if (stat.active && stat.run?.run_type === "sample") {
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
        if (metric.type === "metric") {
          setMetrics((prev) => [...prev.slice(-200), metric]);
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

  const resizePromptInput = (node) => {
    if (!node) return;
    node.style.height = "0px";
    node.style.height = `${Math.min(node.scrollHeight, 96)}px`;
  };

  const handleStart = async () => {
    setError("");
    try {
      const payload = { ...args };
      if (!textConditioningAvailable) {
        payload.prompt = "";
        payload.neg_prompt = "";
      }
      const resp = await api.startSample(payload);
      setRunId(resp.run_id);
      clearLogs();
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

  const lastMetric = metrics[metrics.length - 1];
  const progressMax = lastMetric?.max_steps || 0;
  const progressValue = lastMetric?.step ?? 0;
  const activeRun = status.active ? status.run : null;
  const activeRunType = activeRun?.run_type;
  const isGenerating = activeRunType === "sample";
  const blockingRunType = status.active && activeRunType !== "sample" ? activeRunType : "";
  const inputOverlay = blockingRunType
    ? `Уже выполняется: ${blockingRunType}`
    : !textConditioningAvailable
      ? "Модель не поддерживает генерацию на основе текста"
      : "";

  const negativeSpec = useMemo(
    () => argSpecs.find((spec) => spec.name === "neg_prompt"),
    [argSpecs]
  );
  const quickSpecs = useMemo(
    () => argSpecs.filter((spec) => quickFields.includes(spec.name)),
    [argSpecs]
  );
  const advancedSpecs = useMemo(
    () =>
      argSpecs.filter(
        (spec) =>
          !quickFields.includes(spec.name) &&
          !promptFields.includes(spec.name) &&
          spec.name !== "out"
      ),
    [argSpecs]
  );

  useEffect(() => {
    if (isGenerating) {
      wasGeneratingRef.current = true;
      return;
    }
    if (!wasGeneratingRef.current) return;
    wasGeneratingRef.current = false;
    refreshSamples().catch((err) => console.warn("failed to refresh samples", err));
    const timer = setTimeout(() => {
      setArgs((prev) => ({ ...prev, prompt: "", neg_prompt: "" }));
    }, 900);
    return () => clearTimeout(timer);
  }, [isGenerating]);

  const handlePromptKeyDown = (event) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    if (!status.active) {
      handleStart();
    }
  };

  useLayoutEffect(() => {
    resizePromptInput(promptRef.current);
    resizePromptInput(negativeRef.current);
  }, [args.prompt, args[negativeSpec?.name || "neg_prompt"]]);

  return (
    <div className="page generate-page">
      <h1 className="page-title">Generate</h1>
      <div className="generate-workspace">
        <div className={`generate-stage ${isGenerating ? "is-generating" : ""} ${output ? "has-output" : ""}`}>
          <div className="settings-panel generate-panel settings-square">
            {error && <div className="muted">{error}</div>}
            {status.active && !activeRun && <div className="muted">Job запускается...</div>}

            <div className="quick-settings">
              {quickSpecs.map((spec) => (
                <ArgField
                  key={spec.name}
                  spec={spec}
                  value={args[spec.name]}
                  onChange={handleChange}
                  checkpoints={checkpoints}
                  variant="flat"
                />
              ))}
            </div>

            <div className="settings-section">
              <div className="flat-grid">
                {advancedSpecs.map((spec) => (
                  <ArgField
                    key={spec.name}
                    spec={spec}
                    value={args[spec.name]}
                    onChange={handleChange}
                    checkpoints={checkpoints}
                    variant="flat"
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="generation-preview-square">
            {isGenerating ? (
              <div className="preview-loader" />
            ) : output && toRunsUrl(output) ? (
              <img src={toRunsUrl(output)} alt="sample" />
            ) : (
              <div className="preview-loader" />
            )}
          </div>
        </div>

        <section className="chat-compose-panel">
          {(isGenerating || progressMax > 0) && (
            <div
              className={`compose-progress ${isGenerating ? "active" : ""}`}
              aria-label="Generation progress"
            >
              <span style={{ width: `${progressMax ? Math.min(100, (progressValue / progressMax) * 100) : 0}%` }} />
            </div>
          )}
          <div className={`chat-input-row ${blockingRunType ? "blocked" : ""}`}>
            <div className={`chat-fields ${inputOverlay ? "unavailable" : ""}`}>
              <div className="chat-field prompt-field-main">
                <textarea
                  ref={promptRef}
                  value={args.prompt ?? ""}
                  onChange={(event) => {
                    handleChange("prompt", event.target.value);
                    resizePromptInput(event.target);
                  }}
                  onKeyDown={handlePromptKeyDown}
                  disabled={!textConditioningAvailable || Boolean(blockingRunType)}
                  placeholder="Prompt"
                  rows={1}
                />
              </div>
              <div className="chat-field prompt-field-negative">
                <textarea
                  ref={negativeRef}
                  value={args[negativeSpec?.name || "neg_prompt"] ?? ""}
                  onChange={(event) => {
                    handleChange(negativeSpec?.name || "neg_prompt", event.target.value);
                    resizePromptInput(event.target);
                  }}
                  onKeyDown={handlePromptKeyDown}
                  disabled={!textConditioningAvailable || Boolean(blockingRunType)}
                  placeholder="Negative prompt"
                  rows={1}
                />
              </div>
              {inputOverlay && !blockingRunType && <div className="chat-fields-overlay">{inputOverlay}</div>}
            </div>
            {blockingRunType && <div className="chat-row-overlay">{inputOverlay}</div>}
            <button
              type="button"
              className={`generate-action ${isGenerating ? "stop" : ""}`}
              onClick={isGenerating ? handleStop : handleStart}
              disabled={status.active && !isGenerating}
              aria-label={isGenerating ? "Stop generation" : "Start generation"}
              title={isGenerating ? "Stop" : "Start"}
            >
              {isGenerating ? (
                <svg viewBox="0 -960 960 960" aria-hidden="true">
                  <path d="m336-280 144-144 144 144 56-56-144-144 144-144-56-56-144 144-144-144-56 56 144 144-144 144 56 56ZM480-80q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z" />
                </svg>
              ) : (
                <svg viewBox="0 -960 960 960" aria-hidden="true">
                  <path d="M120-160v-640l760 320-760 320Zm80-120 474-200-474-200v140l240 60-240 60v140Zm0 0v-400 400Z" />
                </svg>
              )}
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
