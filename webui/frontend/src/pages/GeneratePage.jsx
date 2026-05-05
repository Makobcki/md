import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, wsUrl, absoluteFileUrl, absoluteDownloadUrl, API_ORIGIN } from "../api.js";
import useLogBuffer from "../hooks/useLogBuffer.js";
import ArgField from "../components/ArgField.jsx";
import { isMetricEvent, mergeMetricEvents } from "../utils/metrics.js";

const quickFields = ["task", "ckpt", "steps", "n", "seed"];
const promptFields = ["prompt", "neg_prompt"];
const hiddenPathFields = new Set(["init-image", "mask", "control-image"]);

const TASK_HELP = {
  txt2img: "Text → image generation from prompt only.",
  img2img: "Image → image generation. Upload an init image and adjust strength.",
  inpaint: "Inpaint generation. Upload an image and draw the white mask over regions to regenerate.",
  control: "Control generation with a control image.",
};


const absolutePreviewUrl = (value) => {
  if (!value) return "";
  if (/^blob:|^data:|^https?:/i.test(value)) return value;
  if (String(value).startsWith("/")) return `${API_ORIGIN}${value}`;
  return absoluteFileUrl(value) || "";
};

function TaskAssetCard({
  title,
  description,
  previewUrl,
  fileName,
  pathValue,
  onPick,
  onClear,
  disabled = false,
  children = null,
}) {
  const inputRef = useRef(null);

  return (
    <div className="task-asset-card">
      <div className="row task-asset-header">
        <div>
          <div className="card-title">{title}</div>
          {description ? <div className="muted">{description}</div> : null}
        </div>
        <div className="row">
          <button type="button" className="secondary" onClick={() => inputRef.current?.click()} disabled={disabled}>
            Upload
          </button>
          {(previewUrl || pathValue) && (
            <button type="button" className="secondary" onClick={onClear} disabled={disabled}>
              Clear
            </button>
          )}
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onPick(file);
          event.target.value = "";
        }}
        disabled={disabled}
      />
      {previewUrl ? (
        <div className="task-asset-preview">
          <img src={previewUrl} alt={title} />
        </div>
      ) : (
        <div className="task-asset-empty muted">No image selected</div>
      )}
      {fileName ? <div className="badge">{fileName}</div> : null}
      {pathValue ? <div className="task-asset-path muted">{pathValue}</div> : null}
      {children}
    </div>
  );
}

function MaskEditor({ imageUrl, value, onChange, disabled = false }) {
  const imgRef = useRef(null);
  const canvasRef = useRef(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef(null);
  const [brushSize, setBrushSize] = useState(32);
  const [eraseMode, setEraseMode] = useState(false);
  const [imageAspect, setImageAspect] = useState(1);

  const exportMask = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    onChange(canvas.toDataURL("image/png"));
  };

  const fillCanvas = (fillStyle) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = fillStyle;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
  };

  const setupCanvas = () => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas || !img.naturalWidth || !img.naturalHeight) return;
    const sameSize = canvas.width === img.naturalWidth && canvas.height === img.naturalHeight;
    if (!sameSize) {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      fillCanvas("black");
      onChange("");
    }
    setImageAspect(img.naturalWidth / img.naturalHeight);
  };

  useEffect(() => {
    if (!imageUrl) {
      onChange("");
    }
  }, [imageUrl, onChange]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return undefined;
    if (img.complete) setupCanvas();
    const onLoad = () => setupCanvas();
    img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  }, [imageUrl]);

  const pointerPoint = (event) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const drawSegment = (from, to) => {
    const canvas = canvasRef.current;
    if (!canvas || !from || !to) return;
    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = eraseMode ? "black" : "white";
    ctx.fillStyle = eraseMode ? "black" : "white";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = brushSize;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(to.x, to.y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  };

  const startDraw = (event) => {
    if (disabled || !imageUrl) return;
    const point = pointerPoint(event);
    if (!point) return;
    drawingRef.current = true;
    lastPointRef.current = point;
    drawSegment(point, point);
  };

  const moveDraw = (event) => {
    if (!drawingRef.current || disabled) return;
    const point = pointerPoint(event);
    if (!point) return;
    drawSegment(lastPointRef.current, point);
    lastPointRef.current = point;
  };

  const stopDraw = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastPointRef.current = null;
    exportMask();
  };

  const clearMask = () => {
    fillCanvas("black");
    exportMask();
  };

  if (!imageUrl) {
    return <div className="task-asset-empty muted">Upload an init image to start drawing an inpaint mask.</div>;
  }

  return (
    <div className="mask-editor-block">
      <div className="mask-editor-toolbar row">
        <label className="mask-editor-brush">
          Brush
          <input
            type="range"
            min="4"
            max="160"
            step="1"
            value={brushSize}
            onChange={(event) => setBrushSize(Number(event.target.value))}
            disabled={disabled}
          />
          <span>{brushSize}px</span>
        </label>
        <button type="button" className={`secondary mask-toggle ${eraseMode ? "active" : ""}`} onClick={() => setEraseMode((prev) => !prev)} disabled={disabled}>
          {eraseMode ? "Erase" : "Paint mask"}
        </button>
        <button type="button" className="secondary" onClick={clearMask} disabled={disabled}>
          Clear mask
        </button>
        {value ? <span className="badge">Mask ready</span> : <span className="badge">Mask empty</span>}
      </div>
      <div className="mask-editor-stage" style={{ aspectRatio: imageAspect || 1 }}>
        <img ref={imgRef} src={imageUrl} alt="Init preview" draggable="false" />
        <canvas
          ref={canvasRef}
          className="mask-editor-canvas"
          onPointerDown={startDraw}
          onPointerMove={moveDraw}
          onPointerUp={stopDraw}
          onPointerLeave={stopDraw}
          onPointerCancel={stopDraw}
        />
      </div>
      <div className="muted">Paint with white to regenerate those areas. Black regions stay preserved.</div>
    </div>
  );
}

async function dataUrlToFile(dataUrl, fileName) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return new File([blob], fileName, { type: blob.type || "image/png" });
}

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
  const [initFile, setInitFile] = useState(null);
  const [initPreview, setInitPreview] = useState("");
  const [controlFile, setControlFile] = useState(null);
  const [controlPreview, setControlPreview] = useState("");
  const [maskDataUrl, setMaskDataUrl] = useState("");
  const [isUploadingAssets, setIsUploadingAssets] = useState(false);
  const initBlobUrlRef = useRef("");
  const controlBlobUrlRef = useRef("");
  const wasGeneratingRef = useRef(false);
  const promptRef = useRef(null);
  const negativeRef = useRef(null);

  const logKey = runId ? `generate:logs:${runId}` : "generate:logs:idle";
  const { appendLines, replaceLines, clear: clearLogs } = useLogBuffer(logKey, {
    maxLines: 10000,
  });

  useEffect(() => {
    const load = async () => {
      const [argsData, ckptData] = await Promise.all([api.getSampleArgs(), api.listCheckpoints()]);
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
    if (!runId) {
      const samples = await api.listSamples("");
      const items = samples.items || [];
      if (items.length > 0 && !keepOutput) {
        setOutput([...items].reverse()[0]);
      }
      return;
    }
    const artifacts = await api.listArtifacts({ runId, source: "all" });
    const items = (artifacts.items || []).filter((item) =>
      ["webui_sample", "webui_latent"].includes(item.source)
    );
    if (items.length > 0 && !keepOutput) {
      setOutput([...items].sort((a, b) => (a.mtime || 0) - (b.mtime || 0)).reverse()[0]);
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
  }, [status.active, runId]);

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
        if (isMetricEvent(metric)) {
          setMetrics((prev) => mergeMetricEvents(prev, [metric], 500));
        }
      } catch (err) {
        console.warn(err);
      }
    };
    return () => ws.close();
  }, [runId]);

  useEffect(() => {
    return () => {
      if (initBlobUrlRef.current) URL.revokeObjectURL(initBlobUrlRef.current);
      if (controlBlobUrlRef.current) URL.revokeObjectURL(controlBlobUrlRef.current);
    };
  }, []);

  const handleChange = (name, value) => {
    setArgs((prev) => {
      if (name !== "task") return { ...prev, [name]: value };
      return {
        ...prev,
        task: value,
        "init-image": "",
        mask: "",
        "control-image": "",
      };
    });
    if (name === "task") {
      setInitFile(null);
      setControlFile(null);
      setInitPreview("");
      setControlPreview("");
      setMaskDataUrl("");
    }
  };

  const setLocalPreview = (kind, file) => {
    const nextUrl = file ? URL.createObjectURL(file) : "";
    if (kind === "init") {
      if (initBlobUrlRef.current) URL.revokeObjectURL(initBlobUrlRef.current);
      initBlobUrlRef.current = nextUrl;
      setInitFile(file || null);
      setInitPreview(nextUrl);
      setArgs((prev) => ({ ...prev, "init-image": file ? "" : prev["init-image"] || "" }));
      if (!file) setMaskDataUrl("");
      return;
    }
    if (controlBlobUrlRef.current) URL.revokeObjectURL(controlBlobUrlRef.current);
    controlBlobUrlRef.current = nextUrl;
    setControlFile(file || null);
    setControlPreview(nextUrl);
    setArgs((prev) => ({ ...prev, "control-image": file ? "" : prev["control-image"] || "" }));
  };

  const resizePromptInput = (node) => {
    if (!node) return;
    node.style.height = "0px";
    node.style.height = `${Math.min(node.scrollHeight, 96)}px`;
  };

  const currentTask = args.task || "txt2img";
  const needsInit = currentTask === "img2img" || currentTask === "inpaint";
  const needsMask = currentTask === "inpaint";
  const needsControl = currentTask === "control";

  const uploadPendingAssets = async (payload) => {
    const next = { ...payload };
    if (needsInit && initFile) {
      const uploaded = await api.uploadImage(initFile, currentTask === "inpaint" ? "inpaint-init" : "init");
      next["init-image"] = uploaded.path;
      setArgs((prev) => ({ ...prev, "init-image": uploaded.path }));
    }
    if (needsControl && controlFile) {
      const uploaded = await api.uploadImage(controlFile, "control");
      next["control-image"] = uploaded.path;
      setArgs((prev) => ({ ...prev, "control-image": uploaded.path }));
    }
    if (needsMask && maskDataUrl) {
      const maskFile = await dataUrlToFile(maskDataUrl, "inpaint-mask.png");
      const uploaded = await api.uploadImage(maskFile, "inpaint-mask");
      next.mask = uploaded.path;
      setArgs((prev) => ({ ...prev, mask: uploaded.path }));
    }
    return next;
  };

  const handleStart = async () => {
    setError("");
    setIsUploadingAssets(true);
    try {
      let payload = { ...args };
      if (!textConditioningAvailable) {
        payload.prompt = "";
        payload.neg_prompt = "";
      }
      payload = await uploadPendingAssets(payload);
      if (needsInit && !payload["init-image"]) {
        throw new Error(currentTask === "inpaint" ? "Для inpaint нужно загрузить init image." : "Для img2img нужно загрузить init image.");
      }
      if (needsMask && !payload.mask) {
        throw new Error("Для inpaint нужно нарисовать или указать mask.");
      }
      if (needsControl && !payload["control-image"]) {
        throw new Error("Для control generation нужно загрузить control image.");
      }
      const resp = await api.startSample(payload);
      setRunId(resp.run_id);
      clearLogs();
      setMetrics([]);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploadingAssets(false);
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
          spec.name !== "out" &&
          !hiddenPathFields.has(spec.name)
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

  const initPreviewUrl = absolutePreviewUrl(initPreview || args["init-image"]);
  const controlPreviewUrl = absolutePreviewUrl(controlPreview || args["control-image"]);

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

            <div className="task-hint muted">{TASK_HELP[currentTask] || ""}</div>

            {(needsInit || needsControl) && (
              <div className="settings-section task-assets-grid">
                {needsInit && (
                  <TaskAssetCard
                    title="Init image"
                    description="Source image used for img2img / inpaint."
                    previewUrl={initPreviewUrl}
                    fileName={initFile?.name || ""}
                    pathValue={args["init-image"] || ""}
                    onPick={(file) => setLocalPreview("init", file)}
                    onClear={() => {
                      if (initBlobUrlRef.current) URL.revokeObjectURL(initBlobUrlRef.current);
                      initBlobUrlRef.current = "";
                      setInitFile(null);
                      setInitPreview("");
                      setMaskDataUrl("");
                      setArgs((prev) => ({ ...prev, "init-image": "", mask: "" }));
                    }}
                    disabled={status.active || isUploadingAssets}
                  >
                    {needsMask ? (
                      <MaskEditor
                        imageUrl={initPreviewUrl}
                        value={maskDataUrl}
                        onChange={setMaskDataUrl}
                        disabled={status.active || isUploadingAssets}
                      />
                    ) : null}
                  </TaskAssetCard>
                )}
                {needsControl && (
                  <TaskAssetCard
                    title="Control image"
                    description="Conditioning image for control generation."
                    previewUrl={controlPreviewUrl}
                    fileName={controlFile?.name || ""}
                    pathValue={args["control-image"] || ""}
                    onPick={(file) => setLocalPreview("control", file)}
                    onClear={() => {
                      if (controlBlobUrlRef.current) URL.revokeObjectURL(controlBlobUrlRef.current);
                      controlBlobUrlRef.current = "";
                      setControlFile(null);
                      setControlPreview("");
                      setArgs((prev) => ({ ...prev, "control-image": "" }));
                    }}
                    disabled={status.active || isUploadingAssets}
                  />
                )}
              </div>
            )}

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
            ) : output && absoluteFileUrl(output) ? (
              <img src={absoluteFileUrl(output)} alt="sample" />
            ) : output && absoluteDownloadUrl(output) ? (
              <div className="task-asset-empty muted">
                Output is not previewable. <a href={absoluteDownloadUrl(output)}>Download artifact</a>
              </div>
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
                  disabled={!textConditioningAvailable || Boolean(blockingRunType) || isUploadingAssets}
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
                  disabled={!textConditioningAvailable || Boolean(blockingRunType) || isUploadingAssets}
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
              disabled={(status.active && !isGenerating) || isUploadingAssets}
              aria-label={isGenerating ? "Stop generation" : "Start generation"}
              title={isGenerating ? "Stop" : isUploadingAssets ? "Uploading assets" : "Start"}
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
