import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, wsUrl, absoluteFileUrl, absoluteDownloadUrl, API_ORIGIN } from "../api.js";
import useLogBuffer from "../hooks/useLogBuffer.js";
import useRunLogStream from "../hooks/useRunLogStream.js";
import PageHeader from "../components/PageHeader.jsx";
import { isMetricEvent, mergeMetricEvents } from "../utils/metrics.js";
import { buildArgsWithStoredSettings, SAMPLE_SETTINGS_KEY } from "../utils/settingsModel.js";

const absolutePreviewUrl = (value) => {
  if (!value) return "";
  if (/^blob:|^data:|^https?:/i.test(value)) return value;
  if (String(value).startsWith("/")) return `${API_ORIGIN}${value}`;
  return absoluteFileUrl(value) || "";
};

function ImageAttachPortal({ open, onClose, onPick, disabled = false }) {
  const inputRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    const handleKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose, open]);

  if (!open || typeof document === "undefined") return null;

  const pickFile = (file) => {
    if (!file || disabled) return;
    onPick(file);
    onClose();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    pickFile(event.dataTransfer.files?.[0]);
  };

  return createPortal(
    <div className="attach-portal" role="presentation">
      <button
        type="button"
        className="attach-portal-backdrop"
        onClick={onClose}
        aria-label="Close image attachment dialog"
      />
      <div
        className="attach-portal-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Attach source image"
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
      >
        <div>
          <h2>Attach image</h2>
          <p className="muted">Drop an image here or choose a local file.</p>
        </div>
        <button
          type="button"
          className="secondary"
          onClick={() => inputRef.current?.click()}
          disabled={disabled}
        >
          Choose file
        </button>
        <button type="button" className="ghost" onClick={onClose}>
          Cancel
        </button>
      </div>
      <input
        ref={inputRef}
        name="source-image"
        type="file"
        accept="image/png,image/jpeg,image/webp,image/bmp"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          pickFile(file);
          event.target.value = "";
        }}
        disabled={disabled}
      />
    </div>,
    document.body
  );
}

function AttachedImageEditor({
  imageUrl,
  fileName,
  maskValue,
  onMaskChange,
  onClearMask,
  onRemove,
  onOpenPicker,
  disabled = false,
}) {
  const imgRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef(null);
  const [brushSize, setBrushSize] = useState(32);
  const [maskMode, setMaskMode] = useState("reveal");
  const hasDrawnMaskRef = useRef(Boolean(maskValue));

  const resetCanvases = () => {
    const overlayCanvas = overlayCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const img = imgRef.current;
    if (!overlayCanvas || !maskCanvas || !img?.naturalWidth || !img?.naturalHeight) return;

    [overlayCanvas, maskCanvas].forEach((canvas) => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
    });

    const overlayCtx = overlayCanvas.getContext("2d");
    overlayCtx.save();
    overlayCtx.globalCompositeOperation = "source-over";
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.fillStyle = "rgba(236, 236, 241, 0.42)";
    overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    overlayCtx.restore();

    const maskCtx = maskCanvas.getContext("2d");
    maskCtx.save();
    maskCtx.globalCompositeOperation = "source-over";
    maskCtx.fillStyle = "black";
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.restore();
  };

  const clearMask = () => {
    hasDrawnMaskRef.current = false;
    resetCanvases();
    onClearMask();
  };

  const exportMask = () => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !hasDrawnMaskRef.current) {
      onMaskChange("");
      return;
    }
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    let hasWhitePixels = false;
    for (let index = 0; index < pixels.length; index += 4) {
      if (pixels[index] > 8 || pixels[index + 1] > 8 || pixels[index + 2] > 8) {
        hasWhitePixels = true;
        break;
      }
    }
    if (!hasWhitePixels) {
      hasDrawnMaskRef.current = false;
      onMaskChange("");
      return;
    }
    onMaskChange(canvas.toDataURL("image/png"));
  };

  useEffect(() => {
    if (!imageUrl) {
      onMaskChange("");
      return undefined;
    }
    const img = imgRef.current;
    if (!img) return undefined;
    if (img.complete) {
      hasDrawnMaskRef.current = false;
      resetCanvases();
      onMaskChange("");
    }
    const onLoad = () => {
      hasDrawnMaskRef.current = false;
      resetCanvases();
      onMaskChange("");
    };
    img.addEventListener("load", onLoad);
    return () => img.removeEventListener("load", onLoad);
  }, [imageUrl, onMaskChange]);

  const pointerPoint = (event) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const drawSegment = (from, to) => {
    const overlayCanvas = overlayCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!overlayCanvas || !maskCanvas || !from || !to) return;

    const drawLine = (ctx, mode, strokeStyle, fillStyle) => {
      ctx.save();
      ctx.globalCompositeOperation = mode;
      ctx.strokeStyle = strokeStyle;
      ctx.fillStyle = fillStyle;
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

    if (maskMode === "restore") {
      drawLine(overlayCanvas.getContext("2d"), "source-over", "rgba(236, 236, 241, 0.42)", "rgba(236, 236, 241, 0.42)");
      drawLine(maskCanvas.getContext("2d"), "source-over", "black", "black");
    } else {
      drawLine(overlayCanvas.getContext("2d"), "destination-out", "rgba(0,0,0,1)", "rgba(0,0,0,1)");
      drawLine(maskCanvas.getContext("2d"), "source-over", "white", "white");
      hasDrawnMaskRef.current = true;
    }
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

  return (
    <div className={`attached-image-card ${maskValue ? "has-mask" : ""}`}>
      <div className="attached-image-toolbar">
        <button
          type="button"
          className={`mask-mode-button ${maskMode === "reveal" ? "active" : ""}`}
          onClick={() => setMaskMode("reveal")}
          disabled={disabled}
        >
          Reveal
        </button>
        <button
          type="button"
          className={`mask-mode-button ${maskMode === "restore" ? "active" : ""}`}
          onClick={() => setMaskMode("restore")}
          disabled={disabled}
        >
          Restore
        </button>
        <button type="button" className="mask-brush-button" disabled={disabled}>
          Brush {brushSize}px
        </button>
        <input
          className="mask-brush-range"
          type="range"
          min="8"
          max="140"
          step="1"
          value={brushSize}
          onChange={(event) => setBrushSize(Number(event.target.value))}
          disabled={disabled}
          aria-label="Mask brush size"
        />
        <button type="button" className="ghost" onClick={clearMask} disabled={disabled || !maskValue}>
          Clear
        </button>
        <button type="button" className="ghost" onClick={onOpenPicker} disabled={disabled}>
          Replace
        </button>
        <button type="button" className="ghost" onClick={onRemove} disabled={disabled}>
          Remove
        </button>
      </div>
      <div className="attached-image-stage">
        <img ref={imgRef} src={imageUrl} alt="Attached source" draggable="false" />
        <canvas
          ref={overlayCanvasRef}
          className="attached-mask-overlay"
          onPointerDown={startDraw}
          onPointerMove={moveDraw}
          onPointerUp={stopDraw}
          onPointerLeave={stopDraw}
          onPointerCancel={stopDraw}
        />
        <canvas ref={maskCanvasRef} hidden />
      </div>
      <div className="attached-image-meta">
        <span className="badge">{fileName || "attached image"}</span>
        <span className="muted">Hover and draw to erase the overlay into an inpaint mask.</span>
      </div>
    </div>
  );
}

async function dataUrlToFile(dataUrl, fileName) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  return new File([blob], fileName, { type: blob.type || "image/png" });
}

export default function GeneratePage() {
  const [args, setArgs] = useState({});
  const [status, setStatus] = useState({ active: false });
  const [runId, setRunId] = useState(null);
  const [output, setOutput] = useState(null);
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState([]);
  const [textConditioningAvailable, setTextConditioningAvailable] = useState(true);
  const [initFile, setInitFile] = useState(null);
  const [initPreview, setInitPreview] = useState("");
  const [maskDataUrl, setMaskDataUrl] = useState("");
  const [attachPortalOpen, setAttachPortalOpen] = useState(false);
  const [isUploadingAssets, setIsUploadingAssets] = useState(false);
  const initBlobUrlRef = useRef("");
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

      const initial = buildArgsWithStoredSettings(
        argsData.items || [],
        SAMPLE_SETTINGS_KEY,
        { checkpoints: ckptData.items || [] }
      );
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
    if (runId) replaceLines([]);
  }, [runId, replaceLines]);

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
    };
  }, []);

  const handleChange = (name, value) => {
    setArgs((prev) => {
      if (name === "family") {
        const sampler = value === "var" ? "var_autoregressive" : "flow_heun";
        return {
          ...prev,
          family: value,
          task: "txt2img",
          sampler,
          "init-image": "",
          mask: "",
          "control-image": "",
        };
      }
      if (name !== "task") return { ...prev, [name]: value };
      return {
        ...prev,
        task: value,
        "init-image": "",
        mask: "",
        "control-image": "",
      };
    });
    if (name === "task" || name === "family") {
      setInitFile(null);
      setInitPreview("");
      setMaskDataUrl("");
    }
  };

  const setLocalPreview = (file) => {
    const nextUrl = file ? URL.createObjectURL(file) : "";
    if (initBlobUrlRef.current) URL.revokeObjectURL(initBlobUrlRef.current);
    initBlobUrlRef.current = nextUrl;
    setInitFile(file || null);
    setInitPreview(nextUrl);
    setMaskDataUrl("");
    setArgs((prev) => ({
      ...prev,
      task: file ? "img2img" : "txt2img",
      "init-image": file ? "" : "",
      mask: "",
      "control-image": "",
    }));
  };

  const clearAttachedImage = () => {
    setLocalPreview(null);
  };

  const resizePromptInput = (node) => {
    if (!node) return;
    node.style.height = "0px";
    node.style.height = `${Math.min(node.scrollHeight, 96)}px`;
  };

  const promptText = String(args.prompt || "").trim();
  const hasAttachedImage = Boolean(initPreview || args["init-image"]);
  const uiMode = hasAttachedImage
    ? maskDataUrl
      ? "inpaint"
      : "image-image"
    : promptText
      ? "text-image"
      : "none-image";
  const currentTask = uiMode === "inpaint" ? "inpaint" : hasAttachedImage ? "img2img" : "txt2img";
  const needsInit = currentTask === "img2img" || currentTask === "inpaint";
  const needsMask = currentTask === "inpaint";

  useEffect(() => {
    setArgs((prev) => (prev.task === currentTask ? prev : { ...prev, task: currentTask }));
  }, [currentTask]);

  const uploadPendingAssets = async (payload) => {
    const next = { ...payload, task: currentTask };
    if (needsInit && initFile) {
      const uploaded = await api.uploadImage(initFile, currentTask === "inpaint" ? "inpaint-init" : "init");
      next["init-image"] = uploaded.path;
      setArgs((prev) => ({ ...prev, "init-image": uploaded.path }));
    }
    if (needsMask && maskDataUrl) {
      const maskFile = await dataUrlToFile(maskDataUrl, "inpaint-mask.png");
      const uploaded = await api.uploadImage(maskFile, "inpaint-mask");
      next.mask = uploaded.path;
      setArgs((prev) => ({ ...prev, mask: uploaded.path }));
    }
    if (!needsInit) {
      delete next["init-image"];
      delete next.mask;
      delete next["control-image"];
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
        throw new Error(currentTask === "inpaint" ? "Для inpaint нужно прикрепить image." : "Для image-image нужно прикрепить image.");
      }
      if (needsMask && !payload.mask) {
        throw new Error("Для inpaint нужно нарисовать mask.");
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
  const previewUrl = output ? absoluteFileUrl(output) : "";
  const downloadUrl = output ? absoluteDownloadUrl(output) : "";


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

  const handleMaskChange = useCallback((value) => {
    setMaskDataUrl(value);
    setArgs((prev) => ({ ...prev, mask: "" }));
  }, []);

  useLayoutEffect(() => {
    resizePromptInput(promptRef.current);
    resizePromptInput(negativeRef.current);
  }, [args.prompt, args.neg_prompt]);

  const initPreviewUrl = absolutePreviewUrl(initPreview || args["init-image"]);

  return (
    <div className="page fit-page generate-page">
      <ImageAttachPortal
        open={attachPortalOpen}
        onClose={() => setAttachPortalOpen(false)}
        onPick={setLocalPreview}
        disabled={status.active || isUploadingAssets}
      />
      <PageHeader
        eyebrow="Sampling"
        title="Generate"
        description="Prompt, image and mask."
      />
      <div className="generate-workspace">
        {error && <div className="alert error">{error}</div>}
        {status.active && !activeRun && <div className="alert">Job запускается...</div>}

        <div
          className={`generate-stage ${isGenerating ? "is-generating" : ""} ${output ? "has-output" : ""} ${hasAttachedImage ? "has-attachment" : ""}`}
        >
          {hasAttachedImage ? (
            <AttachedImageEditor
              imageUrl={initPreviewUrl}
              fileName={initFile?.name || String(args["init-image"] || "").split("/").pop()}
              maskValue={maskDataUrl}
              onMaskChange={handleMaskChange}
              onClearMask={() => {
                setMaskDataUrl("");
                setArgs((prev) => ({ ...prev, mask: "", task: "img2img" }));
              }}
              onRemove={clearAttachedImage}
              onOpenPicker={() => setAttachPortalOpen(true)}
              disabled={status.active || isUploadingAssets}
            />
          ) : null}
          <div className="generation-preview-square">
            {isGenerating ? (
              <div className="preview-loader" />
            ) : previewUrl ? (
              <img src={previewUrl} alt="sample" />
            ) : downloadUrl ? (
              <div className="empty-state preview-empty-state">
                <strong>Preview unavailable</strong>
                <span>
                  Output is not previewable. <a href={downloadUrl}>Download artifact</a>
                </span>
              </div>
            ) : (
              <div className="empty-state preview-empty-state">
                <strong>No preview yet</strong>
                <span>Start a generation to see the latest sample here.</span>
              </div>
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
            <button
              type="button"
              className={`attach-action ${hasAttachedImage ? "active" : ""}`}
              onClick={() => setAttachPortalOpen(true)}
              disabled={status.active || isUploadingAssets}
              aria-label="Attach image"
              title={hasAttachedImage ? "Replace image" : "Attach image"}
            >
              <svg viewBox="0 -960 960 960" aria-hidden="true">
                <path d="M440-440H200v-80h240v-240h80v240h240v80H520v240h-80v-240Z" />
              </svg>
            </button>
            <div className={`chat-fields ${inputOverlay ? "unavailable" : ""}`}>
              <div className="chat-field prompt-field-main">
                <textarea
                  ref={promptRef}
                  name="prompt"
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
                  name="negative-prompt"
                  value={args.neg_prompt ?? ""}
                  onChange={(event) => {
                    handleChange("neg_prompt", event.target.value);
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
              disabled={(status.active && !isGenerating) || isUploadingAssets || uiMode === "none-image"}
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
