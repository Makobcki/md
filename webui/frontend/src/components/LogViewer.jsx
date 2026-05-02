import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

const classifyLine = (line) => {
  const value = String(line).toLowerCase();
  if (value.startsWith("[stderr]")) {
    return "log-line stderr";
  }
  if (value.includes("error") || value.includes("exception") || value.includes("failed") || value.includes("oom")) {
    return "log-line error";
  }
  if (value.includes("warn") || value.includes("warning")) {
    return "log-line warning";
  }
  if (value.includes("loss") || value.includes("step=") || value.includes("metric")) {
    return "log-line metric";
  }
  if (value.includes("saved") || value.includes("done") || value.includes("completed")) {
    return "log-line success";
  }
  return "log-line";
};

const highlightParts = (line) => {
  const pattern = /(loss|step|lr|grad_norm|img_per_sec|peak_mem|eta_h|error|warning|failed|oom)(=|:)?([^\s,}]*)?/gi;
  const parts = [];
  let lastIndex = 0;
  for (const match of String(line).matchAll(pattern)) {
    if (match.index > lastIndex) {
      parts.push(String(line).slice(lastIndex, match.index));
    }
    parts.push(
      <span key={`${match.index}-${match[0]}`} className="log-token">
        {match[0]}
      </span>
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < String(line).length) {
    parts.push(String(line).slice(lastIndex));
  }
  return parts.length ? parts : line;
};

export default function LogViewer({ lines, chunkSize = 200 }) {
  const ref = useRef(null);
  const pinnedRef = useRef(true);
  const prevTailRef = useRef("");
  const suppressScrollRef = useRef(false);
  const [visibleCount, setVisibleCount] = useState(Math.min(lines.length, 400));

  useEffect(() => {
    setVisibleCount((prev) => Math.min(lines.length, Math.max(prev, 400)));
  }, [lines.length]);

  const scrollToBottom = () => {
    const el = ref.current;
    if (!el) return;
    suppressScrollRef.current = true;
    el.scrollTop = el.scrollHeight;
    pinnedRef.current = true;
    requestAnimationFrame(() => {
      if (ref.current) {
        ref.current.scrollTop = ref.current.scrollHeight;
      }
    });
    window.setTimeout(() => {
      if (ref.current) {
        ref.current.scrollTop = ref.current.scrollHeight;
      }
      suppressScrollRef.current = false;
      pinnedRef.current = true;
    }, 50);
  };

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const tail = `${lines.length}:${lines[lines.length - 1] || ""}`;
    const hasNewLines = tail !== prevTailRef.current;
    prevTailRef.current = tail;
    if (!hasNewLines || !pinnedRef.current) return;
    scrollToBottom();
  }, [lines]);

  const handleScroll = () => {
    const el = ref.current;
    if (!el) return;
    if (suppressScrollRef.current) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    pinnedRef.current = distanceFromBottom < 64;
    if (el.scrollTop === 0 && visibleCount < lines.length) {
      setVisibleCount((prev) => Math.min(lines.length, prev + chunkSize));
    }
  };

  const rendered = useMemo(() => {
    if (!lines.length) return [];
    return lines.slice(-visibleCount);
  }, [lines, visibleCount]);

  return (
    <div className="log-box" ref={ref} onScroll={handleScroll}>
      {rendered.length === 0 ? (
        <div className="muted">No logs yet</div>
      ) : (
        rendered.map((line, idx) => (
          <div className={classifyLine(line)} key={`${idx}-${line.slice(0, 16)}`}>
            {highlightParts(line)}
          </div>
        ))
      )}
    </div>
  );
}
