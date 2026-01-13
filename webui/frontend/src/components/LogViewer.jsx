import React, { useEffect, useMemo, useRef, useState } from "react";

export default function LogViewer({ lines, autoScroll = true, chunkSize = 200 }) {
  const ref = useRef(null);
  const [visibleCount, setVisibleCount] = useState(Math.min(lines.length, 400));

  useEffect(() => {
    setVisibleCount((prev) => Math.min(lines.length, Math.max(prev, 400)));
  }, [lines.length]);

  useEffect(() => {
    if (!ref.current || !autoScroll) return;
    ref.current.scrollTop = ref.current.scrollHeight;
  }, [lines, autoScroll]);

  const handleScroll = () => {
    const el = ref.current;
    if (!el) return;
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
        rendered.map((line, idx) => <div key={`${idx}-${line.slice(0, 16)}`}>{line}</div>)
      )}
    </div>
  );
}
