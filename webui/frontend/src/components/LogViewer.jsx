import React, { useEffect, useRef } from "react";

export default function LogViewer({ lines }) {
  const ref = useRef(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div className="log-box" ref={ref}>
      {lines.map((line, idx) => (
        <div key={idx}>{line}</div>
      ))}
    </div>
  );
}
