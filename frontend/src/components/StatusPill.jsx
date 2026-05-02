import React from "react";

const icons = {
  running: <span className="status-spinner" aria-hidden="true" />,
  stopped: (
    <svg viewBox="0 -960 960 960" aria-hidden="true">
      <path d="M520-200v-560h240v560H520Zm-320 0v-560h240v560H200Zm400-80h80v-400h-80v400Zm-320 0h80v-400h-80v400Zm0-400v400-400Zm320 0v400-400Z" />
    </svg>
  ),
  done: (
    <svg viewBox="0 -960 960 960" aria-hidden="true">
      <path d="M382-240 154-468l57-57 171 171 367-367 57 57-424 424Z" />
    </svg>
  ),
  failed: (
    <svg viewBox="0 -960 960 960" aria-hidden="true">
      <path d="m256-200-56-56 224-224-224-224 56-56 224 224 224-224 56 56-224 224 224 224-56 56-224-224-224 224Z" />
    </svg>
  ),
};

export default function StatusPill({ status, title }) {
  const icon = icons[status];
  return (
    <span className={`status-pill ${status || ""}`} title={title || status}>
      {icon || status}
    </span>
  );
}
