import React from "react";

export default function MetricTile({ label, value, detail = "", tone = "" }) {
  return (
    <div className={`metric-tile ${tone ? `tone-${tone}` : ""}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {detail ? <div className="metric-detail">{detail}</div> : null}
    </div>
  );
}
