import React from "react";

export default function LineChart({ data, height = 110 }) {
  const cleaned = data
    .filter((d) => Number.isFinite(d.step) && Number.isFinite(d.loss))
    .sort((a, b) => a.step - b.step);

  if (!cleaned.length) {
    return <div className="chart muted" style={{ display: "grid", placeItems: "center", height, border: "1px solid var(--border)", background: "var(--panel-alt)", fontFamily: "Space Grotesk, sans-serif", fontSize: "12px", borderRadius: "var(--radius)" }}>No metric data</div>;
  }

  const width = 600;
  const paddingLeft = 45;
  const paddingRight = 15;
  const paddingTop = 10;
  const paddingBottom = 16;

  const xs = cleaned.map((d) => d.step);
  const ys = cleaned.map((d) => d.loss);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const yRange = Math.max(1e-9, maxY - minY);

  const scaleX = (x) =>
    paddingLeft + ((x - minX) / Math.max(1, maxX - minX)) * (width - paddingLeft - paddingRight);
  const scaleY = (y) =>
    height - paddingBottom - ((y - minY) / yRange) * (height - paddingTop - paddingBottom);

  const points = cleaned.map((d) => `${scaleX(d.step)},${scaleY(d.loss)}`).join(" ");

  // Create area path
  let areaPoints = "";
  if (cleaned.length > 0) {
    const firstX = scaleX(cleaned[0].step);
    const lastX = scaleX(cleaned[cleaned.length - 1].step);
    const zeroY = height - paddingBottom;
    areaPoints = `${firstX},${zeroY} ${points} ${lastX},${zeroY}`;
  }

  // Grid lines
  const gridYCount = 3;
  const gridLinesY = [];
  for (let i = 0; i <= gridYCount; i++) {
    const val = minY + (yRange / gridYCount) * i;
    gridLinesY.push({
      y: scaleY(val),
      val: val.toFixed(4)
    });
  }

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" height={height} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id="chart-area-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.25" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0.0" />
        </linearGradient>
      </defs>

      {/* Grid lines */}
      {gridLinesY.map((line, idx) => (
        <g key={idx}>
          <line
            x1={paddingLeft}
            y1={line.y}
            x2={width - paddingRight}
            y2={line.y}
            stroke="var(--border)"
            strokeDasharray="4,4"
            strokeWidth="1"
          />
          <text
            x={paddingLeft - 8}
            y={line.y + 4}
            fill="var(--muted)"
            fontSize="10"
            fontFamily="JetBrains Mono, monospace"
            textAnchor="end"
          >
            {line.val}
          </text>
        </g>
      ))}

      {/* Area under curve */}
      {areaPoints && (
        <polygon points={areaPoints} fill="url(#chart-area-grad)" />
      )}

      {/* Trendline */}
      <polyline fill="none" stroke="var(--accent)" strokeWidth="2.5" points={points} strokeLinecap="round" strokeLinejoin="round" />

      {/* Data points */}
      {cleaned.map((d) => (
        <circle
          key={d.step}
          cx={scaleX(d.step)}
          cy={scaleY(d.loss)}
          r="3"
          fill="var(--accent)"
          stroke="var(--panel)"
          strokeWidth="1.5"
        />
      ))}
    </svg>
  );
}
