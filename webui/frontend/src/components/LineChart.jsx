import React from "react";

export default function LineChart({ data, height = 180 }) {
  const cleaned = data
    .filter((d) => Number.isFinite(d.step) && Number.isFinite(d.loss))
    .sort((a, b) => a.step - b.step);

  if (!cleaned.length) {
    return <div className="chart muted">Нет данных</div>;
  }

  const width = 600;
  const padding = 20;
  const xs = cleaned.map((d) => d.step);
  const ys = cleaned.map((d) => d.loss);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const yRange = Math.max(1e-9, maxY - minY);

  const scaleX = (x) =>
    padding + ((x - minX) / Math.max(1, maxX - minX)) * (width - padding * 2);
  const scaleY = (y) =>
    height - padding - ((y - minY) / yRange) * (height - padding * 2);

  const points = cleaned.map((d) => `${scaleX(d.step)},${scaleY(d.loss)}`).join(" ");

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
      <rect x="0" y="0" width={width} height={height} fill="transparent" />
      <polyline fill="none" stroke="#60a5fa" strokeWidth="2" points={points} />
      {cleaned.map((d) => (
        <circle
          key={d.step}
          cx={scaleX(d.step)}
          cy={scaleY(d.loss)}
          r="2"
          fill="#60a5fa"
        />
      ))}
    </svg>
  );
}
