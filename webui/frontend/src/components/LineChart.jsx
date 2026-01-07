import React from "react";

export default function LineChart({ data, height = 180 }) {
  if (!data.length) {
    return <div className="chart muted">Нет данных</div>;
  }

  const width = 600;
  const padding = 20;
  const xs = data.map((d) => d.step);
  const ys = data.map((d) => d.loss);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const scaleX = (x) =>
    padding + ((x - minX) / Math.max(1, maxX - minX)) * (width - padding * 2);
  const scaleY = (y) =>
    height - padding - ((y - minY) / Math.max(1e-9, maxY - minY)) * (height - padding * 2);

  const points = data.map((d) => `${scaleX(d.step)},${scaleY(d.loss)}`).join(" ");

  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
      <polyline fill="none" stroke="#60a5fa" strokeWidth="2" points={points} />
    </svg>
  );
}
