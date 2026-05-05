import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, absoluteFileUrl, absoluteDownloadUrl } from "../api.js";


export default function TrainSamplesPage() {
  const [samples, setSamples] = useState([]);

  useEffect(() => {
    const load = async () => {
      const data = await api.listTrainSamples();
      setSamples(data.items || []);
    };
    load();
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, []);

  const filtered = useMemo(() => [...samples].reverse(), [samples]);

  return (
    <div className="page">
      <div className="row">
        <h1 className="page-title">Train Samples</h1>
        <Link className="muted" to="/train">
          Back to Train
        </Link>
      </div>
      <div className="card">
        {filtered.length === 0 ? (
          <div className="muted">No samples yet.</div>
        ) : (
          <div className="gallery-grid">
            {filtered.map((item) => {
              const url = absoluteFileUrl(item);
              return url ? (
                <div key={item.path || item.url || item} className="image-card">
                  <img src={url} alt="sample" />
                  <div className="image-meta">
                    <span className="badge">{String(item.path || item).split("/").pop()}</span>
                    <div className="row">
                      <a href={url} target="_blank" rel="noreferrer">
                        Open full
                      </a>
                      <a href={absoluteDownloadUrl(item) || url} download>
                        Download
                      </a>
                    </div>
                  </div>
                </div>
              ) : null;
            })}
          </div>
        )}
      </div>
    </div>
  );
}
