import { useEffect, useRef } from "react";
import { wsUrl } from "../api.js";

export default function useRunLogStream(
  runId,
  { enabled = true, backlog = 2000, onLog, onStatus, onError } = {}
) {
  const onLogRef = useRef(onLog);
  const onStatusRef = useRef(onStatus);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onLogRef.current = onLog;
    onStatusRef.current = onStatus;
    onErrorRef.current = onError;
  }, [onLog, onStatus, onError]);

  useEffect(() => {
    if (!enabled || !runId) return undefined;

    const params = new URLSearchParams({
      backlog: String(Math.max(0, Math.floor(Number(backlog) || 0))),
    });
    let closed = false;
    const ws = new WebSocket(wsUrl(`/ws/logs/${encodeURIComponent(runId)}?${params.toString()}`));

    ws.onmessage = (event) => {
      if (closed) return;
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "log") {
          onLogRef.current?.(payload);
        } else if (payload.type === "status") {
          onStatusRef.current?.(payload);
        }
      } catch (err) {
        onErrorRef.current?.(err);
      }
    };
    ws.onerror = () => {
      if (closed) return;
      onErrorRef.current?.(new Error("log stream connection failed"));
    };

    return () => {
      closed = true;
      ws.close();
    };
  }, [runId, enabled, backlog]);
}
