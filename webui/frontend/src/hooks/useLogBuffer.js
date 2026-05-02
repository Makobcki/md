import { useCallback, useEffect, useRef, useState } from "react";

const STORAGE_VERSION = "v1";

function safeParse(value) {
  if (!value) return null;
  try {
    return JSON.parse(value);
  } catch (err) {
    return null;
  }
}

function safeStorageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch (err) {
    console.warn("failed to read log buffer", err);
    return null;
  }
}

function safeStorageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (err) {
    console.warn("failed to persist log buffer", err);
  }
}

function loadLines(storageKey) {
  if (typeof window === "undefined" || !storageKey) return [];
  const stored = safeParse(safeStorageGet(`${storageKey}:${STORAGE_VERSION}`));
  if (!stored || !Array.isArray(stored.lines)) return [];
  return stored.lines.filter((line) => typeof line === "string");
}

export default function useLogBuffer(storageKey, { maxLines = 10000 } = {}) {
  const [lines, setLines] = useState(() => loadLines(storageKey));
  const persistRef = useRef(storageKey);

  useEffect(() => {
    if (persistRef.current !== storageKey) {
      persistRef.current = storageKey;
      setLines(loadLines(storageKey));
    }
  }, [storageKey]);

  useEffect(() => {
    if (typeof window === "undefined" || !storageKey) return;
    safeStorageSet(
      `${storageKey}:${STORAGE_VERSION}`,
      JSON.stringify({ lines })
    );
  }, [lines, storageKey]);

  const appendLines = useCallback(
    (next) => {
      if (!next) return;
      const incoming = Array.isArray(next) ? next : [next];
      setLines((prev) => {
        const merged = [...prev, ...incoming].filter((line) => typeof line === "string");
        if (merged.length <= maxLines) return merged;
        return merged.slice(-maxLines);
      });
    },
    [maxLines]
  );

  const replaceLines = useCallback(
    (next) => {
      const incoming = Array.isArray(next) ? next : [next];
      const filtered = incoming.filter((line) => typeof line === "string");
      setLines(filtered.length <= maxLines ? filtered : filtered.slice(-maxLines));
    },
    [maxLines]
  );

  const clear = useCallback(() => setLines([]), []);

  return { lines, appendLines, replaceLines, clear };
}
