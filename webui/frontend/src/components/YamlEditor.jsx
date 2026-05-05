import React, { useMemo, useRef } from "react";

const classifyLine = (line) => {
  if (/^\s*#/.test(line)) return [{ text: line, cls: "comment" }];
  const keyMatch = line.match(/^(\s*[\w-]+:)(.*)$/);
  if (!keyMatch) return [{ text: line, cls: "" }];
  const [, key, rest] = keyMatch;
  const valueParts = [];
  const stringMatch = rest.match(/^(\s*)("[^"]*"|'[^']*')(.*)$/);
  const numberMatch = rest.match(/^(\s*)(\d+(?:\.\d+)?)(.*)$/);
  if (stringMatch) {
    valueParts.push({ text: stringMatch[1], cls: "" }, { text: stringMatch[2], cls: "string" }, { text: stringMatch[3], cls: "" });
  } else if (numberMatch) {
    valueParts.push({ text: numberMatch[1], cls: "" }, { text: numberMatch[2], cls: "number" }, { text: numberMatch[3], cls: "" });
  } else {
    valueParts.push({ text: rest, cls: "" });
  }
  return [{ text: key, cls: "key" }, ...valueParts];
};

export default function YamlEditor({ value, onChange, onSave }) {
  const highlightRef = useRef(null);
  const inputRef = useRef(null);
  const highlightedLines = useMemo(() => `${value}\n`.split("\n").map(classifyLine), [value]);

  const syncScroll = () => {
    if (!highlightRef.current || !inputRef.current) return;
    highlightRef.current.scrollTop = inputRef.current.scrollTop;
    highlightRef.current.scrollLeft = inputRef.current.scrollLeft;
  };

  const handleKeyDown = (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      if (onSave) onSave();
    }
  };

  return (
    <div className="editor-shell">
      <pre className="editor-highlight" aria-hidden="true" ref={highlightRef}>
        {highlightedLines.map((parts, lineIdx) => (
          <React.Fragment key={lineIdx}>
            {parts.map((part, partIdx) =>
              part.cls ? (
                <span key={partIdx} className={`token ${part.cls}`}>{part.text}</span>
              ) : (
                <React.Fragment key={partIdx}>{part.text}</React.Fragment>
              )
            )}
            {lineIdx < highlightedLines.length - 1 ? "\n" : null}
          </React.Fragment>
        ))}
      </pre>
      <textarea
        className="editor-input"
        ref={inputRef}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onScroll={syncScroll}
        onKeyDown={handleKeyDown}
        spellCheck={false}
      />
    </div>
  );
}
