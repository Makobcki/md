import React, { useMemo, useRef } from "react";

const escapeHtml = (value) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

const highlightYaml = (value) => {
  const escaped = escapeHtml(value);
  return escaped
    .replace(/(^\s*#.*$)/gm, '<span class="token comment">$1</span>')
    .replace(/(^\s*[\w-]+:)/gm, '<span class="token key">$1</span>')
    .replace(/(:\s*)("[^"]*"|'[^']*')/g, '$1<span class="token string">$2</span>')
    .replace(/(:\s*)(\d+(?:\.\d+)?)/g, '$1<span class="token number">$2</span>');
};

export default function YamlEditor({ value, onChange, onSave }) {
  const highlightRef = useRef(null);
  const inputRef = useRef(null);
  const highlighted = useMemo(() => `${highlightYaml(value)}\n`, [value]);

  const syncScroll = () => {
    if (!highlightRef.current || !inputRef.current) return;
    highlightRef.current.scrollTop = inputRef.current.scrollTop;
    highlightRef.current.scrollLeft = inputRef.current.scrollLeft;
  };

  const handleKeyDown = (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      if (onSave) {
        onSave();
      }
    }
  };

  return (
    <div className="editor-shell">
      <pre
        className="editor-highlight"
        aria-hidden="true"
        ref={highlightRef}
        dangerouslySetInnerHTML={{ __html: highlighted }}
      />
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
