import React from "react";

export default function PageHeader({
  title,
  eyebrow = "",
  description = "",
  meta = null,
  actions = null,
}) {
  return (
    <div className="page-header">
      <div className="page-header-copy">
        {eyebrow ? <div className="page-eyebrow">{eyebrow}</div> : null}
        <h1 className="page-title">{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {(meta || actions) ? (
        <div className="page-header-side">
          {meta}
          {actions ? <div className="row page-actions">{actions}</div> : null}
        </div>
      ) : null}
    </div>
  );
}
