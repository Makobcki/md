import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("UI render failed", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="app-shell">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">UI error</h2>
            </div>
            <div className="muted">{this.state.error.message}</div>
            <button
              style={{ marginTop: "12px" }}
              onClick={() => this.setState({ error: null })}
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
