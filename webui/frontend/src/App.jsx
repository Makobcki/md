import React, { useEffect, useMemo, useState } from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import TrainPage from "./pages/TrainPage.jsx";
import GeneratePage from "./pages/GeneratePage.jsx";
import RunDetails from "./pages/RunDetails.jsx";
import FilesPage from "./pages/FilesPage.jsx";
import PrepareLatentsPage from "./pages/PrepareLatentsPage.jsx";
import TrainSamplesPage from "./pages/TrainSamplesPage.jsx";
import { api, clearLegacyAuthToken } from "./api.js";
import StatusPill from "./components/StatusPill.jsx";
import { formatRunId, formatRunType } from "./utils/formatters.js";

const ICONS = {
  dashboard:
    "M520-600v-240h320v240H520ZM120-440v-400h320v400H120Zm400 320v-400h320v400H520Zm-400 0v-240h320v240H120Zm80-400h160v-240H200v240Zm400 320h160v-240H600v240Zm0-480h160v-80H600v80ZM200-200h160v-80H200v80Z",
  generate:
    "m176-120-56-56 301-302-181-45 198-123-17-234 179 151 216-88-87 217 151 178-234-16-124 198-45-181-301 301Zm379-323 48-79 93 7-60-71 35-86-86 35-71-59 7 92-79 49 90 22 23 90Z",
  train:
    "M280-160v-240l320-200v-120H440v-80h240v240L360-360v200h400v80H280Zm-80 0q-33 0-56.5-23.5T120-240v-480q0-33 23.5-56.5T200-800h120v80H200v480h80v80h-80Z",
  cache:
    "M160-160v-80h110l-16-14q-52-46-73-105t-21-119q0-111 66.5-197.5T400-790v84q-72 26-116 88.5T240-478q0 45 17 87.5t53 78.5l10 10v-98h80v240H160Zm400-10v-84q72-26 116-88.5T720-482q0-45-17-87.5T650-648l-10-10v98h-80v-240h240v80H690l16 14q49 49 71.5 106.5T800-482q0 111-66.5 197.5T560-170Z",
  logs:
    "M240-280h280v-80H240v80Zm0-160h480v-80H240v80Zm0-160h480v-80H240v80ZM160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h640q33 0 56.5 23.5T880-760v560q0 33-23.5 56.5T800-120H160Z",
  samples:
    "M200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm40-160h480L570-480 450-320l-90-120-120 160Z",
};

const navSections = [
  {
    label: "Monitor",
    items: [
      { to: "/", end: true, label: "Dashboard", icon: "dashboard" },
      { to: "/files", label: "Logs", icon: "logs" },
    ],
  },
  {
    label: "Workflows",
    items: [
      { to: "/generate", label: "Generate", icon: "generate" },
      { to: "/train", label: "Train", icon: "train" },
      { to: "/latents", label: "Latents", icon: "cache" },
    ],
  },
  {
    label: "Artifacts",
    items: [
      { to: "/train/samples", label: "Train samples", icon: "samples" },
    ],
  },
];

function NavIcon({ name }) {
  return (
    <svg viewBox="0 -960 960 960" aria-hidden="true">
      <path d={ICONS[name]} />
    </svg>
  );
}

function AuthPage({ onAuthenticated }) {
  const [token, setToken] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const status = await api.login(token);
      onAuthenticated(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "invalid auth token");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page">
      <form className="auth-panel" onSubmit={submit}>
        <div>
          <div className="brand-mark">md</div>
          <h1>md-diffusion WebUI</h1>
          <p className="muted">Enter WEBUI_AUTH_TOKEN to continue.</p>
        </div>
        <label className="auth-field">
          <span>Token</span>
          <input
            type="password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder="WEBUI_AUTH_TOKEN"
            aria-label="WebUI auth token"
            autoFocus
          />
        </label>
        {error ? <div className="auth-error">{error}</div> : null}
        <button type="submit" disabled={submitting || !token.trim()}>
          {submitting ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
}

function ThemeIcon({ dark }) {
  return dark ? (
    <svg viewBox="0 -960 960 960" aria-hidden="true">
      <path d="M480-280q83 0 141.5-58.5T680-480q0-83-58.5-141.5T480-680q-83 0-141.5 58.5T280-480q0 83 58.5 141.5T480-280Zm-40-480v-160h80v160h-80Zm0 720v-160h80v160h-80ZM200-440H40v-80h160v80Zm720 0H760v-80h160v80Z" />
    </svg>
  ) : (
    <svg viewBox="0 -960 960 960" aria-hidden="true">
      <path d="M480-120q-150 0-255-105T120-480q0-150 105-255t255-105q14 0 27.5 1t26.5 3q-41 29-65.5 75.5T444-660q0 90 63 153t153 63q55 0 101-24.5t75-65.5q2 13 3 26.5t1 27.5q0 150-105 255T480-120Z" />
    </svg>
  );
}

function AppShell({ authRequired, onLogout, theme, setTheme }) {
  const [status, setStatus] = useState({ active: false });
  const isDark = theme === "dark";

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const next = await api.getStatus();
        if (alive) setStatus(next);
      } catch {
        if (alive) setStatus({ active: false });
      }
    };
    load();
    const timer = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  const activeRun = status.active ? status.run : null;
  const runCaption = useMemo(() => {
    if (!activeRun) return "No active job";
    return `${formatRunType(activeRun.run_type)} · ${formatRunId(activeRun.run_id)}`;
  }, [activeRun]);

  return (
    <>
      <header className="app-sidebar">
        <div>
          <div className="brand-block">
            <div className="brand-mark">md</div>
            <div>
              <strong>md-diffusion</strong>
              <span>MMDiT RF workspace</span>
            </div>
          </div>
          <nav aria-label="Primary navigation">
            {navSections.map((section) => (
              <div key={section.label} className="nav-section">
                <div className="nav-section-label">{section.label}</div>
                {section.items.map((item) => (
                  <NavLink key={item.to} to={item.to} end={item.end}>
                    <NavIcon name={item.icon} />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </div>
            ))}
          </nav>
        </div>

        <div className="sidebar-footer">
          <div className="job-dock">
            <StatusPill status={activeRun?.status || "stopped"} />
            <div>
              <div className="job-dock-label">Runtime</div>
              <div className="job-dock-run" title={activeRun?.run_id || ""}>
                {runCaption}
              </div>
            </div>
          </div>
          <div className="header-actions">
            {authRequired ? (
              <button type="button" className="ghost" onClick={onLogout}>
                Logout
              </button>
            ) : null}
            <button
              type="button"
              className="theme-toggle"
              onClick={() => setTheme(isDark ? "light" : "dark")}
              aria-pressed={isDark}
              aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
              title={isDark ? "Light theme" : "Dark theme"}
            >
              <ThemeIcon dark={isDark} />
            </button>
          </div>
        </div>
      </header>
      <main className="app-shell">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/train/samples" element={<TrainSamplesPage />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/latents" element={<PrepareLatentsPage />} />
          <Route path="/files" element={<FilesPage />} />
          <Route path="/runs/:runId" element={<RunDetails />} />
        </Routes>
      </main>
    </>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "dark";
    return localStorage.getItem("ui-theme") || "dark";
  });

  const [authStatus, setAuthStatus] = useState({
    loading: true,
    auth_required: false,
    authenticated: false,
  });
  const [authError, setAuthError] = useState("");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ui-theme", theme);
  }, [theme]);

  useEffect(() => {
    clearLegacyAuthToken();
    let alive = true;
    api
      .getAuthStatus()
      .then((status) => {
        if (alive) {
          setAuthStatus({ loading: false, ...status });
          setAuthError("");
        }
      })
      .catch((err) => {
        if (alive) {
          setAuthStatus({ loading: false, auth_required: true, authenticated: false });
          setAuthError(err instanceof Error ? err.message : "Unable to check auth status");
        }
      });
    return () => {
      alive = false;
    };
  }, []);

  const authRequired = Boolean(authStatus.auth_required);
  const authenticated = !authRequired || Boolean(authStatus.authenticated);

  const logout = async () => {
    await api.logout().catch(() => null);
    setAuthStatus({ loading: false, auth_required: authRequired, authenticated: !authRequired });
  };

  if (authStatus.loading) {
    return (
      <div className="auth-page">
        <div className="preview-loader" />
      </div>
    );
  }

  if (!authenticated) {
    return (
      <>
        <AuthPage onAuthenticated={(status) => setAuthStatus({ loading: false, ...status })} />
        {authError ? <div className="auth-toast">{authError}</div> : null}
      </>
    );
  }

  return (
    <AppShell
      authRequired={authRequired}
      onLogout={logout}
      theme={theme}
      setTheme={setTheme}
    />
  );
}
