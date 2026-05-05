import React, { useEffect, useState } from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import TrainPage from "./pages/TrainPage.jsx";
import GeneratePage from "./pages/GeneratePage.jsx";
import RunDetails from "./pages/RunDetails.jsx";
import FilesPage from "./pages/FilesPage.jsx";
import PrepareLatentsPage from "./pages/PrepareLatentsPage.jsx";
import TrainSamplesPage from "./pages/TrainSamplesPage.jsx";
import { getAuthToken, setAuthToken } from "./api.js";

const navItems = [
  {
    to: "/",
    end: true,
    label: "Dashboard",
    icon: (
      <svg viewBox="0 -960 960 960" aria-hidden="true">
        <path d="M520-600v-240h320v240H520ZM120-440v-400h320v400H120Zm400 320v-400h320v400H520Zm-400 0v-240h320v240H120Zm80-400h160v-240H200v240Zm400 320h160v-240H600v240Zm0-480h160v-80H600v80ZM200-200h160v-80H200v80Zm160-320Zm240-160Zm0 240ZM360-280Z" />
      </svg>
    ),
  },
  {
    to: "/generate",
    label: "Generate",
    icon: (
      <svg viewBox="0 -960 960 960" aria-hidden="true">
        <path d="m176-120-56-56 301-302-181-45 198-123-17-234 179 151 216-88-87 217 151 178-234-16-124 198-45-181-301 301Zm24-520-80-80 80-80 80 80-80 80Zm355 197 48-79 93 7-60-71 35-86-86 35-71-59 7 92-79 49 90 22 23 90Zm165 323-80-80 80-80 80 80-80 80ZM569-570Z" />
      </svg>
    ),
  },
  {
    to: "/train",
    label: "Train",
    icon: (
      <svg viewBox="0 -960 960 960" aria-hidden="true">
        <path d="m352-522 86-87-56-57-44 44-56-56 43-44-45-45-87 87 159 158Zm328 329 87-87-45-45-44 43-56-56 43-44-57-56-86 86 158 159Zm24-567 57 57-57-57ZM290-120H120v-170l175-175L80-680l200-200 216 216 151-152q12-12 27-18t31-6q16 0 31 6t27 18l53 54q12 12 18 27t6 31q0 16-6 30.5T816-647L665-495l215 215L680-80 465-295 290-120Zm-90-80h56l392-391-57-57-391 392v56Zm420-419-29-29 57 57-28-28Z" />
      </svg>
    ),
  },
  {
    to: "/latents",
    label: "Prepare Latents",
    icon: (
      <svg viewBox="0 -960 960 960" aria-hidden="true">
        <path d="M160-160v-80h110l-16-14q-52-46-73-105t-21-119q0-111 66.5-197.5T400-790v84q-72 26-116 88.5T240-478q0 45 17 87.5t53 78.5l10 10v-98h80v240H160Zm400-10v-84q72-26 116-88.5T720-482q0-45-17-87.5T650-648l-10-10v98h-80v-240h240v80H690l16 14q49 49 71.5 106.5T800-482q0 111-66.5 197.5T560-170Z" />
      </svg>
    ),
  },
  {
    to: "/files",
    label: "Logs",
    icon: (
      <svg viewBox="0 -960 960 960" aria-hidden="true">
        <path d="M308.5-291.5Q320-303 320-320t-11.5-28.5Q297-360 280-360t-28.5 11.5Q240-337 240-320t11.5 28.5Q263-280 280-280t28.5-11.5ZM240-440h80v-240h-80v240Zm200 160h280v-80H440v80Zm0-160h280v-80H440v80Zm0-160h280v-80H440v80ZM160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h640q33 0 56.5 23.5T880-760v560q0 33-23.5 56.5T800-120H160Zm0-80h640v-560H160v560Zm0 0v-560 560Z" />
      </svg>
    ),
  },
];

function AuthControls() {
  const [token, setToken] = useState(() => getAuthToken());
  const [savedToken, setSavedToken] = useState(() => getAuthToken());

  const save = () => {
    const next = token.trim();
    setAuthToken(next);
    setSavedToken(next);
  };
  const clear = () => {
    setToken("");
    setAuthToken("");
    setSavedToken("");
  };

  return (
    <div className="auth-controls">
      <input
        type="password"
        value={token}
        onChange={(event) => setToken(event.target.value)}
        placeholder="WEBUI_AUTH_TOKEN"
        aria-label="WebUI auth token"
      />
      <button type="button" className="secondary" onClick={save} title="Save token in this browser">
        {savedToken ? "Update token" : "Save token"}
      </button>
      {savedToken ? (
        <button type="button" className="ghost" onClick={clear} title="Clear saved token">
          Logout
        </button>
      ) : null}
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "light";
    return localStorage.getItem("ui-theme") || "light";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ui-theme", theme);
  }, [theme]);

  const isDark = theme === "dark";

  return (
    <>
      <header>
        <div>
          <strong>Diffusion Web UI</strong>
          <nav>
            {navItems.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end}>
                {item.icon}
                <span>{item.label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
        <div className="header-actions">
          <AuthControls />
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme(isDark ? "light" : "dark")}
          aria-pressed={isDark}
          aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
          title={isDark ? "Light theme" : "Dark theme"}
        >
          {isDark ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M565-395q35-35 35-85t-35-85q-35-35-85-35t-85 35q-35 35-35 85t35 85q35 35 85 35t85-35Zm-226.5 56.5Q280-397 280-480t58.5-141.5Q397-680 480-680t141.5 58.5Q680-563 680-480t-58.5 141.5Q563-280 480-280t-141.5-58.5ZM200-440H40v-80h160v80Zm720 0H760v-80h160v80ZM440-760v-160h80v160h-80Zm0 720v-160h80v160h-80ZM256-650l-101-97 57-59 96 100-52 56Zm492 496-97-101 53-55 101 97-57 59Zm-98-550 97-101 59 57-100 96-56-52ZM154-212l101-97 55 53-97 101-59-57Zm326-268Z" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              height="24px"
              viewBox="0 -960 960 960"
              width="24px"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M480-120q-150 0-255-105T120-480q0-150 105-255t255-105q14 0 27.5 1t26.5 3q-41 29-65.5 75.5T444-660q0 90 63 153t153 63q55 0 101-24.5t75-65.5q2 13 3 26.5t1 27.5q0 150-105 255T480-120Zm0-80q88 0 158-48.5T740-375q-20 5-40 8t-40 3q-123 0-209.5-86.5T364-660q0-20 3-40t8-40q-78 32-126.5 102T200-480q0 116 82 198t198 82Zm-10-270Z" />
            </svg>
          )}
        </button>
        </div>
      </header>
      <div className="app-shell">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/train/samples" element={<TrainSamplesPage />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/latents" element={<PrepareLatentsPage />} />
          <Route path="/files" element={<FilesPage />} />
          <Route path="/runs/:runId" element={<RunDetails />} />
        </Routes>
      </div>
    </>
  );
}
