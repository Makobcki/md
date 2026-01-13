import React from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import TrainPage from "./pages/TrainPage.jsx";
import GeneratePage from "./pages/GeneratePage.jsx";
import RunDetails from "./pages/RunDetails.jsx";
import FilesPage from "./pages/FilesPage.jsx";
import PrepareLatentsPage from "./pages/PrepareLatentsPage.jsx";
import TrainSamplesPage from "./pages/TrainSamplesPage.jsx";

export default function App() {
  return (
    <>
      <header>
        <strong>Diffusion Web UI</strong>
        <nav>
          <NavLink to="/" end>
            Dashboard
          </NavLink>
          <NavLink to="/generate">Generate</NavLink>
          <NavLink to="/train">Train</NavLink>
          <NavLink to="/latents">Prepare Latents</NavLink>
          <NavLink to="/files">Files / Logs</NavLink>
        </nav>
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
