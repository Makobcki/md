import React from "react";
import { Routes, Route, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import TrainPage from "./pages/TrainPage.jsx";
import GeneratePage from "./pages/GeneratePage.jsx";
import RunDetails from "./pages/RunDetails.jsx";
import FilesPage from "./pages/FilesPage.jsx";

export default function App() {
  return (
    <>
      <header>
        <strong>Diffusion Web UI</strong>
        <nav className="row">
          <Link to="/">Dashboard</Link>
          <Link to="/train">Train</Link>
          <Link to="/generate">Generate</Link>
          <Link to="/files">Files/Logs</Link>
        </nav>
      </header>
      <div className="container">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/train" element={<TrainPage />} />
          <Route path="/generate" element={<GeneratePage />} />
          <Route path="/files" element={<FilesPage />} />
          <Route path="/runs/:runId" element={<RunDetails />} />
        </Routes>
      </div>
    </>
  );
}
