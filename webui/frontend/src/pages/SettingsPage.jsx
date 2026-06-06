import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api.js";
import ArgField from "../components/ArgField.jsx";
import PageHeader from "../components/PageHeader.jsx";
import StatusPill from "../components/StatusPill.jsx";
import YamlEditor from "../components/YamlEditor.jsx";
import {
  buildArgsWithStoredSettings,
  clearStoredSettings,
  LATENT_SETTINGS_KEY,
  resetArgsToDefaults,
  SAMPLE_SETTINGS_KEY,
  summarizeLatentArgs,
  summarizeSampleArgs,
  writeStoredSettings,
} from "../utils/settingsModel.js";
import { extractConfigUses } from "../utils/uiModel.js";

const presetKindOrder = ["model", "training", "data", "text", "sampler", "webui"];

const presetKindLabels = {
  model: "Model",
  training: "Training",
  data: "Data",
  text: "Text",
  sampler: "Sampler",
  webui: "WebUI",
};

const sampleGroups = [
  {
    title: "Генерация",
    names: ["family", "task", "ckpt", "sampler"],
  },
  {
    title: "Качество и размер",
    names: ["steps", "n", "seed", "width", "height", "cfg", "shift"],
  },
  {
    title: "Image / inpaint / control",
    names: ["strength", "control-strength", "control-type", "fake-vae"],
  },
  {
    title: "Система и вывод",
    names: ["device", "out"],
  },
];

const latentGroups = [
  {
    title: "Файловая система / Кэш",
    names: ["config", "limit", "shard-size", "decode-backend"],
  },
  {
    title: "Производительность",
    names: [
      "batch-size",
      "num-workers",
      "prefetch-factor",
      "pin-memory",
      "queue-size",
      "writer-threads",
      "stats-every-sec",
    ],
  },
  {
    title: "GPU",
    names: ["device", "latent-dtype", "autocast-dtype"],
  },
];

const sampleHiddenFields = new Set(["prompt", "neg_prompt", "init-image", "mask", "control-image"]);
const tabs = [
  { key: "config", label: "Конфиг" },
  { key: "generate", label: "Генерация" },
  { key: "latents", label: "Кэш латентов" },
];

function stableSnapshot(value) {
  return JSON.stringify(value || {}, Object.keys(value || {}).sort());
}

function insertPresetUse(content, kind, name) {
  const section = kind === "sampler" ? "sampling" : kind;
  const lines = String(content || "").split("\n");
  const useLine = `    use "${name}"`;
  
  let inConfig = false;
  let inSection = false;
  let configDepth = 0;
  let sectionDepth = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const openBrackets = (line.match(/\{/g) || []).length;
    const closeBrackets = (line.match(/\}/g) || []).length;
    
    if (/^\s*config\s*\{/.test(line)) {
      inConfig = true;
      configDepth = 1;
      continue;
    }
    
    if (inConfig) {
      configDepth += openBrackets - closeBrackets;
      if (configDepth <= 0) {
        inConfig = false;
        continue;
      }
      
      if (new RegExp(`^\\s*${section}\\s*\\{`).test(line)) {
        inSection = true;
        sectionDepth = 1;
        const inlineUse = line.match(/use\s+"([^"]+)"/);
        if (inlineUse) {
          lines[i] = line.replace(/use\s+"[^"]+"/, `use "${name}"`);
          return lines.join("\n");
        }
        continue;
      }
      
      if (inSection) {
        if (/^\s*use\s+"[^"]+"/.test(line)) {
          lines[i] = line.replace(/use\s+"[^"]+"/, `use "${name}"`);
          return lines.join("\n");
        }
        
        sectionDepth += openBrackets - closeBrackets;
        if (sectionDepth <= 0) {
          inSection = false;
        }
      }
    }
  }

  // Fallback to insertion
  let depth = 0;
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const startsSection = depth === 1 && new RegExp(`^\\s*${section}\\s*\\{`).test(line);
    if (startsSection) {
      lines.splice(index + 1, 0, useLine);
      return lines.join("\n");
    }
    depth += (line.match(/\{/g) || []).length;
    depth -= (line.match(/\}/g) || []).length;
  }

  const insertAt = Math.max(1, lines.findIndex((line) => /^\s*}\s*$/.test(line)));
  const block = [`  ${section} {`, `    use "${name}"`, "  }", ""];
  if (insertAt > 0) {
    lines.splice(insertAt, 0, ...block);
    return lines.join("\n");
  }
  return `${content.trimEnd()}\n\n${block.join("\n")}`;
}

function familyAwareSpec(spec, family) {
  if (spec.name === "task" && family !== "mmdit") {
    return { ...spec, choices: ["txt2img"] };
  }
  if (spec.name === "sampler" && family === "var") {
    return { ...spec, choices: ["var_autoregressive"] };
  }
  if (spec.name === "sampler") {
    return { ...spec, choices: ["flow_euler", "flow_heun"] };
  }
  return spec;
}

function isSampleSpecVisible(spec, family) {
  if (sampleHiddenFields.has(spec.name)) return false;
  if (family !== "mmdit" && ["strength", "control-strength", "control-type"].includes(spec.name)) return false;
  if (family === "var" && ["cfg", "shift", "width", "height", "fake-vae"].includes(spec.name)) return false;
  return true;
}

function groupSpecs(specs, groups) {
  const byName = new Map(specs.map((spec) => [spec.name, spec]));
  const used = new Set();
  const result = groups
    .map((group) => ({
      ...group,
      specs: group.names.map((name) => byName.get(name)).filter(Boolean),
    }))
    .filter((group) => group.specs.length > 0);

  result.forEach((group) => group.specs.forEach((spec) => used.add(spec.name)));
  const other = specs.filter((spec) => !used.has(spec.name));
  if (other.length > 0) {
    result.push({ title: "Прочее", specs: other });
  }
  return result;
}

function SettingsSummary({ title, items }) {
  return (
    <div className="settings-summary-card">
      <div className="card-title">{title}</div>
      <div className="settings-summary-grid">
        {items.map((item) => (
          <div key={`${item.label}:${item.value}`} className="settings-summary-item">
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function PresetLibrary({
  presets,
  selectedKind,
  selectedName,
  onSelectKind,
  onSelectPreset,
  activeUses,
  onInsert,
}) {
  const groups = presets?.groups || {};
  const kinds = presetKindOrder.filter((kind) => (groups[kind] || []).length > 0);
  const items = groups[selectedKind] || [];
  const selected = items.find((item) => item.name === selectedName) || items[0] || null;
  const isActive = (item) => {
    const activeKind = item.kind === "sampler" ? "sampling" : item.kind;
    return (activeUses[activeKind] || activeUses[item.kind] || []).includes(item.name);
  };

  return (
    <div className="preset-library">
      <div className="preset-kind-tabs" role="tablist" aria-label="Preset groups">
        {kinds.map((kind) => (
          <button
            key={kind}
            type="button"
            className={kind === selectedKind ? "active" : ""}
            onClick={() => onSelectKind(kind)}
          >
            {presetKindLabels[kind] || kind}
          </button>
        ))}
      </div>

      <div className="preset-browser">
        <div className="preset-list">
          {items.map((item) => {
            const active = isActive(item);
            return (
              <button
                key={`${item.kind}:${item.name}`}
                type="button"
                className={`preset-list-item ${selected?.name === item.name ? "selected" : ""}`}
                onClick={() => onSelectPreset(item.name)}
              >
                <span>
                  <strong>{item.name}</strong>
                  <small>{item.summary || item.relative_path}</small>
                </span>
                {active ? <em>active</em> : null}
              </button>
            );
          })}
        </div>

        <div className="preset-detail">
          {selected ? (
            <>
              <div className="preset-detail-head">
                <div>
                  <h3>{selected.name}</h3>
                  <p>{selected.summary || selected.relative_path}</p>
                </div>
                <button
                  type="button"
                  className={isActive(selected) ? "secondary" : ""}
                  onClick={() => onInsert(selected)}
                  disabled={isActive(selected)}
                >
                  {isActive(selected) ? "Active" : "Activate preset"}
                </button>
              </div>
              <div className="preset-meta-row">
                <span className="badge">{selected.kind}</span>
                <span className="badge">v{selected.version ?? "?"}</span>
                <span className="badge">{selected.relative_path}</span>
              </div>
              <pre className="preset-preview">{selected.content}</pre>
            </>
          ) : (
            <div className="empty-state">No presets in this group.</div>
          )}
        </div>
      </div>
    </div>
  );
}

function SettingsGroups({ groups, values, onChange, checkpoints }) {
  return (
    <div className="settings-groups">
      {groups.map((group) => (
        <section key={group.title} className="settings-section">
          <h3>{group.title}</h3>
          <div className="flat-grid">
            {group.specs.map((spec) => (
              <ArgField
                key={spec.name}
                spec={spec}
                value={values[spec.name]}
                onChange={onChange}
                checkpoints={checkpoints}
                variant="flat"
              />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("config");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [config, setConfig] = useState("");
  const [lastSavedConfig, setLastSavedConfig] = useState("");
  const [savingConfig, setSavingConfig] = useState(false);
  const [presets, setPresets] = useState({ groups: {}, active: {} });
  const [selectedPresetKind, setSelectedPresetKind] = useState("model");
  const [selectedPresetName, setSelectedPresetName] = useState("");
  const [sampleSpecs, setSampleSpecs] = useState([]);
  const [sampleArgs, setSampleArgs] = useState({});
  const [sampleSnapshot, setSampleSnapshot] = useState("{}");
  const [latentSpecs, setLatentSpecs] = useState([]);
  const [latentArgs, setLatentArgs] = useState({});
  const [latentSnapshot, setLatentSnapshot] = useState("{}");
  const [checkpoints, setCheckpoints] = useState([]);
  const [savedMessage, setSavedMessage] = useState("");

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const [cfg, presetData, sampleData, latentData, ckptData] = await Promise.all([
          api.getConfig(),
          api.getConfigPresets(),
          api.getSampleArgs(),
          api.getLatentArgs(),
          api.listCheckpoints(),
        ]);
        const ckptItems = ckptData.items || [];
        const nextSampleSpecs = sampleData.items || [];
        const nextLatentSpecs = latentData.items || [];
        const nextSampleArgs = buildArgsWithStoredSettings(
          nextSampleSpecs,
          SAMPLE_SETTINGS_KEY,
          { checkpoints: ckptItems }
        );
        const nextLatentArgs = buildArgsWithStoredSettings(nextLatentSpecs, LATENT_SETTINGS_KEY);

        setConfig(cfg.content || "");
        setLastSavedConfig(cfg.content || "");
        setPresets(presetData);
        setSampleSpecs(nextSampleSpecs);
        setSampleArgs(nextSampleArgs);
        setSampleSnapshot(stableSnapshot(nextSampleArgs));
        setLatentSpecs(nextLatentSpecs);
        setLatentArgs(nextLatentArgs);
        setLatentSnapshot(stableSnapshot(nextLatentArgs));
        setCheckpoints(ckptItems);

        const firstKind = presetKindOrder.find((kind) => (presetData.groups?.[kind] || []).length > 0);
        if (firstKind) {
          setSelectedPresetKind(firstKind);
          setSelectedPresetName(presetData.groups[firstKind][0]?.name || "");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Не удалось загрузить настройки");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const configDirty = config !== lastSavedConfig;
  const sampleDirty = stableSnapshot(sampleArgs) !== sampleSnapshot;
  const latentDirty = stableSnapshot(latentArgs) !== latentSnapshot;
  const activeUses = useMemo(() => extractConfigUses(config), [config]);
  const sampleFamily = sampleArgs.family || "mmdit";

  const visibleSampleGroups = useMemo(() => {
    const visibleSpecs = sampleSpecs
      .filter((spec) => isSampleSpecVisible(spec, sampleFamily))
      .map((spec) => familyAwareSpec(spec, sampleFamily));
    return groupSpecs(visibleSpecs, sampleGroups);
  }, [sampleSpecs, sampleFamily]);

  const visibleLatentGroups = useMemo(() => groupSpecs(latentSpecs, latentGroups), [latentSpecs]);

  const handleSaveConfig = async () => {
    setSavingConfig(true);
    setError("");
    try {
      await api.updateConfig(config);
      setLastSavedConfig(config);
      setSavedMessage("Конфигурация сохранена");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось сохранить конфигурацию");
    } finally {
      setSavingConfig(false);
    }
  };

  const handleSampleChange = (name, value) => {
    setSavedMessage("");
    setSampleArgs((prev) => {
      if (name === "family") {
        return {
          ...prev,
          family: value,
          task: "txt2img",
          sampler: value === "var" ? "var_autoregressive" : "flow_heun",
          "init-image": "",
          mask: "",
          "control-image": "",
        };
      }
      if (name === "task") {
        return {
          ...prev,
          task: value,
          "init-image": "",
          mask: "",
          "control-image": "",
        };
      }
      return { ...prev, [name]: value };
    });
  };

  const handleLatentChange = (name, value) => {
    setSavedMessage("");
    setLatentArgs((prev) => ({ ...prev, [name]: value }));
  };

  const saveSampleSettings = () => {
    writeStoredSettings(SAMPLE_SETTINGS_KEY, sampleArgs);
    setSampleSnapshot(stableSnapshot(sampleArgs));
    setSavedMessage("Настройки генерации сохранены");
  };

  const saveLatentSettings = () => {
    writeStoredSettings(LATENT_SETTINGS_KEY, latentArgs);
    setLatentSnapshot(stableSnapshot(latentArgs));
    setSavedMessage("Настройки кэша сохранены");
  };

  const resetSampleSettings = () => {
    const defaults = resetArgsToDefaults(sampleSpecs, SAMPLE_SETTINGS_KEY, { checkpoints });
    setSampleArgs(defaults);
    setSampleSnapshot(stableSnapshot(defaults));
    setSavedMessage("Настройки генерации сброшены");
  };

  const resetLatentSettings = () => {
    clearStoredSettings(LATENT_SETTINGS_KEY);
    const defaults = resetArgsToDefaults(latentSpecs, LATENT_SETTINGS_KEY);
    setLatentArgs(defaults);
    setLatentSnapshot(stableSnapshot(defaults));
    setSavedMessage("Настройки кэша сброшены");
  };

  const handleInsertPreset = (preset) => {
    setConfig((prev) => insertPresetUse(prev, preset.kind, preset.name));
  };

  return (
    <div className="page fit-page settings-page">
      <PageHeader
        eyebrow="Settings"
        title="Settings"
        description="Profiles and config."
      />

      {error ? <div className="alert error">{error}</div> : null}
      {savedMessage ? <div className="alert success">{savedMessage}</div> : null}

      <div className="settings-tabs" role="tablist" aria-label="Settings sections">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            className={activeTab === tab.key ? "active" : ""}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="card">
          <div className="preview-loader" />
        </div>
      ) : null}

      {!loading && activeTab === "config" ? (
        <div className="settings-layout">
          <div className="card train-config-card settings-editor-card">
            <div className="card-header">
              <div>
                <h2 className="card-title">Config Editor</h2>
                <div className="muted">Активная конфигурация проекта и section-scoped preset library.</div>
              </div>
              <span className={configDirty ? "badge dirty" : "badge"}>{configDirty ? "Unsaved" : "Saved"}</span>
            </div>
            <div className="train-editor-layout">
              <div className="train-config-editor">
                <YamlEditor value={config} onChange={setConfig} onSave={handleSaveConfig} />
              </div>
              <PresetLibrary
                presets={presets}
                selectedKind={selectedPresetKind}
                selectedName={selectedPresetName}
                onSelectKind={(kind) => {
                  setSelectedPresetKind(kind);
                  setSelectedPresetName(presets.groups?.[kind]?.[0]?.name || "");
                }}
                onSelectPreset={setSelectedPresetName}
                activeUses={activeUses}
                onInsert={handleInsertPreset}
              />
            </div>
            <div className="settings-actions">
              <button type="button" onClick={handleSaveConfig} disabled={savingConfig || !configDirty}>
                Save config
              </button>
              {savingConfig ? <span className="muted">Saving...</span> : null}
            </div>
          </div>
        </div>
      ) : null}

      {!loading && activeTab === "generate" ? (
        <div className="settings-layout">
          <SettingsSummary title="Активный профиль генерации" items={summarizeSampleArgs(sampleArgs)} />
          <div className="card">
            <div className="card-header">
              <div>
                <h2 className="card-title">Sampling settings</h2>
                <div className="muted">Эти значения используются на странице Generate. Prompt и изображения остаются в рабочей области генерации.</div>
              </div>
              <span className={sampleDirty ? "badge dirty" : "badge"}>{sampleDirty ? "Unsaved" : "Saved"}</span>
            </div>
            <SettingsGroups
              groups={visibleSampleGroups}
              values={sampleArgs}
              onChange={handleSampleChange}
              checkpoints={checkpoints}
            />
            <div className="settings-actions">
              <button type="button" onClick={saveSampleSettings} disabled={!sampleDirty}>
                Save generation settings
              </button>
              <button type="button" className="secondary" onClick={resetSampleSettings}>
                Reset to defaults
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {!loading && activeTab === "latents" ? (
        <div className="settings-layout">
          <SettingsSummary title="Активный профиль кэша" items={summarizeLatentArgs(latentArgs)} />
          <div className="card">
            <div className="card-header">
              <div>
                <h2 className="card-title">Latent cache settings</h2>
                <div className="muted">Эти параметры используются Prepare Latents и Rebuild cache.</div>
              </div>
              <span className={latentDirty ? "badge dirty" : "badge"}>{latentDirty ? "Unsaved" : "Saved"}</span>
            </div>
            <SettingsGroups
              groups={visibleLatentGroups}
              values={latentArgs}
              onChange={handleLatentChange}
              checkpoints={checkpoints}
            />
            <div className="settings-actions">
              <button type="button" onClick={saveLatentSettings} disabled={!latentDirty}>
                Save cache settings
              </button>
              <button type="button" className="secondary" onClick={resetLatentSettings}>
                Reset to defaults
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
