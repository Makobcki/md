import { parseRunDate } from "./formatters.js";

export const workflowCards = [
  {
    key: "generate",
    title: "Prompt workspace",
    description: "A chat-style console for txt2img, img2img, inpaint, control and VAR token flows.",
    to: "/generate",
    action: "Open generator",
  },
  {
    key: "train",
    title: "Training studio",
    description: "Edit active KDL config, apply presets, watch metrics and inspect generated samples.",
    to: "/train",
    action: "Open training",
  },
  {
    key: "latents",
    title: "Cache builder",
    description: "Prepare or rebuild latent cache shards with GPU, throughput and queue controls.",
    to: "/latents",
    action: "Prepare cache",
  },
  {
    key: "logs",
    title: "Runs intelligence",
    description: "Search stdout, stderr, artifacts, configs and metrics from every recorded run.",
    to: "/files",
    action: "Inspect runs",
  },
];

function sameDay(left, right) {
  if (!left || !right) return false;
  return (
    left.getFullYear() === right.getFullYear() &&
    left.getMonth() === right.getMonth() &&
    left.getDate() === right.getDate()
  );
}

export function sortAndFilterRuns(runs = [], filters = {}) {
  const {
    failedOnly = false,
    todayOnly = false,
    sortDir = "desc",
    now = new Date(),
  } = filters;

  return [...runs]
    .filter((run) => (failedOnly ? run.status === "failed" : true))
    .filter((run) => (todayOnly ? sameDay(parseRunDate(run), now) : true))
    .sort((a, b) => {
      const left = parseRunDate(a)?.getTime() || 0;
      const right = parseRunDate(b)?.getTime() || 0;
      return sortDir === "asc" ? left - right : right - left;
    });
}

export function buildRunInventory(runs = []) {
  return runs.reduce(
    (acc, run) => {
      acc.total += 1;
      if (run.status === "running") acc.running += 1;
      if (run.status === "failed") acc.failed += 1;
      if (run.run_type === "train") acc.training += 1;
      if (run.run_type === "sample") acc.generation += 1;
      if (run.run_type === "latent_cache") acc.latentCache += 1;
      return acc;
    },
    {
      total: 0,
      running: 0,
      failed: 0,
      training: 0,
      generation: 0,
      latentCache: 0,
    }
  );
}

export function recentPreviewItems(items = [], limit = 8) {
  return [...items]
    .filter((item) => item?.previewable !== false)
    .sort((a, b) => (b?.mtime || 0) - (a?.mtime || 0))
    .slice(0, limit);
}

export function extractConfigUses(content = "") {
  const used = {};
  const stack = [];
  for (const line of String(content).split("\n")) {
    const inline = line.match(/^\s*([A-Za-z0-9_-]+)\s*\{\s*use\s+"([^"]+)"\s*\}/);
    if (inline) {
      const [, section, name] = inline;
      used[section] = [...(used[section] || []), name];
      continue;
    }

    const sectionStart = line.match(/^\s*([A-Za-z0-9_-]+)(?:\s+[^{}]*)?\s*\{/);
    if (sectionStart) {
      stack.push(sectionStart[1]);
    }

    const use = line.match(/^\s*use\s+"([^"]+)"/);
    if (use && stack[0] === "config" && stack.length >= 2) {
      const section = stack[1];
      used[section] = [...(used[section] || []), use[1]];
    }

    const closeCount = (line.match(/\}/g) || []).length;
    for (let index = 0; index < closeCount; index += 1) {
      stack.pop();
    }
  }
  return used;
}
