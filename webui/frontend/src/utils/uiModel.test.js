import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRunInventory,
  extractConfigUses,
  recentPreviewItems,
  sortAndFilterRuns,
  workflowCards,
} from "./uiModel.js";

const runs = [
  {
    run_id: "20260507_120000_a1",
    run_type: "sample",
    status: "done",
    created_at: "2026-05-07T12:00:00Z",
  },
  {
    run_id: "20260507_121000_b2",
    run_type: "train",
    status: "failed",
    created_at: "2026-05-07T12:10:00Z",
  },
  {
    run_id: "20260506_110000_c3",
    run_type: "latent_cache",
    status: "running",
    created_at: "2026-05-06T11:00:00Z",
  },
];

test("sortAndFilterRuns filters failed runs and keeps newest first", () => {
  const visible = sortAndFilterRuns(runs, {
    failedOnly: true,
    todayOnly: false,
    sortDir: "desc",
    now: new Date("2026-05-07T13:00:00Z"),
  });

  assert.deepEqual(
    visible.map((run) => run.run_id),
    ["20260507_121000_b2"]
  );
});

test("buildRunInventory counts current project workflow types", () => {
  assert.deepEqual(buildRunInventory(runs), {
    total: 3,
    running: 1,
    failed: 1,
    training: 1,
    generation: 1,
    latentCache: 1,
  });
});

test("recentPreviewItems returns newest previewable items with stable limit", () => {
  const previews = recentPreviewItems(
    [
      { path: "old.png", mtime: 1, previewable: true },
      { path: "skip.txt", mtime: 3, previewable: false },
      { path: "new.png", mtime: 4, previewable: true },
    ],
    1
  );

  assert.deepEqual(previews.map((item) => item.path), ["new.png"]);
});

test("workflowCards exposes primary md-diffusion workflows", () => {
  assert.deepEqual(
    workflowCards.map((card) => card.key),
    ["generate", "train", "latents", "logs"]
  );
});

test("extractConfigUses detects section scoped KDL presets", () => {
  const used = extractConfigUses(`
    config target="train" version=2 {
      model { use "mmdit_576" }
      training {
        use "bf16_adamw"
        use "single_gpu_debug"
      }
    }
  `);

  assert.deepEqual(used, {
    model: ["mmdit_576"],
    training: ["bf16_adamw", "single_gpu_debug"],
  });
});
