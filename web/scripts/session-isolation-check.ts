/**
 * Regression checks for Chat/Workspace surface isolation + linked-sample figure strip.
 * Run: npx --yes tsx web/scripts/session-isolation-check.ts
 */
import assert from "node:assert/strict";
import {
  captureSurfaceSnapshot,
  emptySurfaceSnapshot,
  stripUserFiguresForLinkedSample,
} from "../src/lib/surface-session";
import { emptyAnalysisResults, emptyVisionState } from "../src/lib/analysis-results";
import type { Experiment } from "../src/lib/types";
import { migrateClientStorage, NEURO_STORE_VERSION, NEURO_STORE_VERSION_KEY, OBSOLETE_STORAGE_KEYS } from "../src/lib/storage-migration";

function fakeExp(overrides: Partial<Experiment> = {}): Experiment {
  return {
    id: "exp_workspace_1",
    experiment_id: "exp_workspace_1",
    isDemo: false,
    status: "ready",
    eeg_files: [],
    metadata_files: [],
    image_files: [
      {
        id: "img_oip",
        name: "OIP.jpeg",
        kind: "figure",
        sizeBytes: 12,
        status: "ready",
        url: "blob:http://localhost/oip",
      },
    ],
    selected_image_id: "img_oip",
    analysis_history: [],
    metadata: {
      subject: "S001",
      run: "R01",
      taskType: "MI",
      movementCondition: "left_fist",
      samplingRateHz: 160,
      channels: 64,
    },
    visualizations: [
      {
        id: "viz-topo",
        tab: "topomap",
        title: "Topo",
        imageUrl: "/samples/topo.png",
        index: 0,
      },
    ],
    modalities: { eeg: true, metadata: true, vision: true, text: true },
    ...overrides,
  };
}

function testStripLinkedSample() {
  const withUserFig = fakeExp({
    id: "exp_demo_s001",
    experiment_id: "exp_demo_s001",
  });
  const cleaned = stripUserFiguresForLinkedSample(withUserFig);
  assert.equal(cleaned.image_files.length, 0, "D: no user figures on linked sample");
  assert.equal(cleaned.selected_image_id, null, "D: no selected image");
  assert.equal(cleaned.figure, undefined, "D: no figure asset");
  assert.ok(cleaned.visualizations.length > 0, "D: sample visualizations retained");
  assert.equal(cleaned.modalities.vision, true, "D: vision via visualizations only");
  console.log("PASS D — linked sample strips user figures");
}

function testCaptureIsolation() {
  const workspace = fakeExp();
  const snap = captureSurfaceSnapshot({
    sessionId: workspace.experiment_id,
    experiment: workspace,
    answers: [],
    activeAnswerId: null,
    analysisResults: emptyAnalysisResults(),
    visionState: emptyVisionState(),
    focusedVizId: null,
    activeTab: "topomap",
  });
  // Mutate original — snapshot must not share arrays
  workspace.image_files.push({
    id: "img2",
    name: "other.png",
    kind: "figure",
    sizeBytes: 1,
    status: "ready",
  });
  assert.equal(snap.experiment?.image_files.length, 1, "A: snapshot deep-copied images");
  assert.equal(snap.experiment?.selected_image_id, "img_oip");

  const chatEmpty = emptySurfaceSnapshot();
  assert.equal(chatEmpty.experiment, null);
  assert.equal(chatEmpty.visionState.selectedImageId, null);
  console.log("PASS A — workspace snapshot isolated from later mutations / empty chat");
}

function testStorageMigration() {
  // Minimal localStorage polyfill for node
  const store = new Map<string, string>();
  const ls = {
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    setItem: (k: string, v: string) => {
      store.set(k, v);
    },
    removeItem: (k: string) => {
      store.delete(k);
    },
    key: (i: number) => [...store.keys()][i] ?? null,
    get length() {
      return store.size;
    },
  };
  (globalThis as unknown as { window: { localStorage: typeof ls; sessionStorage: typeof ls } }).window =
    {
      localStorage: ls,
      sessionStorage: ls,
    };

  ls.setItem("neuro-agent.demo", "1");
  ls.setItem("neuro-agent.selectedImage", "OIP.jpeg");
  ls.setItem("neuro-agent.theme", "dark"); // preference-like — not in obsolete list by exact key
  ls.setItem(OBSOLETE_STORAGE_KEYS[0], "x");

  migrateClientStorage();
  assert.equal(ls.getItem("neuro-agent.demo"), null, "E: obsolete demo key cleared");
  assert.equal(ls.getItem("neuro-agent.selectedImage"), null, "E: selectedImage cleared");
  assert.equal(ls.getItem(NEURO_STORE_VERSION_KEY), String(NEURO_STORE_VERSION));

  // Second run should be no-op (version current) even if we re-set an obsolete key
  // (version gate skips) — document that migration is one-shot per version bump
  ls.setItem("neuro-agent.demo", "stale-again");
  migrateClientStorage();
  assert.equal(ls.getItem("neuro-agent.demo"), "stale-again", "E: version gate skips re-sweep");
  // bump would clear — acceptable for this strategy
  console.log("PASS E — storage migration clears obsolete keys once per version");
}

function testDemoRedirectFile() {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const fs = require("node:fs") as typeof import("node:fs");
  const path = require("node:path") as typeof import("node:path");
  const demoPage = fs.readFileSync(
    path.join(__dirname, "../src/app/demo/page.tsx"),
    "utf8",
  );
  assert.match(demoPage, /redirect\(["']\/chat["']\)/, "B: /demo redirects to /chat");
  assert.doesNotMatch(demoPage, /beginDemoSession|InteractiveDemo|exp_demo/, "B: no demo init");
  console.log("PASS B — /demo redirects to /chat only");
}

testCaptureIsolation();
testDemoRedirectFile();
testStripLinkedSample();
testStorageMigration();
console.log("\nAll session-isolation checks passed.");
