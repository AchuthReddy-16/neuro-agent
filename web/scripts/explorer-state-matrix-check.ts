/**
 * Smoke matrix: explorer tab × modality load must not leak results.
 * Run: npx --yes tsx web/scripts/explorer-state-matrix-check.ts
 */
import assert from "node:assert/strict";
import {
  analysisResultsFromVisualizations,
  emptyAnalysisResults,
  emptyStateMessage,
  emptyVisionState,
  setReadyResult,
  type AnalysisResultsState,
} from "../src/lib/analysis-results.ts";
import {
  applyAnswerToAnalysisResults,
  applyAnswerToVisionState,
  visionStateFromSelectedImage,
} from "../src/lib/merge-analysis-results.ts";
import type { AgentAnswer, Visualization } from "../src/lib/types.ts";

function check(name: string, fn: () => void) {
  try {
    fn();
    console.log(`ok  ${name}`);
  } catch (e) {
    console.error(`FAIL ${name}`);
    throw e;
  }
}

function plotSrcForTab(
  results: AnalysisResultsState,
  tab: keyof AnalysisResultsState,
  selectedImageUrl: string | null,
): string | null {
  // CORRECT contract: never use selectedImageUrl for EEG tabs
  void selectedImageUrl;
  const r = results[tab];
  if (r.status !== "ready" || !r.payload) return null;
  if ("imageUrl" in r.payload && r.payload.imageUrl) return String(r.payload.imageUrl);
  if (tab === "waveform" && r.payload.kind === "live_eeg") return "__live__";
  return null;
}

const imageUrl = "https://example.com/upload.png";
const psdUrl = "https://example.com/psd.png";

check("A. nothing loaded — all idle / empty messages", () => {
  const r = emptyAnalysisResults();
  for (const tab of ["waveform", "psd", "spectrogram", "bandPower", "topomap", "comparison"] as const) {
    assert.equal(r[tab].status, "idle");
    assert.equal(plotSrcForTab(r, tab, null), null);
  }
  const msg = emptyStateMessage("psd", { hasEeg: false, hasImage: false });
  assert.match(msg.body, /Load an EEG/i);
});

check("B. EEG only — waveform ready; image does not appear in PSD", () => {
  let r = emptyAnalysisResults();
  r = {
    ...r,
    waveform: setReadyResult("waveform", { kind: "live_eeg" }, { source: "eeg" }),
  };
  assert.equal(plotSrcForTab(r, "waveform", imageUrl), "__live__");
  assert.equal(plotSrcForTab(r, "psd", imageUrl), null);
  assert.equal(plotSrcForTab(r, "spectrogram", imageUrl), null);
});

check("C. image only — vision state set; EEG tabs stay idle", () => {
  let r = emptyAnalysisResults();
  let v = emptyVisionState();
  v = visionStateFromSelectedImage(v, {
    id: "img1",
    name: "fig.png",
    kind: "figure",
    sizeBytes: 10,
    status: "ready",
    url: imageUrl,
  });
  assert.equal(v.uploadedFigure.status, "ready");
  assert.equal(plotSrcForTab(r, "psd", imageUrl), null);
  assert.equal(plotSrcForTab(r, "waveform", imageUrl), null);
  assert.equal(plotSrcForTab(r, "topomap", imageUrl), null);
});

check("D. EEG + image — coexistence without crossover", () => {
  let r = emptyAnalysisResults();
  r = {
    ...r,
    waveform: setReadyResult("waveform", { kind: "live_eeg" }, {}),
    psd: setReadyResult(
      "psd",
      { kind: "plot_image", imageUrl: psdUrl, title: "PSD" },
      { source: "sample_visualization" },
    ),
  };
  let v = visionStateFromSelectedImage(emptyVisionState(), {
    id: "img1",
    name: "fig.png",
    kind: "figure",
    sizeBytes: 1,
    status: "ready",
    url: imageUrl,
  });
  assert.equal(plotSrcForTab(r, "psd", imageUrl), psdUrl);
  assert.notEqual(plotSrcForTab(r, "psd", imageUrl), imageUrl);
  assert.equal(v.uploadedFigure.payload?.imageUrl, imageUrl);
});

check("Cross-result: vision analyze does not clear PSD", () => {
  let r = emptyAnalysisResults();
  r = {
    ...r,
    psd: setReadyResult("psd", { kind: "plot_image", imageUrl: psdUrl }, {}),
  };
  const answer = {
    id: "a1",
    question: "What does this figure show?",
    answer: "A topomap of beta power.",
    route: "VISION",
    computedEvidence: [],
    visualEvidence: [
      {
        id: "img1",
        label: "figure",
        tab: "figure",
        imageUrl,
        vlm_interpretation: "beta focus",
      },
    ],
    modelInterpretation: "",
    toolsUsed: [],
    verification: { status: "skipped" },
    uncertainty: "",
    timing: {},
    system: {
      textModel: "t",
      visionModel: "v",
      precision: "BF16",
      serving: "s",
      route: "VISION",
    },
    timeline: [],
    isDemo: false,
    routeDetail: { components: ["TEXT", "VISION"], text_only: false },
    selectedImageId: "img1",
  } as AgentAnswer;

  const next = applyAnswerToAnalysisResults(r, answer, {});
  assert.equal(next.psd.status, "ready");
  assert.equal((next.psd.payload as { imageUrl?: string }).imageUrl, psdUrl);

  const vs = applyAnswerToVisionState(emptyVisionState(), answer, {
    id: "img1",
    name: "fig.png",
    kind: "figure",
    sizeBytes: 1,
    status: "ready",
    url: imageUrl,
  });
  assert.equal(vs.interpretation.status, "ready");
});

check("TEXT only does not mutate analysis results", () => {
  let r = emptyAnalysisResults();
  r = {
    ...r,
    psd: setReadyResult("psd", { kind: "plot_image", imageUrl: psdUrl }, {}),
  };
  const answer = {
    id: "a2",
    question: "hey",
    answer: "Hello!",
    route: "TEXT",
    computedEvidence: [],
    visualEvidence: [],
    modelInterpretation: "",
    toolsUsed: [],
    verification: { status: "skipped" },
    uncertainty: "",
    timing: {},
    system: {
      textModel: "t",
      visionModel: "v",
      precision: "BF16",
      serving: "s",
      route: "TEXT",
    },
    timeline: [],
    isDemo: false,
    routeDetail: { components: ["TEXT"], text_only: true },
  } as AgentAnswer;
  const next = applyAnswerToAnalysisResults(r, answer, {});
  assert.equal(next.psd.status, "ready");
  assert.deepEqual(next.psd.payload, r.psd.payload);
});

check("Sample visualizations seed only matching tabs", () => {
  const viz: Visualization[] = [
    {
      id: "v1",
      tab: "psd",
      title: "PSD",
      imageUrl: psdUrl,
      index: 0,
    },
    {
      id: "v2",
      tab: "topomap",
      title: "Topo",
      imageUrl: "https://example.com/topo.png",
      index: 1,
    },
  ];
  const r = analysisResultsFromVisualizations(viz, { sampleId: "S1" });
  assert.equal(r.psd.status, "ready");
  assert.equal(r.topomap.status, "ready");
  assert.equal(r.spectrogram.status, "idle");
  assert.equal(r.comparison.status, "idle");
});

console.log("\nAll explorer state matrix checks passed.");
