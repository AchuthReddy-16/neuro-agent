/**
 * Smoke matrix: explorer tab × modality load must not leak results.
 * Covers scenarios A–F from missing-input / stale-result UX fix.
 * Run: npx --yes tsx web/scripts/explorer-state-matrix-check.ts
 */
import assert from "node:assert/strict";
import {
  analysisResultsFromVisualizations,
  emptyAnalysisResults,
  emptyStateMessage,
  emptyVisionState,
  resetEegDerivedResults,
  setReadyResult,
  visionEmptyStateMessage,
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

/** Production display source: never treat live_eeg or selected image as a plot. */
function plotSrcForTab(
  results: AnalysisResultsState,
  tab: keyof AnalysisResultsState,
  selectedImageUrl: string | null,
): string | null {
  void selectedImageUrl;
  const r = results[tab];
  if (r.status !== "ready" || !r.payload) return null;
  if (tab === "waveform") {
    const p = r.payload as { kind?: string; imageUrl?: string };
    if (p.kind === "static_plot" && p.imageUrl) return String(p.imageUrl);
    // live_eeg without static plot must not render as a result in production
    return null;
  }
  if ("imageUrl" in r.payload && r.payload.imageUrl) return String(r.payload.imageUrl);
  return null;
}

function assertAllEegIdle(r: AnalysisResultsState, label: string) {
  for (const tab of ["waveform", "psd", "spectrogram", "bandPower", "topomap", "comparison"] as const) {
    assert.equal(r[tab].status, "idle", `${label}: ${tab} should be idle`);
    assert.equal(plotSrcForTab(r, tab, "https://leak.example/x.png"), null);
  }
}

const imageUrl = "https://example.com/upload.png";
const psdUrl = "https://example.com/psd.png";
const waveUrl = "https://example.com/wave.png";
const topoUrl = "https://example.com/topo.png";

check("A. nothing loaded — all idle / empty messages", () => {
  const r = emptyAnalysisResults();
  assertAllEegIdle(r, "NONE");
  assert.match(emptyStateMessage("waveform", { hasEeg: false, hasImage: false }).body, /Load or select an EEG/i);
  assert.match(emptyStateMessage("psd", { hasEeg: false, hasImage: false }).body, /Load an EEG/i);
  assert.match(emptyStateMessage("spectrogram", { hasEeg: false, hasImage: false }).body, /spectrogram/i);
  assert.match(emptyStateMessage("band_power", { hasEeg: false, hasImage: false }).body, /band-power/i);
  assert.match(emptyStateMessage("topomap", { hasEeg: false, hasImage: false }).body, /topomap/i);
  assert.match(emptyStateMessage("comparison", { hasEeg: false, hasImage: false }).body, /Load EEG data/i);
  assert.match(visionEmptyStateMessage({ hasImage: false }).body, /Upload or select an image/i);
});

check("A2. image only — vision ready; EEG tabs empty; no simulated waveform", () => {
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
  assert.equal(v.uploadedFigure.payload?.imageUrl, imageUrl);
  assertAllEegIdle(r, "IMAGE");
  assert.match(
    emptyStateMessage("topomap", { hasEeg: false, hasImage: true }).body,
    /visual interpretation|Load an EEG/i,
  );
  assert.match(emptyStateMessage("waveform", { hasEeg: false, hasImage: true }).body, /Load or select an EEG/i);
  // Production never shows live_eeg as plotSrc
  r = {
    ...r,
    waveform: setReadyResult("waveform", { kind: "live_eeg" }, { source: "should_not_render" }),
  };
  assert.equal(plotSrcForTab(r, "waveform", imageUrl), null);
});

check("B. EEG only — compute prompts when idle; no vision without image", () => {
  const r = emptyAnalysisResults();
  assert.equal(r.psd.status, "idle");
  assert.match(emptyStateMessage("psd", { hasEeg: true, hasImage: false }).body, /Run PSD/i);
  assert.match(emptyStateMessage("waveform", { hasEeg: true, hasImage: false }).body, /Generate the waveform/i);
  assert.match(visionEmptyStateMessage({ hasImage: false }).body, /Upload or select an image/i);
  assert.equal(plotSrcForTab(r, "psd", null), null);
});

check("B2. EEG with sample visualizations — only matching tabs ready", () => {
  const viz: Visualization[] = [
    { id: "v1", tab: "psd", title: "PSD", imageUrl: psdUrl, index: 0 },
    { id: "v2", tab: "topomap", title: "Topo", imageUrl: topoUrl, index: 1 },
    { id: "v3", tab: "waveform", title: "Wave", imageUrl: waveUrl, index: 2 },
  ];
  const r = analysisResultsFromVisualizations(viz, { sampleId: "S1" });
  assert.equal(plotSrcForTab(r, "psd", imageUrl), psdUrl);
  assert.equal(plotSrcForTab(r, "topomap", imageUrl), topoUrl);
  assert.equal(plotSrcForTab(r, "waveform", imageUrl), waveUrl);
  assert.equal(r.spectrogram.status, "idle");
  assert.equal(r.comparison.status, "idle");
  assert.notEqual(plotSrcForTab(r, "psd", imageUrl), imageUrl);
});

check("C. EEG + image — coexistence without crossover", () => {
  let r = emptyAnalysisResults();
  r = {
    ...r,
    waveform: setReadyResult(
      "waveform",
      { kind: "static_plot", imageUrl: waveUrl },
      { source: "sample_visualization" },
    ),
    psd: setReadyResult(
      "psd",
      { kind: "plot_image", imageUrl: psdUrl, title: "PSD" },
      { source: "sample_visualization" },
    ),
  };
  const v = visionStateFromSelectedImage(emptyVisionState(), {
    id: "img1",
    name: "fig.png",
    kind: "figure",
    sizeBytes: 1,
    status: "ready",
    url: imageUrl,
  });
  assert.equal(plotSrcForTab(r, "psd", imageUrl), psdUrl);
  assert.notEqual(plotSrcForTab(r, "psd", imageUrl), imageUrl);
  assert.equal(plotSrcForTab(r, "waveform", imageUrl), waveUrl);
  assert.equal(v.uploadedFigure.payload?.imageUrl, imageUrl);
});

check("D. EEG results → clear EEG → all EEG slots idle (no stale plots)", () => {
  let r = analysisResultsFromVisualizations(
    [
      { id: "v1", tab: "psd", title: "PSD", imageUrl: psdUrl, index: 0 },
      { id: "v2", tab: "topomap", title: "Topo", imageUrl: topoUrl, index: 1 },
      { id: "v3", tab: "waveform", title: "Wave", imageUrl: waveUrl, index: 2 },
    ],
    { sampleId: "S1" },
  );
  assert.equal(r.psd.status, "ready");
  r = resetEegDerivedResults(r);
  assertAllEegIdle(r, "EEG_CLEARED");
});

check("E. experiment A results do not transfer as B without reseed", () => {
  const a = analysisResultsFromVisualizations(
    [{ id: "a1", tab: "psd", title: "A", imageUrl: psdUrl, index: 0 }],
    { experimentId: "exp_A", sampleId: "A" },
  );
  // Switching experiment = replace with empty (or B's own seed), never merge A into B
  const b = emptyAnalysisResults();
  assert.equal(a.psd.status, "ready");
  assertAllEegIdle(b, "EXP_B");
  assert.notEqual(
    (a.psd.payload as { imageUrl?: string })?.imageUrl,
    plotSrcForTab(b, "psd", null),
  );
});

check("F. image A → new image-only experiment — no old EEG/sample results", () => {
  let r = analysisResultsFromVisualizations(
    [{ id: "old", tab: "topomap", title: "Old", imageUrl: topoUrl, index: 0 }],
    { sampleId: "old" },
  );
  assert.equal(r.topomap.status, "ready");
  // New image-only session
  r = resetEegDerivedResults(r);
  const v = visionStateFromSelectedImage(emptyVisionState(), {
    id: "img_new",
    name: "new.png",
    kind: "figure",
    sizeBytes: 1,
    status: "ready",
    url: imageUrl,
  });
  assertAllEegIdle(r, "NEW_IMAGE_ONLY");
  assert.equal(v.uploadedFigure.payload?.imageUrl, imageUrl);
});

check("Matrix NONE / IMAGE / EEG / EEG+IMAGE — zero wrong-result leakage", () => {
  const none = emptyAnalysisResults();
  const imageOnly = emptyAnalysisResults();
  const eegIdle = emptyAnalysisResults();
  const eegPlus = {
    ...emptyAnalysisResults(),
    psd: setReadyResult("psd", { kind: "plot_image", imageUrl: psdUrl }, {}),
  };
  const selected = imageUrl;

  for (const tab of ["waveform", "psd", "spectrogram", "bandPower", "topomap", "comparison"] as const) {
    assert.equal(plotSrcForTab(none, tab, selected), null, `NONE ${tab}`);
    assert.equal(plotSrcForTab(imageOnly, tab, selected), null, `IMAGE ${tab}`);
    assert.equal(plotSrcForTab(eegIdle, tab, selected), null, `EEG-idle ${tab}`);
  }
  assert.equal(plotSrcForTab(eegPlus, "psd", selected), psdUrl);
  assert.notEqual(plotSrcForTab(eegPlus, "psd", selected), selected);
  assert.equal(plotSrcForTab(eegPlus, "topomap", selected), null);
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

check("Comparison empty copy — one condition", () => {
  const msg = emptyStateMessage("comparison", {
    hasEeg: true,
    hasImage: false,
    hasComparableConditions: false,
  });
  assert.match(msg.body, /at least two comparable conditions/i);
});

console.log("\nAll explorer state matrix checks passed.");
