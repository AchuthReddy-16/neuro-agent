/**
 * Vision selection invalidation + provenance mapping checks.
 * Run: npx --yes tsx web/scripts/vision-image-provenance-check.ts
 */
import assert from "node:assert/strict";
import { emptyVisionState, setReadyResult } from "../src/lib/analysis-results.ts";
import { visionStateFromSelectedImage } from "../src/lib/merge-analysis-results.ts";
import { analyzeResponseToAgentAnswer } from "../src/lib/mock/responses.ts";
import type { AnalyzeResponse, ExperimentFile } from "../src/lib/types.ts";

const fileA: ExperimentFile = {
  id: "asset_a",
  name: "Vertex_waves_EEG.png",
  kind: "figure",
  sizeBytes: 10,
  status: "ready",
  url: "https://example.com/a.png",
};
const fileB: ExperimentFile = {
  id: "asset_b",
  name: "scatter_B.png",
  kind: "figure",
  sizeBytes: 10,
  status: "ready",
  url: "https://example.com/b.png",
};

function check(name: string, fn: () => void) {
  try {
    fn();
    console.log(`ok  ${name}`);
  } catch (e) {
    console.error(`FAIL ${name}`);
    throw e;
  }
}

check("switching A→B invalidates interpretation", () => {
  let v = emptyVisionState();
  v = visionStateFromSelectedImage(v, fileA);
  v = {
    ...v,
    interpretation: setReadyResult(
      "vision_interpretation",
      {
        kind: "vision",
        imageId: fileA.id,
        imageName: fileA.name,
        interpretation: "waveform of vertex waves",
      },
      { imageId: fileA.id },
    ),
  };
  assert.equal(v.interpretation.status, "ready");
  v = visionStateFromSelectedImage(v, fileB);
  assert.equal(v.selectedImageId, fileB.id);
  assert.equal(v.interpretation.status, "idle");
  assert.equal(v.uploadedFigure.payload?.imageName, fileB.name);
});

check("clearing selection clears interpretation", () => {
  let v = visionStateFromSelectedImage(emptyVisionState(), fileA);
  v = {
    ...v,
    interpretation: setReadyResult(
      "vision_interpretation",
      { kind: "vision", imageId: fileA.id, interpretation: "x" },
      {},
    ),
  };
  v = visionStateFromSelectedImage(v, null);
  assert.equal(v.selectedImageId, null);
  assert.equal(v.interpretation.status, "idle");
});

check("analyzeResponse maps source image provenance", () => {
  const res = {
    answer: "Vertex waves visible.",
    route: "VISION",
    computed_evidence: [],
    visual_evidence: [
      {
        id: "asset_a",
        label: "Vertex_waves_EEG.png",
        tab: "figure",
        imageUrl: "/api/visualization/asset_a",
      },
    ],
    model_interpretation: "",
    tools_used: [],
    verification: { status: "skipped" },
    uncertainty: "",
    timing: {},
    system: {
      textModel: "t",
      visionModel: "v",
      precision: "BF16",
      serving: "local",
      route: "VISION",
    },
    sourceImageId: "asset_a",
    sourceImageName: "Vertex_waves_EEG.png",
    visionUsed: true,
    visionAssetOrigin: "uploaded",
  } as AnalyzeResponse;
  const ans = analyzeResponseToAgentAnswer(res, { question: "What does this show?" });
  assert.equal(ans.selectedImageId, "asset_a");
  assert.equal(ans.selectedImageName, "Vertex_waves_EEG.png");
  assert.equal(ans.visionUsed, true);
});

console.log("All vision image provenance checks passed.");
