/**
 * Attachment selection → analyze imageId mapping (adversarial).
 * Run: npx --yes tsx web/scripts/attachment-selection-check.ts
 */
import assert from "node:assert/strict";
import { mergeUploadResponse } from "../src/lib/experiment-map.ts";
import { explicitLiveImageId, resolveSelectedImage } from "../src/lib/routing.ts";
import type { ApiUploadResponse, Experiment } from "../src/lib/types.ts";

function bareExp(partial: Partial<Experiment> = {}): Experiment {
  return {
    id: "exp_test",
    experiment_id: "exp_test",
    status: "ready",
    isDemo: false,
    eeg_files: [],
    metadata_files: [],
    image_files: [],
    selected_image_id: null,
    analysis_history: [],
    visualizations: [],
    modalities: { eeg: false, metadata: false, vision: false, text: true },
    metadata: {},
    ...partial,
  } as Experiment;
}

function check(name: string, fn: () => void) {
  try {
    fn();
    console.log(`ok  ${name}`);
  } catch (e) {
    console.error(`FAIL ${name}`);
    throw e;
  }
}

function fakeUpload(
  assetId: string,
  name: string,
  prevArtifacts: Array<Record<string, unknown>> = [],
): ApiUploadResponse {
  const art = {
    id: assetId,
    name,
    kind: "figure",
    sizeBytes: 10,
    status: "ready",
  };
  return {
    experimentId: "exp_test",
    assetId,
    uploaded_artifacts: [...prevArtifacts, art],
    detected_input_types: ["vision"],
    available_visualizations: [],
    status: "ready",
  } as ApiUploadResponse;
}

check("single upload remaps pending local id → assetId selection", () => {
  const pending = "img_local_1";
  let exp = bareExp({
    image_files: [
      {
        id: pending,
        name: "a.png",
        kind: "figure",
        sizeBytes: 1,
        status: "uploading",
      },
    ],
    selected_image_id: pending,
  });
  exp = mergeUploadResponse(exp, fakeUpload("asset_aaa", "a.png"), {
    name: "a.png",
    kind: "figure",
    sizeBytes: 1,
    localPendingId: pending,
  });
  assert.equal(exp.selected_image_id, "asset_aaa");
  assert.equal(explicitLiveImageId(exp), "asset_aaa");
  const r = resolveSelectedImage(exp.image_files, exp.selected_image_id);
  assert.equal(r.ok, true);
  if (r.ok) assert.equal(r.image?.id, "asset_aaa");
});

check("second upload keeps valid prior selection; never stale local id", () => {
  let exp = mergeUploadResponse(null, fakeUpload("asset_a", "a.png"), {
    name: "a.png",
    kind: "figure",
    sizeBytes: 1,
    localPendingId: "img_1",
  });
  assert.equal(exp.selected_image_id, "asset_a");
  const arts = exp.image_files.map((f) => ({
    id: f.id,
    name: f.name,
    kind: "figure",
    sizeBytes: 1,
    status: "ready",
  }));
  exp = {
    ...exp,
    image_files: [
      ...exp.image_files,
      { id: "img_2", name: "b.png", kind: "figure", sizeBytes: 1, status: "uploading" },
    ],
    selected_image_id: "asset_a",
  };
  exp = mergeUploadResponse(exp, fakeUpload("asset_b", "b.png", arts), {
    name: "b.png",
    kind: "figure",
    sizeBytes: 1,
    localPendingId: "img_2",
  });
  assert.equal(exp.selected_image_id, "asset_a");
  assert.ok(exp.image_files.some((f) => f.id === "asset_b"));
  assert.equal(explicitLiveImageId(exp), "asset_a");
});

check("multiple images with null selection → resolve fails (NEEDS_INPUT)", () => {
  const images = [
    { id: "asset_a", name: "a.png", kind: "figure" as const, sizeBytes: 1, status: "ready" as const },
    { id: "asset_b", name: "b.png", kind: "figure" as const, sizeBytes: 1, status: "ready" as const },
  ];
  const r = resolveSelectedImage(images, null);
  assert.equal(r.ok, false);
  if (!r.ok) assert.match(r.reason, /select/i);
  assert.equal(explicitLiveImageId(bareExp({ image_files: images, selected_image_id: null })), null);
});

console.log("All attachment selection checks passed.");
