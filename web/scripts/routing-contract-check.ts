/**
 * Product routing / attachment isolation contract.
 * Run: npx --yes tsx web/scripts/routing-contract-check.ts
 */
import assert from "node:assert/strict";
import {
  classifyLiveInput,
  explicitLiveImageId,
  inferNeedsDataset,
  inferNeedsVision,
  inferRoute,
  isBuiltInDemoAssetId,
} from "../src/lib/routing.ts";

type FileStub = {
  id: string;
  name: string;
  kind: "figure";
  sizeBytes: number;
  status: "ready" | "uploading" | "error";
};

function img(id: string, status: FileStub["status"] = "ready"): FileStub {
  return { id, name: `${id}.png`, kind: "figure", sizeBytes: 10, status };
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

check("1. general text, no attachment → TEXT", () => {
  const d = classifyLiveInput("What is beta-band power?", {
    uploadedImages: [],
    selectedImageId: null,
    hasLinkedSample: false,
    hasEegOrMetadataUpload: false,
  });
  assert.equal(d.route, "TEXT");
  assert.equal(d.needsVision, false);
  assert.equal(d.needsDataset, false);
  assert.equal(d.missingInputMessage, undefined);
  assert.equal(inferRoute("What is beta-band power?"), "TEXT");
});

check("2. vision question, no image → ask to attach", () => {
  const d = classifyLiveInput("What does this topomap figure show?", {
    uploadedImages: [],
    selectedImageId: null,
    hasLinkedSample: true,
    hasEegOrMetadataUpload: true,
  });
  assert.equal(d.route, "VISION");
  assert.equal(d.needsVision, true);
  assert.match(d.missingInputMessage || "", /attach|select/i);
});

check("3. vision question with explicit image → VISION ready", () => {
  const d = classifyLiveInput("Interpret this figure", {
    uploadedImages: [img("asset_uploaded_1")],
    selectedImageId: "asset_uploaded_1",
    hasLinkedSample: false,
    hasEegOrMetadataUpload: false,
  });
  assert.equal(d.route, "VISION");
  assert.equal(d.needsVision, true);
  assert.equal(d.missingInputMessage, undefined);
});

check("4. dataset question, no dataset → ask for sample", () => {
  const d = classifyLiveInput(
    "Which five EEG channels have the highest beta-band power for this sample?",
    {
      uploadedImages: [],
      selectedImageId: null,
      hasLinkedSample: false,
      hasEegOrMetadataUpload: false,
    },
  );
  assert.equal(d.route, "TEXT");
  assert.equal(d.needsDataset, true);
  assert.match(d.missingInputMessage || "", /sample|dataset/i);
});

check("5. dataset question with selected sample → proceed", () => {
  const d = classifyLiveInput(
    "Which five EEG channels have the highest beta-band power for this sample?",
    {
      uploadedImages: [],
      selectedImageId: null,
      hasLinkedSample: true,
      hasEegOrMetadataUpload: false,
    },
  );
  assert.equal(d.route, "TEXT");
  assert.equal(d.needsVision, false);
  assert.equal(d.needsDataset, true);
  assert.equal(d.missingInputMessage, undefined);
});

check("6. ranking stays TEXT (not vision)", () => {
  assert.equal(inferNeedsVision("Which channels have the highest beta power?"), false);
  assert.equal(inferNeedsDataset("Which channels have the highest beta power?"), true);
});

check("7. stale demo image ID is not a live attachment", () => {
  assert.equal(isBuiltInDemoAssetId("img-topo-demo"), true);
  assert.equal(isBuiltInDemoAssetId("viz-topomap-01"), true);
  const d = classifyLiveInput("What does this figure show?", {
    uploadedImages: [img("img-topo-demo"), img("viz-topomap-01")],
    selectedImageId: "img-topo-demo",
    hasLinkedSample: true,
    hasEegOrMetadataUpload: true,
  });
  assert.match(d.missingInputMessage || "", /attach|select/i);

  const demoExp = {
    isDemo: true,
    image_files: [img("img-topo-demo")],
    selected_image_id: "img-topo-demo",
    eeg_files: [],
    metadata_files: [],
  };
  assert.equal(explicitLiveImageId(demoExp as never), null);

  const liveExp = {
    isDemo: false,
    image_files: [img("img-topo-demo"), img("asset_real")],
    selected_image_id: "img-topo-demo",
    eeg_files: [],
    metadata_files: [],
  };
  assert.equal(explicitLiveImageId(liveExp as never), "asset_real");
});

check("8. document/PDF question blocked", () => {
  const d = classifyLiveInput("Summarize this PDF manuscript", {
    uploadedImages: [],
    selectedImageId: null,
    hasLinkedSample: false,
    hasEegOrMetadataUpload: false,
  });
  assert.equal(d.need, "unsupported_document");
  assert.match(d.missingInputMessage || "", /not supported/i);
});

check("9. tool-backed ranking question does not invent vision", () => {
  assert.equal(
    inferNeedsVision(
      "Which five EEG channels have the highest beta-band power for this sample?",
    ),
    false,
  );
});

check("10. offline fixture labeling is separate (route helpers unchanged)", () => {
  assert.equal(inferNeedsVision("Look at this topomap"), true);
});

console.log("\nAll routing contract checks passed.");
