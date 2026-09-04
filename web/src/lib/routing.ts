import type { AnalysisRoute, Experiment, ExperimentFile } from "./types";

/**
 * Product routing: decide what a question needs and whether Live mode can proceed
 * without silently attaching built-in demo assets.
 */

export type InputNeed =
  | "none"
  | "image"
  | "dataset"
  | "document"
  | "unsupported_document";

export type RoutingDecision = {
  route: AnalysisRoute;
  need: InputNeed;
  /** User-facing message when required input is missing (Live mode). */
  missingInputMessage?: string;
  needsVision: boolean;
  needsDataset: boolean;
};

const VISION_HINTS =
  /\b(topomap|spectrogram|figure|plot|image|heatmap|visual(?:ly|s|ization)?|look(?:ing)?\s+at|inspect(?:ing)?\s+(?:the\s+)?(?:map|plot|figure|image)|interpret(?:ing)?\s+(?:the\s+)?(?:map|plot|figure|image)|from\s+the\s+(?:plot|figure|image)|in\s+the\s+(?:plot|figure|image)|what\s+does\s+this\s+(?:figure|plot|image|topomap)|show(?:s|ing)?\s+(?:this\s+)?(?:figure|plot|image|topomap))\b/i;

const TEXT_TOOL_HINTS =
  /\b(discriminative|channel\s*rank|which\s+channels|highest\s+beta|classifier|effect\s+size|unusual\s+channels|outlier|correlation|threshold|band[\s-]?power|rms|psd\s*peak|compare\s+(?:the\s+)?(?:two\s+)?conditions?)\b/i;

const DATASET_HINTS =
  /\b(this\s+(?:sample|dataset|csv|eeg|recording|epoch)|analyze\s+(?:this\s+)?(?:csv|eeg|dataset|recording)|for\s+this\s+sample|band[\s-]?power\s+for\s+this|channels?\s+.*\b(?:sample|recording|eeg)\b|upload(?:ed)?\s+(?:csv|eeg|data))\b/i;

const DOCUMENT_HINTS =
  /\b(pdf|document|paper|manuscript|docx?|this\s+(?:pdf|document|paper))\b/i;

const GENERAL_EXPLAIN_HINTS =
  /\b(what\s+is\s+(?:beta|alpha|mu|theta|delta)|define|explain\s+(?:what|how)|meaning\s+of)\b/i;

export function inferNeedsVision(question: string): boolean {
  const q = question.toLowerCase();
  if (TEXT_TOOL_HINTS.test(q) && !VISION_HINTS.test(q)) return false;
  if (VISION_HINTS.test(q)) return true;
  if (/\balpha\b|\bmu\b/.test(q) && /show|strongest|change|suppress/.test(q)) return true;
  return false;
}

export function inferNeedsDataset(question: string): boolean {
  const q = question.toLowerCase();
  if (GENERAL_EXPLAIN_HINTS.test(q) && !DATASET_HINTS.test(q)) return false;
  if (DOCUMENT_HINTS.test(q)) return false;
  // Tool/numeric questions about a sample need dataset context
  if (TEXT_TOOL_HINTS.test(q) || DATASET_HINTS.test(q)) {
    // Pure definitions like "What is beta-band power?" should not require data
    if (/^what is\b/i.test(question.trim()) && !/this sample|this recording|this eeg|this csv/i.test(q)) {
      return false;
    }
    return true;
  }
  return false;
}

export function inferNeedsDocument(question: string): boolean {
  return DOCUMENT_HINTS.test(question);
}

export function inferRoute(question: string): AnalysisRoute {
  return inferNeedsVision(question) ? "VISION" : "TEXT";
}

/** Demo fixture image IDs must never ride into Live API vision requests. */
export function isBuiltInDemoAssetId(id: string | null | undefined): boolean {
  if (!id) return false;
  return (
    (id.startsWith("img-") && id.includes("-demo")) ||
    id.startsWith("viz-") ||
    id === "img-topo-demo" ||
    id === "img-psd-demo" ||
    id === "img-spec-demo"
  );
}

/**
 * Classify what Live mode needs before calling the API.
 * Built-in demo visualizations must NOT count as an attached image in Live mode.
 */
export function classifyLiveInput(
  question: string,
  opts: {
    uploadedImages: ExperimentFile[];
    selectedImageId: string | null;
    hasLinkedSample: boolean;
    hasEegOrMetadataUpload: boolean;
  },
): RoutingDecision {
  const q = question.trim();
  const needsVision = inferNeedsVision(q);
  const needsDocument = inferNeedsDocument(q);
  const needsDataset = inferNeedsDataset(q);

  if (needsDocument) {
    return {
      route: "TEXT",
      need: "unsupported_document",
      needsVision: false,
      needsDataset: false,
      missingInputMessage:
        "PDF/document questions are not supported yet. Attach an EEG sample JSON or figure, or ask a text/tool question.",
    };
  }

  if (needsVision) {
    // Live path: ignore built-in demo fixture IDs even if still present in state.
    const ready = opts.uploadedImages.filter(
      (f) =>
        (f.status === "ready" || f.status === "uploading") &&
        !isBuiltInDemoAssetId(f.id),
    );
    const selected =
      (opts.selectedImageId && ready.find((f) => f.id === opts.selectedImageId)) ||
      (ready.length === 1 ? ready[0] : null);
    if (!selected) {
      return {
        route: "VISION",
        need: "image",
        needsVision: true,
        needsDataset: false,
        missingInputMessage:
          ready.length > 1
            ? "Multiple figures uploaded — select which image to use for vision analysis."
            : "This question needs an image. Attach or select a figure first.",
      };
    }
    return {
      route: "VISION",
      need: "image",
      needsVision: true,
      needsDataset: false,
    };
  }

  if (needsDataset) {
    const hasData =
      opts.hasLinkedSample || opts.hasEegOrMetadataUpload;
    if (!hasData) {
      return {
        route: "TEXT",
        need: "dataset",
        needsVision: false,
        needsDataset: true,
        missingInputMessage:
          "This question needs an EEG sample or dataset. Upload a sample JSON (with sample_id) or load a built-in sample first.",
      };
    }
  }

  return {
    route: "TEXT",
    need: needsDataset ? "dataset" : "none",
    needsVision: false,
    needsDataset,
  };
}

export function resolveSelectedImage(
  images: ExperimentFile[],
  selectedId: string | null,
): { ok: true; image: ExperimentFile | null } | { ok: false; reason: string } {
  const ready = images.filter(
    (f) =>
      (f.status === "ready" || f.status === "uploading") &&
      !isBuiltInDemoAssetId(f.id),
  );
  if (ready.length === 0) {
    return { ok: false, reason: "This question needs an image. Attach or select a figure first." };
  }
  if (selectedId) {
    const found = ready.find((f) => f.id === selectedId);
    if (found) return { ok: true, image: found };
  }
  if (ready.length === 1) {
    return { ok: true, image: ready[0] };
  }
  return {
    ok: false,
    reason: "Multiple figures uploaded — select which image to use for vision analysis.",
  };
}

/**
 * Explicit user-selected/uploaded image only.
 * Never returns built-in demo / offline sample figure IDs in Live mode.
 */
export function explicitLiveImageId(experiment: Experiment | null): string | null {
  if (!experiment || experiment.isDemo) return null;
  const ready = experiment.image_files.filter(
    (f) =>
      (f.status === "ready" || f.status === "uploading") &&
      !isBuiltInDemoAssetId(f.id),
  );
  if (experiment.selected_image_id) {
    const found = ready.find((f) => f.id === experiment.selected_image_id);
    if (found) return found.id;
  }
  if (ready.length === 1) return ready[0].id;
  return null;
}

export function experimentHasDatasetContext(experiment: Experiment | null): boolean {
  if (!experiment) return false;
  if (experiment.metadata?.sampleId) return true;
  // linked sample from live demo / metadata upload
  const anySample =
    (experiment as Experiment & { linkedSampleId?: string }).linkedSampleId ||
    experiment.eeg_files.some((f) => f.status === "ready") ||
    experiment.metadata_files.some((f) => f.status === "ready");
  // mapExperiment sets eeg when sample attached; also check modalities
  if (experiment.modalities?.eeg && experiment.eeg) return true;
  return Boolean(anySample);
}
