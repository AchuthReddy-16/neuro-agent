import type { AnalysisRoute, ExperimentFile } from "./types";

/**
 * Infer whether a question needs vision (figure interpretation).
 * Default is TEXT + TOOLS — do not require an image for every analysis.
 */
export function inferNeedsVision(question: string): boolean {
  const q = question.toLowerCase();

  const visionHints =
    /topomap|spectrogram|figure|plot|image|visual|look at|inspect (the )?(map|plot|figure)|interpret (the )?(map|plot|figure)|from the (plot|figure|image)|in the (plot|figure|image)|show (the )?(topomap|psd|spectrogram|band power)|waveform image|psd plot/;

  const textOnlyHints =
    /discriminative|channel rank|which channels|highest beta|classifier|effect size|unusual channels|outlier|correlation|threshold|band power analysis(?! plot)/;

  if (visionHints.test(q)) return true;
  if (textOnlyHints.test(q)) return false;
  // Alpha/mu demo path uses spectrogram evidence
  if (/\balpha\b|\bmu\b/.test(q) && /show|strongest|change|suppress/.test(q)) return true;
  return false;
}

export function inferRoute(question: string): AnalysisRoute {
  return inferNeedsVision(question) ? "VISION" : "TEXT";
}

export function resolveSelectedImage(
  images: ExperimentFile[],
  selectedId: string | null,
): { ok: true; image: ExperimentFile | null } | { ok: false; reason: string } {
  const ready = images.filter((f) => f.status === "ready" || f.status === "uploading");
  if (ready.length === 0) {
    return { ok: false, reason: "Vision analysis needs a figure. Upload an image first." };
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
