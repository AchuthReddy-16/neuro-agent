/**
 * Merge agent answers into typed analysis / vision result slots.
 * Never overwrites unrelated slots; never writes uploaded figures into EEG tabs.
 */

import type { AgentAnswer, ExperimentFile } from "./types";
import {
  emptyVisionState,
  setReadyResult,
  type AnalysisResultsState,
  type VisionState,
} from "./analysis-results";

export function applyAnswerToAnalysisResults(
  prev: AnalysisResultsState,
  answer: AgentAnswer,
  opts: { experimentId?: string | null; sampleId?: string | null },
): AnalysisResultsState {
  const next = { ...prev };
  const nowProv = {
    experimentId: opts.experimentId,
    sampleId: opts.sampleId,
    source: "agent_analyze",
    tool: answer.toolsUsed[0] ?? null,
    generatedAt: new Date().toISOString(),
  };

  // TEXT-only / conversational — do not mutate analysis slots
  const components = answer.routeDetail?.components ?? [];
  const textOnly =
    answer.routeDetail?.text_only === true ||
    (components.includes("TEXT") &&
      !components.includes("TOOLS") &&
      !components.includes("VISION") &&
      !answer.toolsUsed.length);

  if (textOnly && answer.route !== "VISION") {
    return prev;
  }

  // Band power / ranking from computed evidence
  if (answer.toolsUsed.some((t) => /rank|band_power|band-power/i.test(t)) ||
      answer.computedEvidence.some((e) => /rank|beta|band/i.test(e.label))) {
    const rankingTools = answer.toolsUsed.some((t) => /rank|band/i.test(t));
    if (rankingTools || answer.computedEvidence.length > 0) {
      next.bandPower = setReadyResult(
        "band_power",
        {
          kind: "band_power_table",
          rows: answer.computedEvidence,
        },
        {
          ...nowProv,
          metric: "band_power",
        },
      );
    }
  }

  // Comparison
  if (
    answer.toolsUsed.some((t) => /compar/i.test(t)) ||
    answer.computedEvidence.some((e) => /vs|compar/i.test(e.label))
  ) {
    const row = answer.computedEvidence.find((e) => /vs|compar/i.test(e.label));
    next.comparison = setReadyResult(
      "comparison",
      {
        kind: "comparison",
        summary: answer.answer.slice(0, 400),
        rows: answer.computedEvidence,
        conditionA: row?.label?.split(" vs ")[0],
        conditionB: row?.label?.split(" vs ")[1],
      },
      {
        ...nowProv,
        conditionA: row?.label?.split(" vs ")[0] ?? null,
        conditionB: row?.label?.split(" vs ")[1] ?? null,
        sampleIdA: opts.sampleId,
        sampleIdB: opts.sampleId,
        metric: "condition_comparison",
      },
    );
  }

  // Typed visual evidence → only matching tab slots (never as generic figure)
  for (const ve of answer.visualEvidence ?? []) {
    const tab = String(ve.tab || "").toLowerCase();
    const url = ve.imageUrl;
    if (!url) continue;
    const base = {
      kind: "plot_image" as const,
      imageUrl: url,
      title: ve.label,
      visualizationId: ve.id,
    };
    const prov = {
      ...nowProv,
      visualizationId: ve.id,
      imageId: ve.id,
      source: "visual_evidence",
    };

    if (tab === "psd") {
      next.psd = setReadyResult("psd", base, prov);
    } else if (tab === "spectrogram") {
      next.spectrogram = setReadyResult("spectrogram", base, prov);
    } else if (tab === "topomap") {
      next.topomap = setReadyResult("topomap", base, prov);
    } else if (tab === "waveform") {
      next.waveform = setReadyResult(
        "waveform",
        { kind: "static_plot", imageUrl: url },
        prov,
      );
    } else if (tab === "band_power") {
      next.bandPower = setReadyResult(
        "band_power",
        {
          kind: "band_power_table",
          rows: answer.computedEvidence,
          imageUrl: url,
        },
        prov,
      );
    } else if (tab === "comparison") {
      next.comparison = setReadyResult(
        "comparison",
        {
          kind: "comparison",
          summary: answer.answer.slice(0, 400),
          imageUrl: url,
          rows: answer.computedEvidence,
        },
        prov,
      );
    }
    // Unknown / figure tabs: do not write into EEG analysis slots
  }

  return next;
}

export function applyAnswerToVisionState(
  prev: VisionState,
  answer: AgentAnswer,
  selected: ExperimentFile | null,
): VisionState {
  if (answer.route !== "VISION") {
    return prev;
  }
  const ve = answer.visualEvidence?.[0];
  const imageId = answer.selectedImageId || selected?.id || ve?.id || prev.selectedImageId;
  const imageUrl = selected?.url || ve?.imageUrl || null;
  const interpretation =
    ve?.vlm_interpretation || ve?.observation || answer.modelInterpretation || answer.answer;

  return {
    ...prev,
    selectedImageId: imageId,
    interpretation: setReadyResult(
      "vision_interpretation",
      {
        kind: "vision",
        imageId: imageId || "unknown",
        imageUrl,
        imageName: answer.selectedImageName || selected?.name || null,
        interpretation,
      },
      {
        imageId,
        source: "vision_analyze",
        generatedAt: new Date().toISOString(),
      },
    ),
  };
}

export function visionStateFromSelectedImage(
  prev: VisionState,
  file: ExperimentFile | null,
): VisionState {
  if (!file) {
    return {
      ...prev,
      selectedImageId: null,
      // Keep interpretation if any; clear uploaded figure marker
      uploadedFigure: emptyVisionState().uploadedFigure,
    };
  }
  return {
    ...prev,
    selectedImageId: file.id,
    uploadedFigure: setReadyResult(
      "uploaded_figure",
      {
        kind: "vision",
        imageId: file.id,
        imageUrl: file.url ?? null,
        imageName: file.name,
        interpretation: null,
      },
      {
        imageId: file.id,
        source: "user_upload",
        generatedAt: new Date().toISOString(),
      },
    ),
  };
}
