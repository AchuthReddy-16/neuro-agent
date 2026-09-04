/**
 * Typed multimodal analysis / vision result contracts.
 * Explorer tabs consume ONLY their matching result type — never a generic
 * "current visualization" or selected uploaded figure fallback.
 */

import type { ComputedEvidenceItem, Visualization, VisualizationTab } from "./types";

export type ResultStatus = "idle" | "loading" | "ready" | "error";

export type AnalysisResultType =
  | "waveform"
  | "spectrogram"
  | "psd"
  | "band_power"
  | "topomap"
  | "comparison"
  | "vision_interpretation"
  | "uploaded_figure"
  | "generated_visualization";

export interface ResultProvenance {
  experimentId?: string | null;
  sampleId?: string | null;
  /** Comparison / multi-source */
  sampleIdA?: string | null;
  sampleIdB?: string | null;
  conditionA?: string | null;
  conditionB?: string | null;
  channel?: string | null;
  channels?: string[] | null;
  band?: string | null;
  metric?: string | null;
  source?: string | null;
  tool?: string | null;
  generatedAt?: string | null;
  visualizationId?: string | null;
  imageId?: string | null;
  note?: string | null;
}

export interface TypedAnalysisResult<TPayload = unknown> {
  type: AnalysisResultType;
  status: ResultStatus;
  payload: TPayload | null;
  provenance: ResultProvenance;
  error: string | null;
  updatedAt: string | null;
}

export interface WaveformPayload {
  kind: "live_eeg" | "static_plot";
  imageUrl?: string | null;
  channelLabels?: string[];
  samplingRateHz?: number;
}

export interface ImagePlotPayload {
  kind: "plot_image";
  imageUrl: string;
  title?: string;
  visualizationId?: string;
  channel?: string;
  band?: string;
  condition?: string;
  compareWith?: string;
}

export interface BandPowerPayload {
  kind: "band_power_table";
  rows: ComputedEvidenceItem[];
  ranking?: string[];
  values?: Record<string, number | string>;
  units?: string | null;
  imageUrl?: string | null;
}

export interface ComparisonPayload {
  kind: "comparison";
  summary?: string;
  rows?: ComputedEvidenceItem[];
  imageUrl?: string | null;
  conditionA?: string;
  conditionB?: string;
  winner?: string;
  valueA?: number | string | null;
  valueB?: number | string | null;
}

export interface VisionPayload {
  kind: "vision";
  imageId: string;
  imageUrl?: string | null;
  imageName?: string | null;
  interpretation?: string | null;
}

/** Per-tab EEG / analysis slots — independent of uploaded figures. */
export interface AnalysisResultsState {
  waveform: TypedAnalysisResult<WaveformPayload>;
  spectrogram: TypedAnalysisResult<ImagePlotPayload>;
  psd: TypedAnalysisResult<ImagePlotPayload>;
  bandPower: TypedAnalysisResult<BandPowerPayload>;
  topomap: TypedAnalysisResult<ImagePlotPayload>;
  comparison: TypedAnalysisResult<ComparisonPayload>;
}

/** Vision / figure path — never feeds waveform/PSD/spectrogram/etc. */
export interface VisionState {
  selectedImageId: string | null;
  uploadedFigure: TypedAnalysisResult<VisionPayload>;
  interpretation: TypedAnalysisResult<VisionPayload>;
}

export function emptyTypedResult<T>(
  type: AnalysisResultType,
): TypedAnalysisResult<T> {
  return {
    type,
    status: "idle",
    payload: null,
    provenance: {},
    error: null,
    updatedAt: null,
  };
}

export function emptyAnalysisResults(): AnalysisResultsState {
  return {
    waveform: emptyTypedResult("waveform"),
    spectrogram: emptyTypedResult("spectrogram"),
    psd: emptyTypedResult("psd"),
    bandPower: emptyTypedResult("band_power"),
    topomap: emptyTypedResult("topomap"),
    comparison: emptyTypedResult("comparison"),
  };
}

export function emptyVisionState(): VisionState {
  return {
    selectedImageId: null,
    uploadedFigure: emptyTypedResult("uploaded_figure"),
    interpretation: emptyTypedResult("vision_interpretation"),
  };
}

export function tabToResultKey(
  tab: VisualizationTab,
): keyof AnalysisResultsState | "vision" | null {
  switch (tab) {
    case "waveform":
      return "waveform";
    case "spectrogram":
      return "spectrogram";
    case "psd":
      return "psd";
    case "band_power":
      return "bandPower";
    case "topomap":
      return "topomap";
    case "comparison":
      return "comparison";
    default:
      return null;
  }
}

export function visualizationToPlotPayload(v: Visualization): ImagePlotPayload {
  return {
    kind: "plot_image",
    imageUrl: v.imageUrl ?? "",
    title: v.title,
    visualizationId: v.id,
    channel: v.channel,
    band: v.band,
    condition: v.condition,
    compareWith: v.compareWith,
  };
}

/**
 * Seed analysis result slots from sample-linked visualizations ONLY.
 * Does not touch vision state.
 */
export function analysisResultsFromVisualizations(
  visualizations: Visualization[],
  provenance: ResultProvenance = {},
): AnalysisResultsState {
  const next = emptyAnalysisResults();
  const now = new Date().toISOString();

  for (const v of visualizations) {
    if (!v.imageUrl) continue;
    const key = tabToResultKey(v.tab);
    if (!key || key === "vision") continue;

    if (key === "waveform") {
      next.waveform = {
        type: "waveform",
        status: "ready",
        payload: {
          kind: "static_plot",
          imageUrl: v.imageUrl,
          channelLabels: v.channel ? [v.channel] : undefined,
        },
        provenance: {
          ...provenance,
          visualizationId: v.id,
          channel: v.channel,
          band: v.band,
          source: "sample_visualization",
        },
        error: null,
        updatedAt: now,
      };
      continue;
    }

    if (key === "bandPower") {
      next.bandPower = {
        type: "band_power",
        status: "ready",
        payload: {
          kind: "band_power_table",
          rows: [],
          imageUrl: v.imageUrl,
        },
        provenance: {
          ...provenance,
          visualizationId: v.id,
          channel: v.channel,
          band: v.band,
          source: "sample_visualization",
        },
        error: null,
        updatedAt: now,
      };
      continue;
    }

    if (key === "comparison") {
      next.comparison = {
        type: "comparison",
        status: "ready",
        payload: {
          kind: "comparison",
          imageUrl: v.imageUrl,
          conditionA: v.condition,
          conditionB: v.compareWith,
        },
        provenance: {
          ...provenance,
          visualizationId: v.id,
          conditionA: v.condition,
          conditionB: v.compareWith,
          sampleIdA: provenance.sampleIdA ?? provenance.sampleId,
          sampleIdB: provenance.sampleIdB,
          source: "sample_visualization",
        },
        error: null,
        updatedAt: now,
      };
      continue;
    }

    const plot = visualizationToPlotPayload(v);
    const slot = next[key] as TypedAnalysisResult<ImagePlotPayload>;
    next[key] = {
      ...slot,
      type: key === "psd" ? "psd" : key === "spectrogram" ? "spectrogram" : "topomap",
      status: "ready",
      payload: plot,
      provenance: {
        ...provenance,
        visualizationId: v.id,
        channel: v.channel,
        band: v.band,
        source: "sample_visualization",
      },
      error: null,
      updatedAt: now,
    };
  }

  return next;
}

export function setReadyResult<T>(
  type: AnalysisResultType,
  payload: T,
  provenance: ResultProvenance = {},
): TypedAnalysisResult<T> {
  return {
    type,
    status: "ready",
    payload,
    provenance,
    error: null,
    updatedAt: new Date().toISOString(),
  };
}

export function emptyStateMessage(
  tab: VisualizationTab,
  opts: { hasEeg: boolean; hasImage: boolean },
): { title: string; body: string } {
  switch (tab) {
    case "waveform":
      return {
        title: "No waveform",
        body: opts.hasEeg
          ? "Waveform is loading or unavailable for this sample."
          : "Load or select an EEG sample to view the waveform.",
      };
    case "spectrogram":
      return {
        title: "No spectrogram",
        body: opts.hasEeg
          ? "No spectrogram result for this sample yet."
          : "Load an EEG sample to compute or view a spectrogram.",
      };
    case "psd":
      return {
        title: "No PSD",
        body: opts.hasEeg
          ? "Run PSD analysis for the selected sample/channel, or load a sample with a PSD plot."
          : "Load an EEG sample to view PSD.",
      };
    case "band_power":
      return {
        title: "No band-power result",
        body: opts.hasEeg
          ? "Ask a band-power / ranking question, or load a sample with band-power plots."
          : "Load an EEG sample to compute band power.",
      };
    case "topomap":
      return {
        title: "No topomap",
        body: opts.hasEeg
          ? "No generated topomap for this sample yet. Upload a topomap figure only if you intend visual inspection."
          : "Load an EEG sample to view a topomap, or upload a figure for vision analysis (not shown in other tabs).",
      };
    case "comparison":
      return {
        title: "No comparison",
        body: "Run a condition comparison analysis, or load a sample that includes a comparison plot.",
      };
    default:
      return {
        title: "No result",
        body: "This tab has no typed analysis result yet.",
      };
  }
}
