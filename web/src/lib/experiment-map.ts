import type {
  AgentAnswer,
  AnalysisHistoryItem,
  EEGMetadata,
  Experiment,
  ExperimentFile,
  ExperimentMetadata,
  ExperimentStatus,
  Visualization,
  VisualizationTab,
} from "./types";
import { resolveApiUrl } from "./config";

const VIZ_TABS = new Set<string>([
  "waveform",
  "topomap",
  "spectrogram",
  "psd",
  "band_power",
  "comparison",
  "figure",
]);

export interface ApiUploadedArtifact {
  id: string;
  name: string;
  kind: "eeg" | "figure" | "metadata";
  sizeBytes?: number;
  size_bytes?: number;
  contentType?: string;
  content_type?: string;
  imageId?: string | null;
  image_id?: string | null;
  storedPath?: string | null;
  stored_path?: string | null;
}

export interface ApiVisualization {
  id: string;
  tab: string;
  title: string;
  imageUrl?: string | null;
  image_url?: string | null;
  index?: number;
  channel?: string | null;
  band?: string | null;
  condition?: string | null;
  sampleId?: string | null;
}

export interface ApiExperimentPayload {
  id: string;
  experiment_id?: string;
  eeg?: Partial<EEGMetadata> & { sampleId?: string | null; format?: string | null } | null;
  figure?: {
    id?: string;
    filename?: string;
    url?: string;
    type?: string;
    label?: string;
  } | null;
  metadata?: Partial<ExperimentMetadata> & { sampleId?: string | null } | null;
  visualizations?: ApiVisualization[];
  modalities?: Partial<Record<"eeg" | "metadata" | "vision" | "text", boolean>>;
  files?: ApiUploadedArtifact[];
  status?: ExperimentStatus;
  isDemo?: boolean;
  is_demo?: boolean;
  errorMessage?: string | null;
  error_message?: string | null;
  analysis_history?: AnalysisHistoryItem[];
}

export interface ApiUploadResponse {
  experimentId: string;
  assetId: string;
  uploaded_artifacts?: ApiUploadedArtifact[];
  detected_input_types?: string[];
  available_visualizations?: ApiVisualization[];
  metadata?: Partial<ExperimentMetadata> & { sampleId?: string | null } | null;
  eeg?: Partial<EEGMetadata> & { sampleId?: string | null; format?: string | null } | null;
  status?: string;
  error?: string | null;
}

function normalizeTab(tab: string): VisualizationTab {
  if (tab === "figure") return "topomap";
  if (VIZ_TABS.has(tab) && tab !== "figure") return tab as VisualizationTab;
  return "topomap";
}

function artifactToFile(a: ApiUploadedArtifact): ExperimentFile {
  const id = a.imageId ?? a.image_id ?? a.id;
  const kind = a.kind;
  const url =
    kind === "figure" ? resolveApiUrl(`/api/visualization/${id}`) : undefined;
  return {
    id,
    name: a.name,
    kind,
    sizeBytes: a.sizeBytes ?? a.size_bytes ?? 0,
    status: "ready",
    progress: 100,
    url,
    mimeType: a.contentType ?? a.content_type ?? undefined,
  };
}

function mapVisualizations(list: ApiVisualization[] | undefined): Visualization[] {
  return (list ?? []).map((v, i) => ({
    id: v.id,
    tab: normalizeTab(v.tab),
    title: v.title,
    imageUrl: resolveApiUrl(v.imageUrl ?? v.image_url ?? undefined),
    index: v.index ?? i,
    channel: v.channel ?? undefined,
    band: v.band ?? undefined,
    condition: v.condition ?? undefined,
  }));
}

function defaultMetadata(
  meta?: (Partial<ExperimentMetadata> & { sampleId?: string | null }) | null,
): ExperimentMetadata {
  return {
    subject: meta?.subject ?? "",
    run: meta?.run ?? "",
    taskType: meta?.taskType ?? "",
    movementCondition: meta?.movementCondition ?? "",
    samplingRateHz: meta?.samplingRateHz ?? 160,
    channels: meta?.channels ?? 64,
    recordingDurationSec: meta?.recordingDurationSec,
  };
}

function mapHistoryItems(raw: unknown[] | undefined): AnalysisHistoryItem[] {
  if (!raw?.length) return [];
  return raw.map((item, i) => {
    const h = item as Record<string, unknown>;
    const nested = h.answer as AgentAnswer | undefined;
    if (nested && typeof nested === "object" && typeof nested.answer === "string") {
      return {
        id: String(h.id ?? nested.id ?? `hist-${i}`),
        question: String(h.question ?? nested.question ?? ""),
        route: (h.route as AnalysisHistoryItem["route"]) ?? nested.route ?? "TEXT",
        timestamp: String(h.timestamp ?? new Date().toISOString()),
        answerPreview: String(
          h.answerPreview ?? h.answer_preview ?? nested.answer.slice(0, 110),
        ),
        answer: nested,
      };
    }
    const preview = String(h.answer_preview ?? h.answerPreview ?? "");
    const id = String(h.id ?? `hist-api-${i}`);
    const route = (h.route as AnalysisHistoryItem["route"]) ?? "TEXT";
    const question = String(h.question ?? "");
    const stub: AgentAnswer = {
      id: `ans-from-${id}`,
      question,
      answer: preview || "(Prior analysis — full payload not stored on server.)",
      route,
      computedEvidence: [],
      visualEvidence: [],
      modelInterpretation: "",
      toolsUsed: [],
      verification: { status: "skipped" },
      uncertainty: "",
      timing: { totalMs: typeof h.total_ms === "number" ? h.total_ms : undefined },
      system: {
        textModel: "—",
        visionModel: "—",
        precision: "INT8 W8A8",
        serving: "—",
        route,
      },
      timeline: [],
      isDemo: false,
    };
    return {
      id,
      question,
      route,
      timestamp: String(h.timestamp ?? new Date().toISOString()),
      answerPreview: preview.slice(0, 110) || question.slice(0, 110),
      answer: stub,
    };
  });
}

/** Map GET /api/experiment/{id} (or upload merge) into frontend Experiment. */
export function mapExperimentFromApi(
  raw: ApiExperimentPayload,
  opts?: { preserveHistory?: AnalysisHistoryItem[]; selectedImageId?: string | null },
): Experiment {
  const files = raw.files ?? [];
  const eeg_files = files.filter((f) => f.kind === "eeg").map(artifactToFile);
  const metadata_files = files.filter((f) => f.kind === "metadata").map(artifactToFile);
  const image_files = files.filter((f) => f.kind === "figure").map(artifactToFile);
  const visualizations = mapVisualizations(raw.visualizations);

  const selected_image_id =
    opts?.selectedImageId !== undefined
      ? opts.selectedImageId
      : image_files[0]?.id ?? null;

  const selected = image_files.find((f) => f.id === selected_image_id);
  const eeg = raw.eeg
    ? {
        filename: raw.eeg.filename,
        format: (raw.eeg.format as EEGMetadata["format"]) ?? undefined,
        samplingRateHz: Number(raw.eeg.samplingRateHz ?? 160),
        channels: Number(raw.eeg.channels ?? 64),
        channelLabels: raw.eeg.channelLabels,
        durationSec: raw.eeg.durationSec,
        autoDetected: raw.eeg.autoDetected,
      }
    : undefined;

  const figure = selected
    ? {
        id: selected.id,
        filename: selected.name,
        url: selected.url ?? "",
        type: "figure" as const,
        label: selected.name,
      }
    : raw.figure?.id
      ? {
          id: raw.figure.id,
          filename: raw.figure.filename ?? raw.figure.label ?? raw.figure.id,
          url: resolveApiUrl(raw.figure.url) ?? "",
          type: "figure" as const,
          label: raw.figure.label ?? raw.figure.filename ?? raw.figure.id,
        }
      : undefined;

  return {
    id: raw.id,
    experiment_id: raw.experiment_id ?? raw.id,
    eeg_files,
    metadata_files,
    image_files,
    selected_image_id,
    analysis_history: opts?.preserveHistory ?? mapHistoryItems(raw.analysis_history as unknown[]),
    eeg,
    figure,
    metadata: defaultMetadata(raw.metadata),
    visualizations,
    modalities: {
      eeg: (raw.modalities?.eeg ?? false) || eeg_files.length > 0 || !!eeg,
      metadata: (raw.modalities?.metadata ?? false) || metadata_files.length > 0,
      vision:
        (raw.modalities?.vision ?? false) ||
        image_files.length > 0 ||
        visualizations.length > 0,
      text: raw.modalities?.text ?? true,
    },
    status: raw.status ?? "ready",
    isDemo: raw.isDemo ?? raw.is_demo ?? false,
    errorMessage: raw.errorMessage ?? raw.error_message ?? undefined,
  };
}

/** Apply upload response onto current experiment (or create one). */
export function mergeUploadResponse(
  prev: Experiment | null,
  res: ApiUploadResponse,
  localFile: { name: string; kind: "eeg" | "figure" | "metadata"; sizeBytes: number },
): Experiment {
  const artifacts = res.uploaded_artifacts ?? [];
  const visualizations = mapVisualizations(res.available_visualizations);

  let eeg_files = artifacts.filter((f) => f.kind === "eeg").map(artifactToFile);
  let metadata_files = artifacts.filter((f) => f.kind === "metadata").map(artifactToFile);
  let image_files = artifacts.filter((f) => f.kind === "figure").map(artifactToFile);

  // If artifacts list is incomplete, keep prior files and upsert this asset
  if (artifacts.length === 0 && prev) {
    eeg_files = prev.eeg_files;
    metadata_files = prev.metadata_files;
    image_files = prev.image_files;
    const file: ExperimentFile = {
      id: res.assetId,
      name: localFile.name,
      kind: localFile.kind,
      sizeBytes: localFile.sizeBytes,
      status: "ready",
      progress: 100,
      url:
        localFile.kind === "figure"
          ? resolveApiUrl(`/api/visualization/${res.assetId}`)
          : undefined,
    };
    if (localFile.kind === "eeg") eeg_files = [...eeg_files.filter((f) => f.id !== file.id), file];
    if (localFile.kind === "metadata")
      metadata_files = [...metadata_files.filter((f) => f.id !== file.id), file];
    if (localFile.kind === "figure")
      image_files = [...image_files.filter((f) => f.id !== file.id), file];
  }

  const selected_image_id =
    localFile.kind === "figure"
      ? prev?.selected_image_id ?? res.assetId
      : prev?.selected_image_id ?? image_files[0]?.id ?? null;

  const selected = image_files.find((f) => f.id === selected_image_id);
  const eegRaw = res.eeg;
  const eeg = eegRaw
    ? {
        filename: eegRaw.filename ?? localFile.name,
        format: (eegRaw.format as EEGMetadata["format"]) ?? "json",
        samplingRateHz: Number(eegRaw.samplingRateHz ?? 160),
        channels: Number(eegRaw.channels ?? 64),
        channelLabels: eegRaw.channelLabels,
        autoDetected: eegRaw.autoDetected ?? true,
      }
    : prev?.eeg;

  return {
    id: res.experimentId,
    experiment_id: res.experimentId,
    isDemo: false,
    status: (res.status as ExperimentStatus) ?? "ready",
    eeg_files,
    metadata_files,
    image_files,
    selected_image_id,
    analysis_history: prev?.analysis_history ?? [],
    eeg,
    figure: selected
      ? {
          id: selected.id,
          filename: selected.name,
          url: selected.url ?? "",
          type: "figure",
          label: selected.name,
        }
      : prev?.figure,
    metadata: defaultMetadata(res.metadata ?? prev?.metadata),
    visualizations: visualizations.length ? visualizations : prev?.visualizations ?? [],
    modalities: {
      eeg: eeg_files.length > 0 || !!eeg,
      metadata: metadata_files.length > 0,
      vision: image_files.length > 0 || visualizations.length > 0,
      text: true,
    },
  };
}
