/** Frontend ↔ FastAPI contract types (API-ready; mock fallback until backend is live) */

export type Modality = "eeg" | "metadata" | "vision" | "text";

export type FrequencyBand = "delta" | "theta" | "alpha_mu" | "beta";

export type VisualizationTab =
  | "waveform"
  | "topomap"
  | "spectrogram"
  | "psd"
  | "band_power"
  | "comparison";

export type AnalysisRoute = "TEXT" | "VISION";

export type ExperimentStatus = "empty" | "ready" | "processing" | "error";

export type UploadStatus = "idle" | "uploading" | "ready" | "error";

export type BackendMode = "demo" | "live" | "unavailable";

export type ExperimentFileKind = "eeg" | "figure" | "metadata";

export type TimelineStageStatus =
  | "pending"
  | "running"
  | "complete"
  | "error"
  | "skipped";

export type VerificationStatus =
  | "passed"
  | "triggered"
  | "recovered"
  | "skipped"
  | "unavailable";

export interface EEGMetadata {
  filename?: string;
  format?: "edf" | "csv" | "npy" | "hdf5" | "json";
  samplingRateHz: number;
  channels: number;
  channelLabels?: string[];
  durationSec?: number;
  autoDetected?: boolean;
}

export interface ExperimentMetadata {
  subject: string;
  run: string;
  taskType: string;
  movementCondition: string;
  samplingRateHz: number;
  channels: number;
  recordingDurationSec?: number;
  sampleId?: string;
}

export interface ImageAsset {
  id: string;
  filename: string;
  url: string;
  type: VisualizationTab | "figure";
  label: string;
  channel?: string;
  band?: string;
  condition?: string;
}

/** Unified uploaded / demo asset for EEG, figures, or metadata */
export interface ExperimentFile {
  id: string;
  name: string;
  kind: ExperimentFileKind;
  sizeBytes: number;
  status: UploadStatus;
  error?: string;
  progress?: number;
  /** Object URL or static path (figures) */
  url?: string;
  mimeType?: string;
}

/** @deprecated Prefer ExperimentFile — kept for gradual migration */
export type UploadedFile = ExperimentFile;

export interface AnalysisHistoryItem {
  id: string;
  question: string;
  route: AnalysisRoute;
  timestamp: string;
  answerPreview: string;
  answer: AgentAnswer;
}

export interface Experiment {
  /** Demo-mode local id; maps to experiment_id for API calls. */
  id: string;
  experiment_id: string;
  eeg_files: ExperimentFile[];
  metadata_files: ExperimentFile[];
  image_files: ExperimentFile[];
  selected_image_id: string | null;
  analysis_history: AnalysisHistoryItem[];
  eeg?: EEGMetadata;
  /** Derived from selected_image_id for explorer convenience */
  figure?: ImageAsset;
  metadata: ExperimentMetadata;
  visualizations: Visualization[];
  modalities: Record<Modality, boolean>;
  status?: ExperimentStatus;
  isDemo?: boolean;
  errorMessage?: string;
}

export interface Visualization {
  id: string;
  tab: VisualizationTab;
  title: string;
  imageUrl?: string;
  index: number;
  channel?: string;
  band?: string;
  condition?: string;
  compareWith?: string;
}

export interface ResearchQuestion {
  text: string;
  experimentId?: string;
}

export interface TimelineStage {
  id: string;
  name: string;
  status: TimelineStageStatus;
  latencyMs?: number;
  summary?: string;
}

/** @deprecated Prefer TimelineStage */
export type ToolInvocation = TimelineStage;

export interface ComputedEvidenceItem {
  label: string;
  value: string;
  unit?: string;
  tool?: string;
  highlight?: boolean;
}

export type EvidenceItem = ComputedEvidenceItem;

export interface VisualEvidenceItem {
  id: string;
  label: string;
  tab: VisualizationTab | string;
  observation?: string;
  imageUrl?: string;
  image_type?: string | null;
  vlm_interpretation?: string | null;
  provenance?: string | null;
}

export interface VerificationInfo {
  status: VerificationStatus;
  message?: string;
  recoveryPerformed?: boolean;
}

export interface TimingInfo {
  totalMs?: number;
  routingMs?: number;
  toolsMs?: number;
  visionMs?: number;
  synthesisMs?: number;
  verificationMs?: number;
}

export interface SystemInfo {
  textModel: string;
  visionModel: string;
  precision: string;
  serving: string;
  route: AnalysisRoute;
  verifierStatus?: VerificationStatus;
}

export interface AgentAnswer {
  id: string;
  question: string;
  answer: string;
  route: AnalysisRoute;
  computedEvidence: ComputedEvidenceItem[];
  visualEvidence: VisualEvidenceItem[];
  modelInterpretation: string;
  toolsUsed: string[];
  verification: VerificationInfo;
  uncertainty: string;
  timing: TimingInfo;
  system: SystemInfo;
  timeline: TimelineStage[];
  isDemo: boolean;
  rawToolOutput?: string;
  selectedImageId?: string | null;
  selectedImageName?: string | null;
  evidence?: ComputedEvidenceItem[];
  visualRefs?: VisualEvidenceItem[];
}

export interface SystemMetrics {
  model: string;
  visionModel: string;
  postTraining: string;
  serving: string;
  precision: "BF16" | "INT8" | "INT4" | "INT8 W8A8";
  /** Null until measured — do not invent zeros as live telemetry */
  ttftMs: number | null;
  tokensPerSec: number | null;
  p95LatencyMs: number | null;
  gpuUtilizationPct: number | null;
  gpuMemoryUsedMb?: number | null;
  gpuMemoryTotalMb?: number | null;
  lastRequestLatencyMs?: number | null;
  route?: AnalysisRoute | null;
  verifierStatus?: VerificationStatus | string | null;
  servingMode?: string | null;
}

export interface AppSettings {
  theme: "dark" | "light";
  defaultBand: FrequencyBand;
  answerDetail: "concise" | "detailed";
  showRawToolOutput: boolean;
  showUncertainty: boolean;
  showSystemMetrics: boolean;
  autoGenerateVisuals: boolean;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "unavailable";
  version?: string;
  backend?: string;
  textModel?: string;
  visionModel?: string;
  servingMode?: string;
  agentLoaded?: boolean;
  visionLoaded?: boolean;
}

export interface UploadRequest {
  fileType: "eeg" | "figure" | "metadata";
  filename: string;
  contentType?: string;
  experimentId?: string | null;
}

export interface UploadResponse {
  experimentId: string;
  assetId: string;
  metadata?: Partial<ExperimentMetadata>;
  eeg?: Partial<EEGMetadata>;
  status?: UploadStatus | string;
  error?: string;
  uploaded_artifacts?: unknown[];
  detected_input_types?: string[];
  available_visualizations?: Visualization[];
}

export interface AnalyzeRequest {
  experimentId: string;
  question: string;
  tools?: string[];
  settings?: Partial<AppSettings>;
  /** Preferred — maps to backend imageId */
  imageId?: string | null;
  /** @deprecated use imageId */
  selectedImageId?: string | null;
  visualizationId?: string | null;
  context?: Record<string, unknown> | null;
}

export interface AnalyzeResponse {
  answer: string;
  route: AnalysisRoute;
  computed_evidence: ComputedEvidenceItem[];
  visual_evidence: VisualEvidenceItem[];
  model_interpretation: string;
  tools_used: string[];
  verification: VerificationInfo;
  uncertainty: string;
  timing: TimingInfo;
  system: SystemInfo;
  timeline?: TimelineStage[];
  question?: string;
  id?: string;
  raw_tool_output?: string;
  route_detail?: {
    intent?: Record<string, unknown>;
    requires_vision?: boolean;
    requested_visual_type?: string | null;
    question_type?: string | null;
  };
  experiment_id?: string;
}

export type GetExperimentResponse = Experiment;
export type GetVisualizationResponse = Visualization;
export type GetSystemMetricsResponse = SystemMetrics;
