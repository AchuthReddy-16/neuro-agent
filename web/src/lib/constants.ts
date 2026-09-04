import type {
  AnalyzeResponse,
  Experiment,
  SystemInfo,
  SystemMetrics,
  Visualization,
  VisualizationTab,
} from "./types";

export const DEMO_SUBJECT = "S026";
export const DEMO_RUN = "R12";

export const SAMPLE_IMAGES: Record<string, string> = {
  topomap_left: "/samples/s026_topomap_left_fist.png",
  topomap_right: "/samples/s026_topomap_right_fist.png",
  psd_left: "/samples/s026_psd_left_fist.png",
  psd_right: "/samples/s026_psd_right_fist.png",
  spectrogram: "/samples/s026_spectrogram_left_fist.png",
  band_power: "/samples/s026_band_power_left_fist.png",
  comparison: "/samples/s026_comparison.png",
  waveform_static: "/samples/s026_waveform_left_fist.png",
};

export const EXPLORER_TABS: { id: VisualizationTab; label: string }[] = [
  { id: "waveform", label: "Waveform" },
  { id: "topomap", label: "Topomap" },
  { id: "spectrogram", label: "Spectrogram" },
  { id: "psd", label: "PSD" },
  { id: "band_power", label: "Band Power" },
  { id: "comparison", label: "Comparison" },
];

export const EEG_CHANNELS = [
  "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6",
  "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
  "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
];

export const WAVEFORM_CHANNELS = ["C3", "CZ", "C4", "FC3", "FC4", "CP3", "CP4"];

export const ANALYSIS_TOOLS = [
  "Band Power",
  "Channel Ranking",
  "Condition Comparison",
  "Effect Size",
  "Correlation",
  "Classifier",
  "Outlier Detection",
  "Generate Topomap",
] as const;

export const EXAMPLE_QUESTIONS = [
  "Which channels are most discriminative?",
  "Which channels have the highest beta power?",
  "Interpret the selected topomap figure",
  "Look at the spectrogram — where is mu suppression strongest?",
  "Show the strongest alpha/mu changes",
];

export const DEMO_QUESTION =
  "Compare beta-band activity between left- and right-fist conditions.";

export const TIMELINE_STAGE_NAMES = [
  "Routing",
  "Tool execution",
  "Vision analysis",
  "Evidence assembly",
  "Synthesis",
  "Verification",
  "Recovery",
] as const;

/** Honest backend support: EEG/metadata = JSON with sample_id; figures = images. Raw EDF is not parsed. */
export const EEG_ACCEPT = ".json,application/json";
export const FIGURE_ACCEPT = ".png,.jpg,.jpeg,.webp";
export const METADATA_ACCEPT = ".json,application/json";
export const EEG_SUPPORTED = [".json"] as const;
export const FIGURE_SUPPORTED = [".png", ".jpg", ".jpeg", ".webp"] as const;
export const METADATA_SUPPORTED = [".json"] as const;

export function buildDemoVisualizations(): Visualization[] {
  return [
    {
      id: "viz-waveform-01",
      tab: "waveform",
      title: "Waveform — Left Fist",
      imageUrl: SAMPLE_IMAGES.waveform_static,
      index: 1,
      channel: "C3",
      condition: "Left Fist",
    },
    {
      id: "viz-topomap-01",
      tab: "topomap",
      title: "Topomap — Left Fist (Beta)",
      imageUrl: SAMPLE_IMAGES.topomap_left,
      index: 1,
      channel: "C3",
      band: "Beta",
      condition: "Left Fist",
    },
    {
      id: "viz-topomap-02",
      tab: "topomap",
      title: "Topomap — Right Fist (Beta)",
      imageUrl: SAMPLE_IMAGES.topomap_right,
      index: 2,
      channel: "C4",
      band: "Beta",
      condition: "Right Fist",
    },
    {
      id: "viz-spectrogram-01",
      tab: "spectrogram",
      title: "Spectrogram — Left Fist",
      imageUrl: SAMPLE_IMAGES.spectrogram,
      index: 1,
      channel: "C3",
      condition: "Left Fist",
    },
    {
      id: "viz-psd-01",
      tab: "psd",
      title: "PSD — Left Fist",
      imageUrl: SAMPLE_IMAGES.psd_left,
      index: 1,
      channel: "C3",
      condition: "Left Fist",
    },
    {
      id: "viz-psd-02",
      tab: "psd",
      title: "PSD — Right Fist",
      imageUrl: SAMPLE_IMAGES.psd_right,
      index: 2,
      channel: "C4",
      condition: "Right Fist",
    },
    {
      id: "viz-band-01",
      tab: "band_power",
      title: "Band Power — Left Fist",
      imageUrl: SAMPLE_IMAGES.band_power,
      index: 1,
      band: "Beta",
      condition: "Left Fist",
    },
    {
      id: "viz-compare-01",
      tab: "comparison",
      title: "Condition Comparison",
      imageUrl: SAMPLE_IMAGES.comparison,
      index: 1,
      condition: "Left Fist",
      compareWith: "Right Fist",
    },
  ];
}

export function createDemoExperiment(): Experiment {
  const eegFile = {
    id: "file-eeg-demo",
    name: "S026R12.edf",
    kind: "eeg" as const,
    sizeBytes: 2_450_000,
    status: "ready" as const,
  };
  const metaFile = {
    id: "file-meta-demo",
    name: "metadata.json",
    kind: "metadata" as const,
    sizeBytes: 420,
    status: "ready" as const,
  };
  const images = [
    {
      id: "img-topo-demo",
      name: "beta_topomap.png",
      kind: "figure" as const,
      sizeBytes: 186_000,
      status: "ready" as const,
      url: SAMPLE_IMAGES.topomap_left,
    },
    {
      id: "img-psd-demo",
      name: "psd_left_fist.png",
      kind: "figure" as const,
      sizeBytes: 142_000,
      status: "ready" as const,
      url: SAMPLE_IMAGES.psd_left,
    },
    {
      id: "img-spec-demo",
      name: "spectrogram_left_fist.png",
      kind: "figure" as const,
      sizeBytes: 198_000,
      status: "ready" as const,
      url: SAMPLE_IMAGES.spectrogram,
    },
  ];

  return {
    id: "exp-s026-demo",
    experiment_id: "exp-s026-demo",
    isDemo: true,
    status: "ready",
    eeg_files: [eegFile],
    metadata_files: [metaFile],
    image_files: images,
    selected_image_id: images[0].id,
    analysis_history: [],
    eeg: {
      filename: "S026R12.edf",
      format: "edf",
      samplingRateHz: 160,
      channels: 64,
      durationSec: 480,
      autoDetected: true,
      channelLabels: EEG_CHANNELS,
    },
    figure: {
      id: images[0].id,
      filename: images[0].name,
      url: images[0].url,
      type: "topomap",
      label: "Beta Topomap",
      channel: "C3",
      band: "Beta",
      condition: "Left Fist",
    },
    metadata: {
      subject: DEMO_SUBJECT,
      run: DEMO_RUN,
      taskType: "Imagery",
      movementCondition: "Left Fist",
      samplingRateHz: 160,
      channels: 64,
      recordingDurationSec: 480,
    },
    visualizations: buildDemoVisualizations(),
    modalities: {
      eeg: true,
      metadata: true,
      vision: true,
      text: true,
    },
  };
}

export const DEFAULT_SYSTEM_INFO: SystemInfo = {
  textModel: "Qwen3-4B",
  visionModel: "Qwen2.5-VL-3B",
  precision: "INT8 W8A8",
  serving: "vLLM",
  route: "TEXT",
  verifierStatus: "skipped",
};

export const MOCK_SYSTEM_METRICS: SystemMetrics = {
  model: "Qwen3-4B",
  visionModel: "Qwen2.5-VL-3B",
  postTraining: "SFT + RLVR",
  serving: "vLLM",
  precision: "INT8 W8A8",
  ttftMs: 118,
  tokensPerSec: 54.3,
  p95LatencyMs: 740,
  gpuUtilizationPct: 68,
  route: "TEXT",
  verifierStatus: "skipped",
};

/** Map AnalyzeResponse → internal AgentAnswer helper lives in mock/responses */
export type { AnalyzeResponse };
