"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type {
  AgentAnswer,
  AnalysisHistoryItem,
  AppSettings,
  BackendMode,
  Experiment,
  ExperimentFile,
  ExperimentStatus,
  HealthResponse,
  ImageAsset,
  SystemMetrics,
  VisualizationTab,
} from "@/lib/types";
import {
  ApiError,
  analyzeLive,
  checkHealth,
  createExperiment,
  fetchSystemMetricsWithFallback,
  getExperiment,
  uploadAsset,
} from "@/lib/api";
import { mergeUploadResponse, mapExperimentFromApi } from "@/lib/experiment-map";
import {
  explicitLiveImageId,
} from "@/lib/routing";
import { progressiveTimeline } from "@/lib/mock/responses";
import {
  EEG_SUPPORTED,
  FIGURE_SUPPORTED,
  METADATA_SUPPORTED,
  MOCK_SYSTEM_METRICS,
} from "@/lib/constants";
import {
  analysisResultsFromVisualizations,
  emptyAnalysisResults,
  emptyVisionState,
  resetEegDerivedResults,
  type AnalysisResultsState,
  type VisionState,
} from "@/lib/analysis-results";
import {
  applyAnswerToAnalysisResults,
  applyAnswerToVisionState,
  visionStateFromSelectedImage,
} from "@/lib/merge-analysis-results";
import {
  captureSurfaceSnapshot,
  emptySurfaceSnapshot,
  stripUserFiguresForLinkedSample,
  type SurfaceSnapshot,
} from "@/lib/surface-session";
import { migrateClientStorage } from "@/lib/storage-migration";
/** Which product surface owns the current client session. */
export type ExperienceMode = "chat" | "workspace";

interface ExperimentContextValue {
  experienceMode: ExperienceMode;
  experiment: Experiment | null;
  answers: AgentAnswer[];
  currentAnswer: AgentAnswer | null;
  activeAnswerId: string | null;
  isAnalyzing: boolean;
  activeTab: VisualizationTab;
  focusedVizId: string | null;
  settings: AppSettings;
  systemMetrics: SystemMetrics;
  precision: SystemMetrics["precision"];
  backendMode: BackendMode;
  experimentStatus: ExperimentStatus;
  uploadError: string | null;
  analysisError: string | null;
  explorerError: string | null;
  explorerLoading: boolean;
  workspaceEpoch: number;
  selectedImage: ExperimentFile | null;
  /** Typed EEG/analysis result slots — independent of uploaded figures. */
  analysisResults: AnalysisResultsState;
  /** Vision / figure attachment + interpretation — never feeds EEG tabs. */
  visionState: VisionState;
  setActiveTab: (tab: VisualizationTab) => void;
  focusVisualization: (id: string, tab?: VisualizationTab) => void;
  /** Enter live Chat — isolated session; never inherits workspace figures/selection. */
  beginChatSession: () => Promise<void>;
  /** @deprecated use beginChatSession */
  beginDemoSession: () => Promise<void>;
  /** Enter / restore Research Workspace surface (does not wipe prior workspace snap). */
  beginWorkspaceSession: () => void;
  /** Load linked live sample from API when available (workspace helper). */
  loadDemo: () => void | Promise<void>;
  /** Stable ids for dual-surface isolation (debug / tests). */
  chatSessionId: string | null;
  workspaceExperimentId: string | null;  uploadEEG: (file: File) => Promise<void>;
  uploadFigure: (file: File) => Promise<void>;
  uploadMetadata: (file: File) => Promise<void>;
  removeFile: (fileId: string) => void;
  selectImage: (imageId: string) => void;
  clearImageSelection: () => void;
  updateMetadata: (patch: Partial<Experiment["metadata"]>) => void;
  clearExperiment: () => void;
  analyze: (question: string) => Promise<void>;
  /** @deprecated */
  runDemoAnalysis: () => Promise<void>;
  restoreAnalysis: (historyId: string) => void;
  updateSettings: (patch: Partial<AppSettings>) => void;
  setPrecision: (p: SystemMetrics["precision"]) => void;
  runTool: (toolName: string) => void;
  clearUploadError: () => void;
  clearAnalysisError: () => void;
  healthInfo: HealthResponse | null;
}

const defaultSettings: AppSettings = {
  theme: "dark",
  defaultBand: "beta",
  answerDetail: "detailed",
  showRawToolOutput: false,
  showUncertainty: true,
  showSystemMetrics: true,
  autoGenerateVisuals: true,
};

const ExperimentContext = createContext<ExperimentContextValue | null>(null);

const LIVE_DEMO_EXPERIMENT_ID = "exp_demo_s001";

function humanizeAnalysisError(err: unknown): string {
  if (err instanceof ApiError) {
    const code = (err.code || "").toLowerCase();
    const msg = (err.message || "").toLowerCase();
    if (
      err.status === 503 ||
      code.includes("unavailable") ||
      code.includes("model") ||
      msg.includes("loading") ||
      msg.includes("warming") ||
      msg.includes("cuda")
    ) {
      return "Preparing research model… Please try again in a moment.";
    }
    if (err.status === 408 || msg.includes("timeout")) {
      return "The analysis timed out. Please try again.";
    }
    return err.message || "Analysis request failed.";
  }
  return "Analysis request failed. Check that the backend is reachable.";
}

function revokeBlobUrls(exp: Experiment | null) {
  exp?.image_files.forEach((f) => {
    if (f.url?.startsWith("blob:")) URL.revokeObjectURL(f.url);
  });
}

function extOf(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function newLocalId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

function emptyExperiment(): Experiment {
  const id = newLocalId("exp");
  return {
    id,
    experiment_id: id,
    isDemo: false,
    status: "ready",
    eeg_files: [],
    metadata_files: [],
    image_files: [],
    selected_image_id: null,
    analysis_history: [],
    metadata: {
      subject: "",
      run: "",
      taskType: "",
      movementCondition: "",
      samplingRateHz: 160,
      channels: 64,
    },
    visualizations: [],
    modalities: { eeg: false, metadata: false, vision: false, text: true },
  };
}

function figureFromFile(file: ExperimentFile): ImageAsset | undefined {
  if (!file.url) return undefined;
  return {
    id: file.id,
    filename: file.name,
    url: file.url,
    type: "figure",
    label: file.name,
  };
}

function syncModalities(exp: Experiment): Experiment {
  const hasEeg = exp.eeg_files.some((f) => f.status === "ready" || f.status === "uploading");
  const hasMeta =
    exp.metadata_files.some((f) => f.status === "ready" || f.status === "uploading") ||
    !!(exp.metadata.subject || exp.metadata.run);
  const hasVision = exp.image_files.some((f) => f.status === "ready" || f.status === "uploading");
  const selected = exp.image_files.find((f) => f.id === exp.selected_image_id);
  return {
    ...exp,
    modalities: {
      eeg: hasEeg || !!exp.eeg,
      metadata: hasMeta,
      vision: hasVision,
      text: true,
    },
    figure: selected ? figureFromFile(selected) : undefined,
    status:
      exp.status === "error"
        ? "error"
        : exp.eeg_files.some((f) => f.status === "uploading") ||
            exp.image_files.some((f) => f.status === "uploading") ||
            exp.metadata_files.some((f) => f.status === "uploading")
          ? "processing"
          : hasEeg || hasVision || hasMeta
            ? "ready"
            : "empty",
  };
}

function isExperimentEmpty(exp: Experiment): boolean {
  return (
    exp.eeg_files.length === 0 &&
    exp.image_files.length === 0 &&
    exp.metadata_files.length === 0 &&
    !exp.isDemo
  );
}

function validateEEG(file: File): string | null {
  const ext = extOf(file.name);
  if (!(EEG_SUPPORTED as readonly string[]).includes(ext)) {
    if ([".edf", ".csv", ".npy", ".h5", ".hdf5"].includes(ext)) {
      return `Raw ${ext.toUpperCase()} is not parsed by the backend. Upload a JSON file with sample_id (e.g. {"sample_id":"S001_R01_E000"}).`;
    }
    return `Unsupported EEG/data type "${ext || file.name}". Supported: JSON with sample_id.`;
  }
  if (file.size === 0) return "File appears empty or malformed.";
  return null;
}

function validateFigure(file: File): string | null {
  const ext = extOf(file.name);
  if (!(FIGURE_SUPPORTED as readonly string[]).includes(ext)) {
    return `Unsupported image type "${ext || file.name}". Supported: PNG, JPG, WEBP.`;
  }
  if (file.size === 0) return "File appears empty or malformed.";
  return null;
}

function validateMetadata(file: File): string | null {
  const ext = extOf(file.name);
  if (!(METADATA_SUPPORTED as readonly string[]).includes(ext)) {
    return `Unsupported metadata type "${ext || file.name}". Supported: JSON with sample_id.`;
  }
  if (file.size === 0) return "File appears empty or malformed.";
  return null;
}

async function simulateUploadProgress(onProgress: (p: number) => void): Promise<void> {
  for (const p of [22, 48, 72, 100]) {
    await new Promise((r) => setTimeout(r, 70));
    onProgress(p);
  }
}

function previewOf(answer: string): string {
  const t = answer.trim();
  if (t.length <= 110) return t;
  return `${t.slice(0, 107)}…`;
}

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [answers, setAnswers] = useState<AgentAnswer[]>([]);
  const [activeAnswerId, setActiveAnswerId] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState<VisualizationTab>("waveform");
  const [focusedVizId, setFocusedVizId] = useState<string | null>(null);
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [precision, setPrecision] = useState<SystemMetrics["precision"]>("INT8 W8A8");
  const [backendMode, setBackendMode] = useState<BackendMode>("unavailable");
  const [experienceMode, setExperienceMode] = useState<ExperienceMode>("workspace");
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [explorerError, setExplorerError] = useState<string | null>(null);
  const [explorerLoading, setExplorerLoading] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<SystemMetrics | null>(null);
  const [workspaceEpoch, setWorkspaceEpoch] = useState(0);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResultsState>(emptyAnalysisResults);
  const [visionState, setVisionState] = useState<VisionState>(emptyVisionState);
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [workspaceExperimentId, setWorkspaceExperimentId] = useState<string | null>(null);
  const experimentRef = useRef<Experiment | null>(null);
  experimentRef.current = experiment;
  const answersRef = useRef<AgentAnswer[]>([]);
  answersRef.current = answers;
  const analysisResultsRef = useRef(analysisResults);
  analysisResultsRef.current = analysisResults;
  const visionStateRef = useRef(visionState);
  visionStateRef.current = visionState;
  const experienceModeRef = useRef(experienceMode);
  experienceModeRef.current = experienceMode;
  const chatSnapRef = useRef<SurfaceSnapshot>(emptySurfaceSnapshot());
  const workspaceSnapRef = useRef<SurfaceSnapshot>(emptySurfaceSnapshot());
  const chatSessionIdRef = useRef<string | null>(null);
  const workspaceExperimentIdRef = useRef<string | null>(null);

  useEffect(() => {
    migrateClientStorage();
  }, []);

  const currentAnswer = useMemo(() => {
    if (activeAnswerId) {
      const found = answers.find((a) => a.id === activeAnswerId);
      if (found) return found;
      const fromHistory = experiment?.analysis_history.find((h) => h.id === activeAnswerId);
      if (fromHistory) return fromHistory.answer;
    }
    return answers[answers.length - 1] ?? null;
  }, [answers, activeAnswerId, experiment?.analysis_history]);

  const selectedImage = useMemo(() => {
    if (!experiment?.selected_image_id) return null;
    return experiment.image_files.find((f) => f.id === experiment.selected_image_id) ?? null;
  }, [experiment]);

  const systemMetrics = useMemo(() => {
    const base = liveMetrics ?? MOCK_SYSTEM_METRICS;
    const route = currentAnswer?.route ?? base.route ?? undefined;
    const verifierStatus =
      currentAnswer?.verification.status ?? base.verifierStatus ?? undefined;
    // Live metrics: never invent TTFT/tok/s/p95 — show nulls as-is
    if (liveMetrics && backendMode === "live") {
      return {
        ...base,
        precision: liveMetrics.precision ?? precision,
        route: route ?? null,
        verifierStatus: verifierStatus ?? null,
      };
    }
    return {
      ...base,
      precision,
      route: route ?? "TEXT",
      verifierStatus,
      ttftMs:
        precision === "INT4"
          ? 98
          : precision === "INT8" || precision === "INT8 W8A8"
            ? 118
            : 142,
      tokensPerSec:
        precision === "INT4"
          ? 62.1
          : precision === "INT8" || precision === "INT8 W8A8"
            ? 54.3
            : 48.6,
    };
  }, [precision, liveMetrics, currentAnswer, backendMode]);

  const experimentStatus: ExperimentStatus =
    experiment?.status ?? (experiment ? "ready" : "empty");

  useEffect(() => {
    document.documentElement.classList.toggle("dark", settings.theme === "dark");
    document.documentElement.classList.toggle("light", settings.theme === "light");
  }, [settings.theme]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const health = await checkHealth();
      if (cancelled) return;
      setHealthInfo(health);
      if (health.status === "ok" || health.status === "degraded") {
        setBackendMode("live");
        const { metrics, source } = await fetchSystemMetricsWithFallback();
        if (!cancelled && source === "live") setLiveMetrics(metrics);
      } else {
        setBackendMode("unavailable");
      }
    })();
    const id = window.setInterval(async () => {
      const health = await checkHealth();
      if (cancelled) return;
      setHealthInfo(health);
      if (health.status === "ok" || health.status === "degraded") {
        setBackendMode("live");
        try {
          const { metrics, source } = await fetchSystemMetricsWithFallback();
          if (!cancelled && source === "live") setLiveMetrics(metrics);
        } catch {
          /* ignore poll errors */
        }
      } else if (!cancelled) {
        setBackendMode("unavailable");
      }
    }, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const captureCurrent = useCallback((): SurfaceSnapshot => {
    return captureSurfaceSnapshot({
      sessionId:
        experienceModeRef.current === "chat"
          ? chatSessionIdRef.current
          : workspaceExperimentIdRef.current,
      experiment: experimentRef.current,
      answers: answersRef.current,
      activeAnswerId,
      analysisResults: analysisResultsRef.current,
      visionState: visionStateRef.current,
      focusedVizId,
      activeTab,
    });
  }, [activeAnswerId, focusedVizId, activeTab]);

  const applySnapshot = useCallback((snap: SurfaceSnapshot) => {
    experimentRef.current = snap.experiment;
    setExperiment(snap.experiment);
    setAnswers(snap.answers);
    setActiveAnswerId(snap.activeAnswerId);
    setAnalysisResults(snap.analysisResults);
    setVisionState(snap.visionState);
    setFocusedVizId(snap.focusedVizId);
    setActiveTab(snap.activeTab);
  }, []);

  const clearActiveSurface = useCallback(() => {
    // Critical: clear ref synchronously before any async ensureLiveExperiment
    experimentRef.current = null;
    setExperiment(null);
    setAnswers([]);
    setActiveAnswerId(null);
    setIsAnalyzing(false);
    setAnalysisResults(emptyAnalysisResults());
    setVisionState(emptyVisionState());
    setFocusedVizId(null);
    setUploadError(null);
    setAnalysisError(null);
    setExplorerError(null);
  }, []);

  /**
   * Ensure a live backend experiment for the *current* surface.
   * forceNew: always create — never reuse another surface's experiment.
   */
  const ensureLiveExperiment = useCallback(
    async (opts?: { forceNew?: boolean }): Promise<Experiment | null> => {
      const health = await checkHealth();
      if (health.status !== "ok" && health.status !== "degraded") {
        setBackendMode("unavailable");
        setHealthInfo(health);
        return null;
      }
      setBackendMode("live");
      setHealthInfo(health);

      if (!opts?.forceNew) {
        const current = experimentRef.current;
        const expId = current?.experiment_id ?? current?.id;
        if (expId && /^exp_/.test(expId) && !current?.isDemo) {
          return current;
        }
      }

      const raw = await createExperiment();
      const mapped = mapExperimentFromApi(raw, { selectedImageId: null });
      const next = {
        ...mapped,
        isDemo: false,
        image_files: [],
        selected_image_id: null,
        figure: undefined,
      };
      experimentRef.current = next;
      setExperiment(next);
      return next;
    },
    [],
  );

  /** Workspace helper: attach the API-linked sample experiment (live only). */
  const loadDemo = useCallback(async () => {
    revokeBlobUrls(experimentRef.current);
    clearActiveSurface();
    setExplorerLoading(true);
    setWorkspaceEpoch((e) => e + 1);

    try {
      const health = await checkHealth();
      if (health.status === "ok" || health.status === "degraded") {
        setBackendMode("live");
        setHealthInfo(health);
        const raw = await getExperiment(LIVE_DEMO_EXPERIMENT_ID);
        const mapped = stripUserFiguresForLinkedSample(
          mapExperimentFromApi(raw, { selectedImageId: null }),
        );
        // Linked sample: EEG/metadata + sample visualizations only — never user figures
        experimentRef.current = mapped;
        setExperiment(mapped);
        const wid = mapped.experiment_id ?? mapped.id;
        workspaceExperimentIdRef.current = wid;
        setWorkspaceExperimentId(wid);
        setAnalysisResults(
          analysisResultsFromVisualizations(mapped.visualizations, {
            experimentId: wid,
            sampleId: mapped.metadata?.sampleId ?? null,
            source: "linked_sample",
          }),
        );
        // Do NOT mark waveform ready with live_eeg — only static sample_visualization plots
        // or explicit analysis responses populate EEG-derived slots.
        setVisionState(emptyVisionState());
        setActiveTab(
          mapped.visualizations.some((v) => v.tab === "topomap")
            ? "topomap"
            : mapped.visualizations.some((v) => v.tab === "waveform")
              ? "waveform"
              : mapped.eeg
                ? "waveform"
                : "topomap",
        );
        setFocusedVizId(mapped.visualizations[0]?.id ?? null);
        setExplorerLoading(false);
        return;
      }
      setBackendMode("unavailable");
      setAnalysisError("Live API unavailable — cannot load sample.");
    } catch {
      setBackendMode("unavailable");
      setAnalysisError("Live API unavailable — cannot load sample.");
    }
    setExplorerLoading(false);
  }, [clearActiveSurface]);

  const beginChatSession = useCallback(async () => {
    // Remount while already on Chat with an active session — keep it
    if (
      experienceModeRef.current === "chat" &&
      experimentRef.current &&
      chatSessionIdRef.current
    ) {
      chatSnapRef.current = captureCurrent();
      return;
    }

    // Persist workspace before switching so return navigation restores it
    if (experienceModeRef.current === "workspace") {
      workspaceSnapRef.current = captureCurrent();
      const wid =
        experimentRef.current?.experiment_id ?? experimentRef.current?.id ?? null;
      workspaceExperimentIdRef.current = wid;
      setWorkspaceExperimentId(wid);
    }

    setExperienceMode("chat");

    // Prefer restoring an existing *chat* session (never workspace artifacts)
    const priorChat = chatSnapRef.current;
    if (priorChat.experiment && priorChat.sessionId) {
      experimentRef.current = null;
      applySnapshot(priorChat);
      chatSessionIdRef.current = priorChat.sessionId;
      setChatSessionId(priorChat.sessionId);
      setWorkspaceEpoch((e) => e + 1);
      setExplorerLoading(false);
      setIsAnalyzing(false);
      setUploadError(null);
      setAnalysisError(null);
      setExplorerError(null);
      return;
    }

    // Fresh chat — clear sync + always create a new backend experiment
    clearActiveSurface();
    setExplorerLoading(true);
    setWorkspaceEpoch((e) => e + 1);
    try {
      const next = await ensureLiveExperiment({ forceNew: true });
      const sid = next?.experiment_id ?? next?.id ?? newLocalId("chat");
      chatSessionIdRef.current = sid;
      setChatSessionId(sid);
      chatSnapRef.current = captureSurfaceSnapshot({
        sessionId: sid,
        experiment: next,
        answers: [],
        activeAnswerId: null,
        analysisResults: emptyAnalysisResults(),
        visionState: emptyVisionState(),
        focusedVizId: null,
        activeTab: "waveform",
      });
    } catch {
      setBackendMode("unavailable");
      setAnalysisError("Live API unavailable.");
      chatSessionIdRef.current = null;
      setChatSessionId(null);
    }
    setExplorerLoading(false);
  }, [captureCurrent, clearActiveSurface, applySnapshot, ensureLiveExperiment]);

  const beginDemoSession = beginChatSession;

  const beginWorkspaceSession = useCallback(() => {
    // Remount while already on Workspace with live state — keep it
    if (experienceModeRef.current === "workspace" && experimentRef.current) {
      workspaceSnapRef.current = captureCurrent();
      const wid =
        experimentRef.current.experiment_id ?? experimentRef.current.id ?? null;
      workspaceExperimentIdRef.current = wid;
      setWorkspaceExperimentId(wid);
      return;
    }

    // Persist chat before switching (keep blob URLs for later chat restore)
    if (experienceModeRef.current === "chat") {
      chatSnapRef.current = captureCurrent();
      chatSessionIdRef.current =
        experimentRef.current?.experiment_id ??
        experimentRef.current?.id ??
        chatSessionIdRef.current;
      setChatSessionId(chatSessionIdRef.current);
    }

    setExperienceMode("workspace");
    setWorkspaceEpoch((e) => e + 1);

    const prior = workspaceSnapRef.current;
    if (prior.experiment) {
      experimentRef.current = null;
      applySnapshot(prior);
      const wid = prior.sessionId ?? prior.experiment.experiment_id ?? prior.experiment.id;
      workspaceExperimentIdRef.current = wid;
      setWorkspaceExperimentId(wid);
      setExplorerLoading(false);
      setIsAnalyzing(false);
      setUploadError(null);
      setAnalysisError(null);
      setExplorerError(null);
      return;
    }

    // First visit / empty workspace
    clearActiveSurface();
    setActiveTab("waveform");
    workspaceExperimentIdRef.current = null;
    setWorkspaceExperimentId(null);
    setExplorerLoading(false);
  }, [captureCurrent, applySnapshot, clearActiveSurface]);  /** Workspace helper: load built-in sample without leaving workspace mode. */
  const loadWorkspaceDemoSample = useCallback(async () => {
    setExperienceMode("workspace");
    await loadDemo();
    workspaceSnapRef.current = captureCurrent();
  }, [loadDemo, captureCurrent]);
  const patchFileProgress = useCallback(
    (kind: "eeg" | "figure" | "metadata", fileId: string, patch: Partial<ExperimentFile>) => {
      setExperiment((prev) => {
        if (!prev) return prev;
        const key =
          kind === "eeg" ? "eeg_files" : kind === "figure" ? "image_files" : "metadata_files";
        const next = {
          ...prev,
          [key]: prev[key].map((f) => (f.id === fileId ? { ...f, ...patch } : f)),
        };
        return syncModalities(next);
      });
    },
    [],
  );

  const uploadViaApi = useCallback(
    async (
      file: File,
      fileType: "eeg" | "figure" | "metadata",
      localPendingId: string,
    ) => {
      const current = experimentRef.current;
      const expId = current?.experiment_id ?? current?.id ?? null;
      // Never attach user uploads to the shared live demo experiment id
      const isSharedDemo =
        !!current?.isDemo ||
        expId === LIVE_DEMO_EXPERIMENT_ID ||
        expId === "exp-s026-demo";
      const backendExpId =
        !isSharedDemo && expId && /^exp_[a-zA-Z0-9]+/.test(expId) ? expId : null;
      const res = await uploadAsset(
        {
          fileType,
          filename: file.name,
          contentType: file.type,
          experimentId: backendExpId,
        },
        file,
      );
      // If forking off demo, drop demo assets — only keep the pending upload slot
      const cleaned = current
        ? isSharedDemo
          ? {
              ...emptyExperiment(),
              eeg_files: current.eeg_files.filter((f) => f.id === localPendingId),
              metadata_files: current.metadata_files.filter((f) => f.id === localPendingId),
              image_files: current.image_files.filter((f) => f.id === localPendingId),
            }
          : {
              ...current,
              isDemo: false,
              eeg_files: current.eeg_files.filter((f) => f.id !== localPendingId),
              metadata_files: current.metadata_files.filter((f) => f.id !== localPendingId),
              image_files: current.image_files.filter((f) => f.id !== localPendingId),
            }
        : null;
      if (isSharedDemo && current) {
        current.image_files.forEach((f) => {
          if (f.id !== localPendingId && f.url?.startsWith("blob:")) {
            URL.revokeObjectURL(f.url);
          }
        });
      }
      const merged = mergeUploadResponse(cleaned, res, {
        name: file.name,
        kind: fileType,
        sizeBytes: file.size,
      });
      const next = { ...merged, isDemo: false };
      experimentRef.current = next;
      setExperiment(next);
      return res;
    },
    [],
  );

  /** Base experiment for uploads — fork away from demo assets instead of mutating them. */
  const baseForUpload = useCallback((prev: Experiment | null): Experiment => {
    if (!prev || prev.isDemo || prev.experiment_id === LIVE_DEMO_EXPERIMENT_ID) {
      if (prev?.isDemo) revokeBlobUrls(prev);
      return emptyExperiment();
    }
    return prev;
  }, []);

  const uploadEEG = useCallback(
    async (file: File) => {
      const err = validateEEG(file);
      if (err) {
        setUploadError(err);
        return;
      }
      setUploadError(null);
      const fileId = newLocalId("eeg");
      const pending: ExperimentFile = {
        id: fileId,
        name: file.name,
        kind: "eeg",
        sizeBytes: file.size,
        status: "uploading",
        progress: 0,
      };

      setExperiment((prev) => {
        const base = baseForUpload(prev);
        return syncModalities({
          ...base,
          isDemo: false,
          eeg_files: [...base.eeg_files.filter((f) => f.status !== "error"), pending],
        });
      });

      if (backendMode === "live") {
        try {
          patchFileProgress("eeg", fileId, { progress: 40 });
          await uploadViaApi(file, "eeg", fileId);
          setActiveTab("waveform");
          // EEG present → leave waveform idle until a real plot/analysis exists
          // (do not seed simulated live_eeg).
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : "EEG upload failed";
          setUploadError(msg);
          patchFileProgress("eeg", fileId, { status: "error", error: msg, progress: 0 });
        }
        return;
      }

      await simulateUploadProgress((p) => patchFileProgress("eeg", fileId, { progress: p }));
      patchFileProgress("eeg", fileId, { status: "ready", progress: 100 });
      setActiveTab("waveform");
      setExperiment((prev) => {
        if (!prev) return prev;
        return syncModalities({
          ...prev,
          eeg: {
            filename: file.name,
            format: "json",
            samplingRateHz: 160,
            channels: 64,
            autoDetected: true,
          },
        });
      });
    },
    [backendMode, patchFileProgress, uploadViaApi, baseForUpload],
  );

  const uploadFigure = useCallback(
    async (file: File) => {
      const err = validateFigure(file);
      if (err) {
        setUploadError(err);
        return;
      }
      setUploadError(null);
      const fileId = newLocalId("img");
      const blobUrl = URL.createObjectURL(file);
      const pending: ExperimentFile = {
        id: fileId,
        name: file.name,
        kind: "figure",
        sizeBytes: file.size,
        status: "uploading",
        progress: 0,
        url: blobUrl,
        mimeType: file.type,
      };

      // Image-only (or figure added with no EEG): never keep stale EEG-derived plots
      const cur = experimentRef.current;
      const hasReadyEeg = Boolean(
        cur?.eeg || cur?.eeg_files.some((f) => f.status === "ready"),
      );
      if (!hasReadyEeg) {
        setAnalysisResults(resetEegDerivedResults());
      }

      setExperiment((prev) => {
        const base = baseForUpload(prev);
        const images = [...base.image_files, pending];
        return syncModalities({
          ...base,
          isDemo: false,
          image_files: images,
          selected_image_id: base.selected_image_id ?? fileId,
        });
      });

      if (backendMode === "live") {
        try {
          patchFileProgress("figure", fileId, { progress: 40 });
          const prevSelected =
            experiment && !experiment.isDemo ? experiment.selected_image_id : null;
          const res = await uploadViaApi(file, "figure", fileId);
          URL.revokeObjectURL(blobUrl);
          setExperiment((prev) => {
            if (!prev) return prev;
            const keep =
              prevSelected && prev.image_files.some((f) => f.id === prevSelected)
                ? prevSelected
                : prev.selected_image_id ?? res.assetId;
            const next = syncModalities({ ...prev, selected_image_id: keep });
            const sel = next.image_files.find((f) => f.id === keep) ?? null;
            setVisionState((vs) => visionStateFromSelectedImage(vs, sel));
            return next;
          });
          // Do NOT force topomap tab / analysis slots — figure is vision-only.
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : "Figure upload failed";
          setUploadError(msg);
          patchFileProgress("figure", fileId, { status: "error", error: msg, progress: 0 });
        }
        return;
      }

      await simulateUploadProgress((p) => patchFileProgress("figure", fileId, { progress: p }));
      patchFileProgress("figure", fileId, { status: "ready", progress: 100 });
      setVisionState((vs) =>
        visionStateFromSelectedImage(vs, {
          id: fileId,
          name: file.name,
          kind: "figure",
          sizeBytes: file.size,
          status: "ready",
          url: blobUrl,
        }),
      );
    },
    [backendMode, patchFileProgress, uploadViaApi, experiment, baseForUpload],
  );

  const uploadMetadata = useCallback(
    async (file: File) => {
      const err = validateMetadata(file);
      if (err) {
        setUploadError(err);
        return;
      }
      setUploadError(null);
      const fileId = newLocalId("meta");
      const pending: ExperimentFile = {
        id: fileId,
        name: file.name,
        kind: "metadata",
        sizeBytes: file.size,
        status: "uploading",
        progress: 0,
        mimeType: file.type,
      };

      setExperiment((prev) => {
        const base = baseForUpload(prev);
        return syncModalities({
          ...base,
          isDemo: false,
          metadata_files: [...base.metadata_files, pending],
        });
      });

      if (backendMode === "live") {
        try {
          patchFileProgress("metadata", fileId, { progress: 40 });
          await uploadViaApi(file, "metadata", fileId);
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : "Metadata upload failed";
          setUploadError(msg);
          patchFileProgress("metadata", fileId, { status: "error", error: msg, progress: 0 });
        }
        return;
      }

      await simulateUploadProgress((p) => patchFileProgress("metadata", fileId, { progress: p }));
      patchFileProgress("metadata", fileId, { status: "ready", progress: 100 });
    },
    [backendMode, patchFileProgress, uploadViaApi, baseForUpload],
  );

  const selectImage = useCallback((imageId: string) => {
    setExperiment((prev) => {
      if (!prev) return prev;
      if (!prev.image_files.some((f) => f.id === imageId)) return prev;
      const next = syncModalities({ ...prev, selected_image_id: imageId });
      const sel = next.image_files.find((f) => f.id === imageId) ?? null;
      setVisionState((vs) => visionStateFromSelectedImage(vs, sel));
      return next;
    });
    // Tab click unchanged — selecting a figure must not rewrite analysis tabs.
  }, []);

  const clearImageSelection = useCallback(() => {
    setExperiment((prev) => {
      if (!prev) return prev;
      return syncModalities({ ...prev, selected_image_id: null });
    });
    setVisionState((vs) => visionStateFromSelectedImage(vs, null));
  }, []);

  const removeFile = useCallback((fileId: string) => {
    const prev = experimentRef.current;
    if (!prev) return;

    const removingEeg = prev.eeg_files.some((f) => f.id === fileId);
    const removingImg = prev.image_files.find((f) => f.id === fileId);
    if (removingImg?.url?.startsWith("blob:")) URL.revokeObjectURL(removingImg.url);

    const remainingEeg = prev.eeg_files.filter((f) => f.id !== fileId);
    const remainingImages = prev.image_files.filter((f) => f.id !== fileId);
    const eegFullyCleared =
      removingEeg && !remainingEeg.some((f) => f.status === "ready");

    let next: Experiment = {
      ...prev,
      eeg_files: remainingEeg,
      metadata_files: prev.metadata_files.filter((f) => f.id !== fileId),
      image_files: remainingImages,
    };

    if (removingEeg) {
      next.eeg = undefined;
    }
    if (prev.selected_image_id === fileId) {
      // Require explicit re-selection — do not auto-pick another image
      next.selected_image_id = null;
    }

    next = syncModalities(next);

    if (isExperimentEmpty(next)) {
      experimentRef.current = null;
      setExperiment(null);
      setAnalysisResults(emptyAnalysisResults());
      setVisionState(emptyVisionState());
      return;
    }

    experimentRef.current = next;
    setExperiment(next);

    // Last ready EEG removed → wipe EEG-derived explorer slots (no stale plots)
    if (eegFullyCleared) {
      setAnalysisResults(resetEegDerivedResults());
    }

    // Vision follows current selection only; removing selected image clears interpretation
    if (removingImg) {
      const sel =
        next.selected_image_id != null
          ? (next.image_files.find((f) => f.id === next.selected_image_id) ?? null)
          : null;
      setVisionState((vs) => visionStateFromSelectedImage(vs, sel));
    }
  }, []);

  const updateMetadata = useCallback((patch: Partial<Experiment["metadata"]>) => {
    setExperiment((prev) =>
      prev
        ? syncModalities({
            ...prev,
            metadata: { ...prev.metadata, ...patch },
          })
        : null,
    );
  }, []);

  const clearExperiment = useCallback(() => {
    revokeBlobUrls(experimentRef.current);
    setExperiment(null);
    setAnswers([]);
    setActiveAnswerId(null);
    setFocusedVizId(null);
    setActiveTab("waveform");
    setAnalysisResults(emptyAnalysisResults());
    setVisionState(emptyVisionState());
    setUploadError(null);
    setAnalysisError(null);
    setExplorerError(null);
    setIsAnalyzing(false);
    setWorkspaceEpoch((e) => e + 1);
  }, []);

  const simulateAnalysis = useCallback(
    async (question: string) => {
      setIsAnalyzing(true);
      setAnalysisError(null);
      setExplorerError(null);

      const q = question.trim();
      if (!q) {
        setAnalysisError("Enter a research question before analyzing.");
        setIsAnalyzing(false);
        return;
      }

      if (backendMode === "unavailable") {
        setAnalysisError(
          "Live API is unavailable. Start the backend and refresh — offline fixtures are disabled.",
        );
        setIsAnalyzing(false);
        return;
      }

      let exp = experimentRef.current;
      try {
        if (!exp || !(exp.experiment_id ?? exp.id)?.match(/^exp_/)) {
          exp = await ensureLiveExperiment();
        }
      } catch {
        setAnalysisError("Live API unavailable.");
        setIsAnalyzing(false);
        return;
      }
      if (!exp) {
        setAnalysisError("Live API unavailable — cannot create an experiment session.");
        setIsAnalyzing(false);
        return;
      }

      const expId = exp.experiment_id ?? exp.id;
      const imageId = explicitLiveImageId(exp);
      const selected =
        imageId ? exp.image_files.find((f) => f.id === imageId) ?? null : null;

      // Conversation history for multi-turn task planning (backend).
      const conversationHistory: Array<Record<string, unknown>> = [];
      for (const a of answersRef.current.slice(-6)) {
        conversationHistory.push({ role: "user", content: a.question });
        conversationHistory.push({
          role: "assistant",
          content: a.answer,
          tools_used: a.toolsUsed,
          route: a.route,
          evidence_summary: a.computedEvidence
            ?.slice(0, 8)
            .map((e) => `${e.label}=${e.value}`)
            .join("; "),
        });
      }

      let final: AgentAnswer;
      try {
        final = await analyzeLive({
          experimentId: expId,
          question: q,
          imageId,
          visualizationId: null,
          conversationHistory,
          context: imageId
            ? {
                has_image: true,
                selected_image_id: imageId,
                selected_image_name: selected?.name ?? null,
              }
            : undefined,
        });
        // Prefer backend provenance — never invent a different image id
        final = {
          ...final,
          isDemo: false,
          selectedImageId: final.selectedImageId ?? imageId,
          selectedImageName: final.selectedImageName ?? selected?.name ?? null,
        };
        try {
          const { metrics, source } = await fetchSystemMetricsWithFallback();
          if (source === "live") setLiveMetrics(metrics);
        } catch {
          /* ignore */
        }
      } catch (e) {
        setAnalysisError(humanizeAnalysisError(e));
        setIsAnalyzing(false);
        return;
      }

      const steps = final.timeline.length
        ? final.timeline
        : [
            {
              id: "t-done",
              name: "Synthesis",
              status: "complete" as const,
              latencyMs: final.timing.totalMs,
            },
          ];
      setActiveAnswerId(final.id);
      for (let i = 0; i < Math.min(steps.length, 4); i++) {
        await new Promise((r) => setTimeout(r, 40));
        setAnswers((prev) => {
          const others = prev.filter((a) => a.id !== final.id);
          return [
            ...others,
            {
              ...final,
              answer: i < steps.length - 1 ? "" : final.answer,
              modelInterpretation: "",
              timeline: progressiveTimeline(steps, i),
            },
          ];
        });
      }

      setAnswers((prev) => {
        const withoutDup = prev.filter((a) => a.id !== final.id);
        return [...withoutDup, final];
      });
      setActiveAnswerId(final.id);
      setIsAnalyzing(false);

      // Write into typed result slots — never cross-overwrite unrelated tabs.
      if (experienceMode === "workspace") {
        const sampleId =
          (exp.metadata as { sampleId?: string } | undefined)?.sampleId ?? null;
        setAnalysisResults((prev) =>
          applyAnswerToAnalysisResults(prev, final, {
            experimentId: expId,
            sampleId,
          }),
        );
        setVisionState((prev) => applyAnswerToVisionState(prev, final, selected));
      } else {
        // Chat: vision interpretation only; do not populate workspace explorer slots.
        setVisionState((prev) => applyAnswerToVisionState(prev, final, selected));
      }

      const historyItem: AnalysisHistoryItem = {
        id: newLocalId("hist"),
        question: final.question || q,
        route: final.route,
        timestamp: new Date().toISOString(),
        answerPreview: previewOf(final.answer),
        answer: final,
      };

      setExperiment((prev) => {
        if (!prev) return prev;
        const history = [historyItem, ...prev.analysis_history].slice(0, 20);
        return { ...prev, analysis_history: history };
      });

      if (settings.autoGenerateVisuals && experienceMode === "workspace") {
        // Switch view to the result type that was produced — not to a selected figure.
        if (final.route === "VISION") {
          /* keep current analysis tab; vision lives in attachment strip + agent answer */
        } else if (final.toolsUsed.some((t) => /rank|band/i.test(t))) {
          setActiveTab("band_power");
        } else if (final.toolsUsed.some((t) => /compar/i.test(t))) {
          setActiveTab("comparison");
        } else {
          const firstViz = final.visualEvidence[0];
          const tab = firstViz?.tab as VisualizationTab | undefined;
          if (tab && ["waveform", "psd", "spectrogram", "topomap", "band_power", "comparison"].includes(tab)) {
            setActiveTab(tab);
            setFocusedVizId(firstViz.id);
          }
        }
      }
    },
    [backendMode, settings.autoGenerateVisuals, ensureLiveExperiment, experienceMode],
  );

  const analyze = useCallback(
    async (question: string) => {
      if (!question.trim()) return;
      await simulateAnalysis(question);
    },
    [simulateAnalysis],
  );

  const runDemoAnalysis = useCallback(async () => {
    setAnalysisError("Use Chat or ask a question in the workspace — demo fixtures are disabled.");
  }, []);

  const restoreAnalysis = useCallback(
    (historyId: string) => {
      const item = experiment?.analysis_history.find((h) => h.id === historyId);
      if (!item?.answer) return;
      setAnswers((prev) => {
        if (prev.some((a) => a.id === item.answer.id)) return prev;
        return [...prev, item.answer];
      });
      setActiveAnswerId(item.answer.id);
      setAnalysisError(null);
    },
    [experiment?.analysis_history],
  );

  const focusVisualization = useCallback((id: string, tab?: VisualizationTab) => {
    setFocusedVizId(id);
    if (tab) setActiveTab(tab);
  }, []);

  const updateSettings = useCallback((patch: Partial<AppSettings>) => {
    setSettings((s) => ({ ...s, ...patch }));
  }, []);

  const runTool = useCallback(
    (toolName: string) => {
      void analyze(`Run ${toolName} on the current experiment.`);
    },
    [analyze],
  );

  const clearUploadError = useCallback(() => setUploadError(null), []);
  const clearAnalysisError = useCallback(() => setAnalysisError(null), []);

  const value = useMemo(
    () => ({
      experienceMode,
      experiment,
      answers,
      currentAnswer,
      activeAnswerId,
      isAnalyzing,
      activeTab,
      focusedVizId,
      settings,
      systemMetrics,
      precision,
      backendMode,
      experimentStatus,
      uploadError,
      analysisError,
      explorerError,
      explorerLoading,
      workspaceEpoch,
      selectedImage,
      analysisResults,
      visionState,
      setActiveTab,
      focusVisualization,
      beginChatSession,
      beginDemoSession,
      beginWorkspaceSession,
      loadDemo: loadWorkspaceDemoSample,
      uploadEEG,
      uploadFigure,
      uploadMetadata,
      removeFile,
      selectImage,
      clearImageSelection,
      updateMetadata,
      clearExperiment,
      analyze,
      runDemoAnalysis,
      restoreAnalysis,
      updateSettings,
      setPrecision,
      runTool,
      clearUploadError,
      clearAnalysisError,
      healthInfo,
      chatSessionId,
      workspaceExperimentId,
    }),
    [
      experienceMode,
      experiment,
      answers,
      currentAnswer,
      activeAnswerId,
      isAnalyzing,
      activeTab,
      focusedVizId,
      settings,
      systemMetrics,
      precision,
      backendMode,
      experimentStatus,
      uploadError,
      analysisError,
      explorerError,
      explorerLoading,
      workspaceEpoch,
      selectedImage,
      analysisResults,
      visionState,
      focusVisualization,
      beginChatSession,
      beginDemoSession,
      beginWorkspaceSession,
      loadWorkspaceDemoSample,
      uploadEEG,
      uploadFigure,
      uploadMetadata,
      removeFile,
      selectImage,
      clearImageSelection,
      updateMetadata,
      clearExperiment,
      analyze,
      runDemoAnalysis,
      restoreAnalysis,
      updateSettings,
      runTool,
      clearUploadError,
      clearAnalysisError,
      healthInfo,
      chatSessionId,
      workspaceExperimentId,
    ],
  );

  return (
    <ExperimentContext.Provider value={value}>{children}</ExperimentContext.Provider>
  );
}

export function useExperiment() {
  const ctx = useContext(ExperimentContext);
  if (!ctx) throw new Error("useExperiment must be used within ExperimentProvider");
  return ctx;
}
