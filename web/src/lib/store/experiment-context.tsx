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
  fetchSystemMetricsWithFallback,
  getExperiment,
  uploadAsset,
} from "@/lib/api";
import { mergeUploadResponse, mapExperimentFromApi } from "@/lib/experiment-map";
import { inferNeedsVision, resolveSelectedImage } from "@/lib/routing";
import {
  createDemoAgentAnswer,
  createMockAnswer,
  progressiveTimeline,
} from "@/lib/mock/responses";
import {
  createDemoExperiment,
  EEG_SUPPORTED,
  FIGURE_SUPPORTED,
  METADATA_SUPPORTED,
  MOCK_SYSTEM_METRICS,
} from "@/lib/constants";

interface ExperimentContextValue {
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
  setActiveTab: (tab: VisualizationTab) => void;
  focusVisualization: (id: string, tab?: VisualizationTab) => void;
  loadDemo: () => void | Promise<void>;
  uploadEEG: (file: File) => Promise<void>;
  uploadFigure: (file: File) => Promise<void>;
  uploadMetadata: (file: File) => Promise<void>;
  removeFile: (fileId: string) => void;
  selectImage: (imageId: string) => void;
  clearImageSelection: () => void;
  updateMetadata: (patch: Partial<Experiment["metadata"]>) => void;
  clearExperiment: () => void;
  analyze: (question: string) => Promise<void>;
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
  const [backendMode, setBackendMode] = useState<BackendMode>("demo");
  const [healthInfo, setHealthInfo] = useState<HealthResponse | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [explorerError, setExplorerError] = useState<string | null>(null);
  const [explorerLoading, setExplorerLoading] = useState(false);
  const [liveMetrics, setLiveMetrics] = useState<SystemMetrics | null>(null);
  const [workspaceEpoch, setWorkspaceEpoch] = useState(0);
  const experimentRef = useRef<Experiment | null>(null);
  experimentRef.current = experiment;

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
        setBackendMode((prev) => (prev === "unavailable" ? "live" : prev === "demo" ? "demo" : "live"));
        try {
          const { metrics, source } = await fetchSystemMetricsWithFallback();
          if (!cancelled && source === "live") setLiveMetrics(metrics);
        } catch {
          /* ignore poll errors */
        }
      } else if (!cancelled) {
        setBackendMode((prev) => (prev === "demo" ? "demo" : "unavailable"));
      }
    }, 15000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const loadDemo = useCallback(async () => {
    setUploadError(null);
    setAnalysisError(null);
    setExplorerError(null);
    setExplorerLoading(true);
    setAnswers([]);
    setActiveAnswerId(null);
    setWorkspaceEpoch((e) => e + 1);

    try {
      const health = await checkHealth();
      if (health.status === "ok" || health.status === "degraded") {
        setBackendMode("live");
        setHealthInfo(health);
        const raw = await getExperiment("exp_demo_s001");
        const mapped = mapExperimentFromApi(raw);
        setExperiment(mapped);
        setActiveTab("topomap");
        setFocusedVizId(mapped.visualizations[0]?.id ?? null);
        setExplorerLoading(false);
        return;
      }
    } catch {
      /* fall through to local demo */
    }

    setBackendMode((m) => (m === "live" ? "unavailable" : "demo"));
    setExperiment(createDemoExperiment());
    setActiveTab("topomap");
    setFocusedVizId("viz-topomap-01");
    setExplorerLoading(false);
  }, []);

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
      // Only reuse real backend experiment ids (exp_…); local temp ids (exp-…) are not sent
      const backendExpId = expId && /^exp_[a-zA-Z0-9]+/.test(expId) ? expId : null;
      const res = await uploadAsset(
        {
          fileType,
          filename: file.name,
          contentType: file.type,
          experimentId: backendExpId,
        },
        file,
      );
      const cleaned = current
        ? {
            ...current,
            eeg_files: current.eeg_files.filter((f) => f.id !== localPendingId),
            metadata_files: current.metadata_files.filter((f) => f.id !== localPendingId),
            image_files: current.image_files.filter((f) => f.id !== localPendingId),
          }
        : null;
      const merged = mergeUploadResponse(cleaned, res, {
        name: file.name,
        kind: fileType,
        sizeBytes: file.size,
      });
      experimentRef.current = merged;
      setExperiment(merged);
      return res;
    },
    [],
  );

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
        const base = prev ?? emptyExperiment();
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
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : "EEG upload failed";
          setUploadError(msg);
          patchFileProgress("eeg", fileId, { status: "error", error: msg, progress: 0 });
        }
        return;
      }

      await simulateUploadProgress((p) => patchFileProgress("eeg", fileId, { progress: p }));
      patchFileProgress("eeg", fileId, { status: "ready", progress: 100 });
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
    [backendMode, patchFileProgress, uploadViaApi],
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

      setExperiment((prev) => {
        const base = prev ?? emptyExperiment();
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
          const prevSelected = experiment?.selected_image_id;
          const res = await uploadViaApi(file, "figure", fileId);
          URL.revokeObjectURL(blobUrl);
          setExperiment((prev) => {
            if (!prev) return prev;
            // Preserve prior selection when adding additional figures
            const keep =
              prevSelected && prev.image_files.some((f) => f.id === prevSelected)
                ? prevSelected
                : prev.selected_image_id ?? res.assetId;
            return syncModalities({ ...prev, selected_image_id: keep });
          });
          setActiveTab("topomap");
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : "Figure upload failed";
          setUploadError(msg);
          patchFileProgress("figure", fileId, { status: "error", error: msg, progress: 0 });
        }
        return;
      }

      await simulateUploadProgress((p) => patchFileProgress("figure", fileId, { progress: p }));
      patchFileProgress("figure", fileId, { status: "ready", progress: 100 });
      setActiveTab("topomap");
    },
    [backendMode, patchFileProgress, uploadViaApi, experiment?.selected_image_id],
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
        const base = prev ?? emptyExperiment();
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
    [backendMode, patchFileProgress, uploadViaApi],
  );

  const selectImage = useCallback((imageId: string) => {
    setExperiment((prev) => {
      if (!prev) return prev;
      if (!prev.image_files.some((f) => f.id === imageId)) return prev;
      return syncModalities({ ...prev, selected_image_id: imageId });
    });
    setActiveTab("topomap");
  }, []);

  const clearImageSelection = useCallback(() => {
    setExperiment((prev) => {
      if (!prev) return prev;
      return syncModalities({ ...prev, selected_image_id: null });
    });
  }, []);

  const removeFile = useCallback((fileId: string) => {
    setExperiment((prev) => {
      if (!prev) return prev;
      const removedImg = prev.image_files.find((f) => f.id === fileId);
      if (removedImg?.url?.startsWith("blob:")) URL.revokeObjectURL(removedImg.url);

      let next: Experiment = {
        ...prev,
        eeg_files: prev.eeg_files.filter((f) => f.id !== fileId),
        metadata_files: prev.metadata_files.filter((f) => f.id !== fileId),
        image_files: prev.image_files.filter((f) => f.id !== fileId),
      };

      if (prev.eeg_files.some((f) => f.id === fileId)) {
        next.eeg = undefined;
      }
      if (prev.selected_image_id === fileId) {
        next.selected_image_id = next.image_files[0]?.id ?? null;
      }

      next = syncModalities(next);
      if (isExperimentEmpty(next)) return null;
      return next;
    });
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
    setExperiment((prev) => {
      prev?.image_files.forEach((f) => {
        if (f.url?.startsWith("blob:")) URL.revokeObjectURL(f.url);
      });
      return null;
    });
    setAnswers([]);
    setActiveAnswerId(null);
    setFocusedVizId(null);
    setActiveTab("waveform");
    setUploadError(null);
    setAnalysisError(null);
    setExplorerError(null);
    setIsAnalyzing(false);
    setWorkspaceEpoch((e) => e + 1);
  }, []);

  const simulateAnalysis = useCallback(
    async (question: string, forceLocalDemo = false) => {
      setIsAnalyzing(true);
      setAnalysisError(null);
      setExplorerError(null);

      const q = question.trim();
      if (!q) {
        setAnalysisError("Enter a research question before analyzing.");
        setIsAnalyzing(false);
        return;
      }

      const needsVision = forceLocalDemo ? true : inferNeedsVision(q);
      let selected: ExperimentFile | null = null;

      if (needsVision && !forceLocalDemo) {
        const images = experiment?.image_files ?? [];
        if (images.length > 0) {
          const resolved = resolveSelectedImage(images, experiment?.selected_image_id ?? null);
          if (!resolved.ok) {
            setAnalysisError(resolved.reason);
            setIsAnalyzing(false);
            return;
          }
          selected = resolved.image;
          if (selected && experiment && experiment.selected_image_id !== selected.id) {
            setExperiment((prev) =>
              prev ? syncModalities({ ...prev, selected_image_id: selected!.id }) : prev,
            );
          }
        } else if (!(backendMode === "live" && experiment?.isDemo)) {
          setAnalysisError("Vision analysis needs a figure. Upload an image first.");
          setIsAnalyzing(false);
          return;
        }
      }

      const useLive = backendMode === "live" && !forceLocalDemo;
      let final: AgentAnswer;

      try {
        if (useLive) {
          const expId = experiment?.experiment_id ?? experiment?.id;
          if (!expId || !/^exp_/.test(expId)) {
            setAnalysisError(
              "No backend experiment yet — upload metadata/figures (or load the live demo) first.",
            );
            setIsAnalyzing(false);
            return;
          }
          final = await analyzeLive({
            experimentId: expId,
            question: q,
            imageId: selected?.id ?? experiment?.selected_image_id ?? null,
            context: needsVision ? { requires_vision: true } : undefined,
          });
          final = {
            ...final,
            selectedImageId: selected?.id ?? experiment?.selected_image_id ?? null,
            selectedImageName: selected?.name ?? null,
          };
          // Refresh metrics after live analyze
          try {
            const { metrics, source } = await fetchSystemMetricsWithFallback();
            if (source === "live") setLiveMetrics(metrics);
          } catch {
            /* ignore */
          }
        } else if (forceLocalDemo) {
          final = createDemoAgentAnswer();
        } else {
          final = createMockAnswer(q, {
            selectedImage: needsVision
              ? selected
                ? { id: selected.id, name: selected.name, url: selected.url }
                : experiment?.figure
                  ? {
                      id: experiment.figure.id,
                      name: experiment.figure.filename,
                      url: experiment.figure.url,
                    }
                  : null
              : null,
          });
        }
      } catch (e) {
        if (useLive) {
          const msg =
            e instanceof ApiError
              ? e.message
              : "Analysis request failed. Check backend logs.";
          setAnalysisError(msg);
          setIsAnalyzing(false);
          return;
        }
        final = createMockAnswer(q, {
          selectedImage: selected
            ? { id: selected.id, name: selected.name, url: selected.url }
            : null,
        });
      }

      // Local demo / offline: enforce frontend TEXT routing when tools-only
      if (!useLive && !needsVision) {
        final = {
          ...final,
          route: "TEXT",
          visualEvidence: [],
          visualRefs: [],
          system: { ...final.system, route: "TEXT" },
          selectedImageId: null,
          selectedImageName: null,
          timeline: final.timeline.map((s) =>
            s.name === "Vision analysis" ? { ...s, status: "skipped", latencyMs: undefined } : s,
          ),
        };
      }

      const steps = final.timeline.length
        ? final.timeline
        : [
            { id: "t-done", name: "Synthesis", status: "complete" as const, latencyMs: final.timing.totalMs },
          ];
      setActiveAnswerId(final.id);
      const stepDelay = useLive ? 80 : forceLocalDemo ? 380 : 260;
      for (let i = 0; i < steps.length; i++) {
        await new Promise((r) => setTimeout(r, stepDelay));
        setAnswers([
          {
            ...final,
            answer: "",
            modelInterpretation: "",
            timeline: progressiveTimeline(steps, i),
          },
        ]);
      }

      setAnswers((prev) => {
        const withoutDup = prev.filter((a) => a.id !== final.id);
        return [...withoutDup, final];
      });
      setActiveAnswerId(final.id);
      setIsAnalyzing(false);

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

      if (settings.autoGenerateVisuals && experiment) {
        if (final.route === "VISION" && (selected?.url || experiment.figure?.url)) {
          setActiveTab("topomap");
        } else {
          const firstViz = final.visualEvidence[0];
          if (firstViz) {
            setActiveTab(firstViz.tab as VisualizationTab);
            setFocusedVizId(firstViz.id);
          }
        }
      }
    },
    [experiment, settings.autoGenerateVisuals, backendMode],
  );

  const analyze = useCallback(
    async (question: string) => {
      if (!question.trim()) return;
      if (!experiment) {
        setAnalysisError("Load or create an experiment before analyzing.");
        return;
      }
      await simulateAnalysis(question);
    },
    [simulateAnalysis, experiment],
  );

  const runDemoAnalysis = useCallback(async () => {
    if (!experiment) await loadDemo();
    // When live, prefer real API on the demo experiment (not local mock fixtures)
    if (backendMode === "live") {
      await simulateAnalysis(createDemoAgentAnswer().question, false);
    } else {
      await simulateAnalysis(createDemoAgentAnswer().question, true);
    }
  }, [experiment, loadDemo, simulateAnalysis, backendMode]);

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
      setActiveTab,
      focusVisualization,
      loadDemo,
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
    }),
    [
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
      focusVisualization,
      loadDemo,
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
