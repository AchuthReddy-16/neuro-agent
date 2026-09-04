import type {
  AnalyzeRequest,
  AnalyzeResponse,
  GetSystemMetricsResponse,
  HealthResponse,
  SystemMetrics,
  UploadRequest,
  UploadResponse,
} from "./types";
import { MOCK_SYSTEM_METRICS } from "./constants";
import {
  analyzeResponseToAgentAnswer,
  createDemoAnalyzeResponse,
  createMockAnswer,
} from "./mock/responses";
import type { AgentAnswer } from "./types";
import { getApiBaseUrl } from "./config";
import type { ApiExperimentPayload, ApiUploadResponse } from "./experiment-map";

const API_BASE = getApiBaseUrl();

export class ApiError extends Error {
  status: number;
  path: string;
  code?: string;

  constructor(status: number, path: string, message?: string, code?: string) {
    super(message ?? `API error ${status}: ${path}`);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
    this.code = code;
  }
}

async function parseError(res: Response, path: string): Promise<ApiError> {
  let message = `Request failed (${res.status})`;
  let code: string | undefined;
  try {
    const body = (await res.json()) as {
      error?: string;
      detail?: string;
      code?: string;
    };
    code = body.code ?? body.error;
    if (body.detail) message = body.detail;
    else if (body.error) message = body.error;
  } catch {
    /* ignore non-JSON */
  }
  return new ApiError(res.status, path, message, code);
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    throw await parseError(res, path);
  }
  return res.json() as Promise<T>;
}

/** Probe whether the FastAPI backend is reachable */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    return await apiFetch<HealthResponse>("/api/health", {
      method: "GET",
      signal: AbortSignal.timeout(8000),
    });
  } catch {
    return { status: "unavailable" };
  }
}

/** GET /api/health */
export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/health");
}

/** POST /api/upload — multipart */
export async function uploadAsset(
  req: UploadRequest & { experimentId?: string | null },
  file: File,
): Promise<UploadResponse & ApiUploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("fileType", req.fileType);
  form.append("filename", req.filename || file.name);
  if (req.experimentId) {
    form.append("experiment_id", req.experimentId);
  }
  return apiFetch<UploadResponse & ApiUploadResponse>("/api/upload", {
    method: "POST",
    body: form,
  });
}

/** POST /api/analyze */
export function analyzeExperiment(req: AnalyzeRequest): Promise<AnalyzeResponse> {
  const body = {
    experimentId: req.experimentId,
    question: req.question,
    imageId: req.imageId ?? req.selectedImageId ?? null,
    visualizationId: req.visualizationId ?? null,
    tools: req.tools,
    settings: req.settings,
    context: req.context ?? null,
    conversationHistory: req.conversationHistory ?? null,
  };
  // Cold VLM load can exceed 60s; keep a hard upper bound (matches NEURO_API_VLM_TIMEOUT_S).
  return apiFetch<AnalyzeResponse>("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(300_000),
  });
}

/** POST /api/experiment — create empty live session */
export function createExperiment(): Promise<ApiExperimentPayload> {
  return apiFetch<ApiExperimentPayload>("/api/experiment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
}

/** GET /api/experiment/{id} */
export function getExperiment(id: string): Promise<ApiExperimentPayload> {
  return apiFetch<ApiExperimentPayload>(`/api/experiment/${id}`);
}

/** GET /api/visualization/{id} as JSON metadata */
export function getVisualization(id: string): Promise<{
  id: string;
  tab: string;
  title: string;
  imageUrl?: string;
  index?: number;
}> {
  return apiFetch(`/api/visualization/${id}`, {
    headers: { Accept: "application/json" },
  });
}

/** GET /api/system/metrics */
export function getSystemMetrics(): Promise<GetSystemMetricsResponse> {
  return apiFetch<SystemMetrics>("/api/system/metrics");
}

/**
 * Live analyze only. Throws ApiError on failure — callers must not silently mock when live.
 */
export async function analyzeLive(
  req: AnalyzeRequest,
): Promise<AgentAnswer> {
  const res = await analyzeExperiment(req);
  return analyzeResponseToAgentAnswer(res, {
    isDemo: false,
    question: req.question,
  });
}

/**
 * Demo/offline fallback only. Prefer analyzeLive when backendMode === "live".
 */
export async function analyzeWithFallback(
  req: AnalyzeRequest,
  opts?: { preferDemo?: boolean },
): Promise<{ answer: AgentAnswer; source: "live" | "demo" }> {
  if (!opts?.preferDemo) {
    try {
      const health = await checkHealth();
      if (health.status === "ok" || health.status === "degraded") {
        const answer = await analyzeLive(req);
        return { answer, source: "live" };
      }
    } catch (err) {
      // When caller asked for live, rethrow — do not swallow into mock.
      if (err instanceof ApiError) throw err;
      throw err;
    }
  }

  const mock =
    opts?.preferDemo || !req.question
      ? analyzeResponseToAgentAnswer(createDemoAnalyzeResponse(), {
          isDemo: true,
          question: req.question,
        })
      : createMockAnswer(req.question);

  return { answer: mock, source: "demo" };
}

export async function fetchSystemMetricsWithFallback(): Promise<{
  metrics: SystemMetrics;
  source: "live" | "demo";
}> {
  try {
    const health = await checkHealth();
    if (health.status === "ok" || health.status === "degraded") {
      const metrics = await getSystemMetrics();
      return { metrics, source: "live" };
    }
  } catch {
    /* mock */
  }
  return { metrics: MOCK_SYSTEM_METRICS, source: "demo" };
}

export { API_BASE };
