/** Public API base URL for the FastAPI backend.

Vercel / production:
  Set NEXT_PUBLIC_API_BASE_URL to the deployed GPU API origin
  (e.g. https://api.example.com). Do not hardcode localhost.

Development:
  Defaults to http://127.0.0.1:8080 for Next.js rewrites.
  Browser requests use same-origin `/api/*` which next.config.ts
  proxies to that base (avoids CORS between localhost variants).

Aliases: NEXT_PUBLIC_API_BASE_URL (preferred), NEXT_PUBLIC_API_BASE.
*/

export function getServerApiBaseUrl(): string {
  const raw =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    process.env.NEXT_PUBLIC_API_BASE ??
    (process.env.NODE_ENV === "production" ? "" : "http://127.0.0.1:8080");
  return raw.replace(/\/$/, "");
}

/** Browser: same-origin proxy. Server: absolute backend base. */
export function getApiBaseUrl(): string {
  if (typeof window !== "undefined") {
    return "";
  }
  return getServerApiBaseUrl();
}

/** Resolve a backend-relative path (e.g. /api/visualization/…) for browser use. */
export function resolveApiUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  if (/^https?:\/\//i.test(path) || path.startsWith("blob:") || path.startsWith("data:")) {
    return path;
  }
  if (path.startsWith("/")) return path;
  return `/${path}`;
}

/** Human-readable backend readiness from health payload. */
export function describeBackendStatus(health: {
  status?: string;
  agentLoaded?: boolean;
  agent_loaded?: boolean;
  visionLoaded?: boolean;
  vision_loaded?: boolean;
} | null): string {
  if (!health || health.status === "unavailable") {
    return "Backend unavailable";
  }
  const agentLoaded = health.agentLoaded ?? health.agent_loaded;
  if (health.status === "degraded" || agentLoaded === false) {
    return "Live — preparing research model…";
  }
  if (health.status === "ok") {
    return "Live — backend ready";
  }
  return "Connecting…";
}
