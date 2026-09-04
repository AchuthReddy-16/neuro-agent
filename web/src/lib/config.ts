/** Public API base URL for the FastAPI backend.

Browser traffic prefers same-origin `/api/*` (proxied by next.config.ts rewrites)
to avoid CORS issues between localhost and 127.0.0.1.
*/

export function getApiBaseUrl(): string {
  // Same-origin proxy in the browser
  if (typeof window !== "undefined") {
    return "";
  }
  const raw =
    process.env.NEXT_PUBLIC_API_BASE_URL ??
    process.env.NEXT_PUBLIC_API_BASE ??
    "http://127.0.0.1:8080";
  return raw.replace(/\/$/, "");
}

/** Resolve a backend-relative path (e.g. /api/visualization/…) for browser use. */
export function resolveApiUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined;
  if (/^https?:\/\//i.test(path) || path.startsWith("blob:") || path.startsWith("data:")) {
    return path;
  }
  // Keep relative so Next rewrites and <img> tags stay same-origin
  if (path.startsWith("/")) return path;
  return `/${path}`;
}
