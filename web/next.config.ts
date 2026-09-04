import type { NextConfig } from "next";

/**
 * Proxy /api/* → FastAPI.
 * Destination is resolved at build time from NEXT_PUBLIC_API_BASE_URL
 * (set in Vercel project env for production).
 *
 * Production must set NEXT_PUBLIC_API_BASE_URL explicitly — there is no
 * silent localhost fallback (avoids shipping a broken same-origin proxy).
 */
const apiBase =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  (process.env.NODE_ENV === "production" ? "" : "http://127.0.0.1:8080");

if (process.env.NODE_ENV === "production" && !apiBase) {
  console.warn(
    "[neuro-agent] NEXT_PUBLIC_API_BASE_URL is unset in production. " +
      "Set it to the public GPU FastAPI origin before building.",
  );
}

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    const dest = apiBase.replace(/\/$/, "");
    if (!dest) {
      // Production misconfig — skip rewrite rather than proxying to localhost
      return [];
    }
    return [
      {
        source: "/api/:path*",
        destination: `${dest}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
