"use client";

import clsx from "clsx";
import { useExperiment } from "@/lib/store/experiment-context";

export function SystemStatusStrip() {
  const { systemMetrics, backendMode, currentAnswer, isAnalyzing, settings, healthInfo } =
    useExperiment();

  if (!settings.showSystemMetrics) return null;

  const route = currentAnswer?.route ?? systemMetrics.route ?? null;
  const routeLabel =
    route == null ? "—" : route === "VISION" ? "VISION + TOOLS" : "TEXT + TOOLS";
  const latency =
    currentAnswer?.timing.totalMs ??
    (isAnalyzing
      ? undefined
      : systemMetrics.lastRequestLatencyMs ?? systemMetrics.p95LatencyMs ?? undefined);
  const verifier =
    currentAnswer?.verification.status ?? systemMetrics.verifierStatus ?? "—";

  const items: { key: string; label: string; value: string; accent?: string }[] = [
    { key: "text", label: "Text", value: systemMetrics.model },
    { key: "prec", label: "Precision", value: systemMetrics.precision },
    { key: "vision", label: "Vision", value: systemMetrics.visionModel },
    {
      key: "route",
      label: "Route",
      value: isAnalyzing ? "…" : routeLabel,
      accent:
        route === "VISION"
          ? "text-signal-vision"
          : route === "TEXT"
            ? "text-signal-text"
            : undefined,
    },
    {
      key: "lat",
      label: "Latency",
      value: latency != null ? `${Math.round(latency)} ms` : isAnalyzing ? "…" : "—",
    },
    {
      key: "ver",
      label: "Verifier",
      value: String(verifier),
      accent:
        verifier === "passed"
          ? "text-accent"
          : verifier === "triggered" || verifier === "recovered"
            ? "text-signal-warning"
            : undefined,
    },
  ];

  if (backendMode === "live" && healthInfo && healthInfo.agentLoaded === false) {
    items.push({
      key: "agent",
      label: "Status",
      value: "Preparing research model…",
      accent: "text-signal-warning",
    });
  }

  const modeLabel =
    backendMode === "live" ? "Live" : backendMode === "unavailable" ? "Offline" : "Offline";

  return (
    <div
      className="shrink-0 border-b border-default bg-surface"
      role="status"
      aria-label="System status"
    >
      <div className="flex items-center justify-between gap-3 px-4 py-1 overflow-x-auto">
        <div className="flex items-center gap-0 text-[10px] font-mono min-w-0">
          {items.map((item, i) => (
            <div
              key={item.key}
              className={clsx(
                "flex items-center gap-1.5 px-2.5 py-0.5 whitespace-nowrap",
                i > 0 && "border-l border-default/60",
              )}
            >
              <span className="text-muted/80 uppercase tracking-wider text-[9px]">
                {item.label}
              </span>
              <span className={clsx("text-secondary", item.accent)}>{item.value}</span>
            </div>
          ))}
        </div>
        <span
          className={clsx(
            "text-[9px] uppercase tracking-wider shrink-0",
            backendMode === "live"
              ? "text-accent"
              : backendMode === "unavailable"
                ? "text-signal-warning"
                : "text-muted",
          )}
        >
          {modeLabel}
        </span>
      </div>
    </div>
  );
}
