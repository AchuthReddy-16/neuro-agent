"use client";

import { useExperiment } from "@/lib/store/experiment-context";
import { Collapsible } from "@/components/ui/Collapsible";
import type { SystemMetrics } from "@/lib/types";

const PRECISION_OPTIONS: SystemMetrics["precision"][] = ["BF16", "INT8", "INT8 W8A8", "INT4"];

function fmt(n: number | null | undefined, suffix = ""): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `${typeof n === "number" && !Number.isInteger(n) ? n.toFixed(1) : n}${suffix}`;
}

export function SystemPanel() {
  const { systemMetrics, precision, setPrecision, settings, backendMode, healthInfo } =
    useExperiment();

  if (!settings.showSystemMetrics) return null;

  return (
    <Collapsible title="System details" compact>
      <p className="text-[10px] text-muted mb-2.5 leading-relaxed">
        {backendMode === "live"
          ? "Connected to backend metrics. Null fields mean not yet measured — not zero."
          : backendMode === "unavailable"
            ? "Backend offline — showing demo metrics only."
            : "Demo metrics — not live inference."}
      </p>

      <dl className="space-y-1.5 text-[11px]">
        <MetricRow label="Text" value={systemMetrics.model} />
        <MetricRow label="Vision" value={systemMetrics.visionModel} />
        <MetricRow label="Serving" value={systemMetrics.serving} />
        <MetricRow label="Post-training" value={systemMetrics.postTraining} />
        {backendMode === "live" && (
          <>
            <MetricRow
              label="Agent loaded"
              value={healthInfo?.agentLoaded ? "yes" : "no"}
            />
            <MetricRow
              label="Vision loaded"
              value={healthInfo?.visionLoaded ? "yes" : "no"}
            />
          </>
        )}
      </dl>

      <div className="flex items-center justify-between gap-2 pt-2.5 mt-2.5 border-t border-default/60">
        <span className="text-[10px] text-muted">Precision</span>
        <div className="flex gap-1 flex-wrap justify-end">
          {PRECISION_OPTIONS.map((p) => (
            <button
              key={p}
              type="button"
              onClick={() => setPrecision(p)}
              aria-pressed={precision === p}
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono border transition-colors ${
                precision === p
                  ? "bg-accent/15 border-accent/40 text-accent"
                  : "border-transparent text-muted hover:text-secondary"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1 mt-2.5 pt-2.5 border-t border-default/60 text-[11px]">
        <MetricRow label="TTFT" value={fmt(systemMetrics.ttftMs, " ms")} />
        <MetricRow label="tok/s" value={fmt(systemMetrics.tokensPerSec)} />
        <MetricRow
          label="Last req"
          value={fmt(systemMetrics.lastRequestLatencyMs ?? systemMetrics.p95LatencyMs, " ms")}
        />
        <MetricRow label="GPU" value={fmt(systemMetrics.gpuUtilizationPct, "%")} />
      </div>
    </Collapsible>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted shrink-0">{label}</dt>
      <dd className="text-primary text-right font-mono text-[11px] truncate">{value}</dd>
    </div>
  );
}
