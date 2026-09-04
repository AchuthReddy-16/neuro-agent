"use client";

import clsx from "clsx";
import type { TimelineStage } from "@/lib/types";

export function ToolTimeline({
  steps,
  active,
  compact,
}: {
  steps: TimelineStage[];
  active?: boolean;
  compact?: boolean;
}) {
  return (
    <div
      className={clsx("space-y-0", compact && "space-y-0")}
      role="list"
      aria-label="Execution timeline"
    >
      {steps.map((step, i) => {
        const skipped = step.status === "skipped";
        return (
          <div key={step.id} className="flex gap-3" role="listitem">
            <div className="flex flex-col items-center">
              <div
                className={clsx(
                  "w-5 h-5 rounded-full border-2 shrink-0 flex items-center justify-center text-[9px]",
                  step.status === "complete" && "bg-accent/20 border-accent text-accent",
                  step.status === "running" &&
                    active &&
                    "border-accent bg-accent/10 animate-pulse text-accent",
                  step.status === "running" && !active && "border-accent/50 bg-transparent",
                  step.status === "pending" && "border-default bg-transparent text-transparent",
                  step.status === "error" &&
                    "bg-signal-error/20 border-signal-error text-signal-error",
                  skipped && "border-default/60 bg-transparent text-muted",
                )}
                aria-hidden
              >
                {step.status === "complete" && "✓"}
                {skipped && "–"}
              </div>
              {i < steps.length - 1 && (
                <div
                  className={clsx(
                    "w-px flex-1 min-h-[16px] my-0.5",
                    step.status === "complete"
                      ? "bg-accent/35"
                      : skipped
                        ? "bg-border/50"
                        : "bg-border",
                  )}
                />
              )}
            </div>
            <div className="pb-2.5 flex-1 min-w-0">
              <div className="flex items-baseline justify-between gap-2">
                <p
                  className={clsx(
                    "text-xs font-medium",
                    step.status === "complete" && "text-primary",
                    step.status === "running" && active && "text-accent",
                    step.status === "pending" && "text-muted",
                    skipped && "text-muted line-through decoration-border",
                    step.status === "error" && "text-signal-error",
                  )}
                >
                  {step.name}
                </p>
                <span className="text-[10px] font-mono text-muted tabular-nums shrink-0">
                  {skipped
                    ? "skipped"
                    : step.latencyMs != null && step.status === "complete"
                      ? `${step.latencyMs} ms`
                      : step.status === "running"
                        ? "…"
                        : ""}
                </span>
              </div>
              {step.summary && !compact && (
                <p className="text-[10px] text-muted mt-0.5 leading-relaxed">{step.summary}</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Compact horizontal pipeline for status awareness */
export function TimelinePipeline({ steps }: { steps: TimelineStage[] }) {
  return (
    <div
      className="flex flex-wrap items-center gap-1 text-[10px] font-mono"
      role="list"
      aria-label="Pipeline stages"
    >
      {steps.map((step, i) => (
        <span key={step.id} className="inline-flex items-center gap-1" role="listitem">
          {i > 0 && <span className="text-muted/60 mx-0.5">→</span>}
          <span
            className={clsx(
              "px-1.5 py-0.5 rounded border",
              step.status === "complete" && "border-accent/35 text-accent bg-accent/8",
              step.status === "running" && "border-accent text-accent bg-accent/10 animate-pulse",
              step.status === "pending" && "border-default text-muted",
              step.status === "skipped" && "border-default/50 text-muted/70",
              step.status === "error" && "border-signal-error/40 text-signal-error",
            )}
          >
            {step.name}
          </span>
        </span>
      ))}
    </div>
  );
}
