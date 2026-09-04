"use client";

import clsx from "clsx";
import type { ReactNode } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { Chip } from "@/components/ui/Chip";
import { Collapsible } from "@/components/ui/Collapsible";
import { ToolTimeline, TimelinePipeline } from "./ToolTimeline";
import type { AgentAnswer, AnalysisRoute } from "@/lib/types";

function RouteBadge({ route }: { route: AnalysisRoute }) {
  const vision = route === "VISION";
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 text-[10px] font-mono",
        vision ? "text-signal-vision" : "text-signal-text",
      )}
    >
      <span
        className={clsx("w-1.5 h-1.5 rounded-full", vision ? "bg-signal-vision" : "bg-signal-text")}
        aria-hidden
      />
      {vision ? "VISION + TOOLS" : "TEXT + TOOLS"}
    </span>
  );
}

function SectionHead({
  children,
  aside,
  tone = "default",
}: {
  children: ReactNode;
  aside?: ReactNode;
  tone?: "default" | "computed" | "visual" | "model" | "warn" | "accent";
}) {
  const tones = {
    default: "text-muted",
    computed: "text-signal-eeg",
    visual: "text-signal-vision",
    model: "text-accent-secondary",
    warn: "text-signal-warning",
    accent: "text-accent",
  };
  return (
    <div className="flex items-baseline justify-between gap-2 mb-2">
      <h4 className={clsx("text-[10px] font-semibold tracking-[0.12em] uppercase", tones[tone])}>
        {children}
      </h4>
      {aside && <span className="text-[10px] text-muted shrink-0">{aside}</span>}
    </div>
  );
}

export function AgentResponseView({ answer }: { answer: AgentAnswer }) {
  const { settings, focusVisualization } = useExperiment();
  const computed = answer.computedEvidence?.length
    ? answer.computedEvidence
    : answer.evidence ?? [];
  const visual =
    answer.route === "VISION"
      ? (answer.visualEvidence?.length
          ? answer.visualEvidence
          : answer.visualRefs ?? []
        ).filter((v) => v.observation || v.vlm_interpretation || v.imageUrl)
      : [];

  const researchAnswer = (answer.answer || "")
    .replace(/^(Answer|Evidence|Tools used):\s*/gim, "")
    .trim();

  const uncertainty = (answer.uncertainty || "").trim();
  const showUncertainty =
    settings.showUncertainty &&
    uncertainty &&
    uncertainty.toLowerCase() !== "none" &&
    !uncertainty.startsWith("raw_model_output=") &&
    !uncertainty.includes("raw_model_output={");

  const interpretation = (answer.modelInterpretation || "").trim();
  const showInterpretation =
    interpretation &&
    interpretation !== researchAnswer &&
    !interpretation.startsWith(researchAnswer + "\n\nVision model:");

  const showVerification =
    answer.verification.status !== "skipped" ||
    Boolean(answer.verification.recoveryPerformed) ||
    Boolean(answer.verification.message);

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <RouteBadge route={answer.route} />
        {answer.isDemo && (
          <span className="text-[10px] text-muted">Offline fixture — not a live model run</span>
        )}
        {!answer.isDemo && (
          <span className="text-[10px] text-muted">Live API</span>
        )}
        {answer.timing.totalMs != null && (
          <span className="text-[10px] font-mono text-muted ml-auto tabular-nums">
            {answer.timing.totalMs} ms
          </span>
        )}
      </div>

      <section>
        <SectionHead tone="accent">Research Answer</SectionHead>
        {researchAnswer ? (
          <p className="text-[13px] text-primary leading-relaxed whitespace-pre-wrap">
            {researchAnswer}
          </p>
        ) : (
          <p className="text-[12px] text-muted leading-relaxed">
            No grounded answer text was returned for this request. See evidence and provenance below
            when available.
          </p>
        )}
      </section>

      {computed.length > 0 && (
        <section className="border-l-2 border-signal-eeg/50 pl-3">
          <SectionHead tone="computed" aside="Deterministic · tool-derived">
            Computed Evidence
          </SectionHead>
          <table className="w-full text-[12px]">
            <tbody>
              {computed.map((item, idx) => (
                <tr
                  key={`${item.label}-${idx}`}
                  className="border-b border-default/50 last:border-0"
                >
                  <td className="py-1.5 pr-3 text-muted align-top">
                    {item.label}
                    {item.tool && (
                      <span className="block text-[10px] text-muted/70 font-mono mt-0.5">
                        {item.tool}
                      </span>
                    )}
                  </td>
                  <td
                    className={clsx(
                      "py-1.5 text-right font-mono tabular-nums align-top",
                      item.highlight ? "text-signal-eeg font-semibold" : "text-primary",
                    )}
                  >
                    {item.value}
                    {item.unit && (
                      <span className="text-muted ml-1 font-sans text-[10px]">{item.unit}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {visual.length > 0 && (
        <section className="border-l-2 border-signal-vision/50 pl-3">
          <SectionHead
            tone="visual"
            aside={
              visual.some((v) => v.vlm_interpretation || v.observation)
                ? "Vision context"
                : "Image refs"
            }
          >
            Visual Evidence
          </SectionHead>
          <div className="space-y-2">
            {visual.map((ref) => (
              <button
                key={ref.id}
                type="button"
                onClick={() =>
                  focusVisualization(ref.id, ref.tab as Parameters<typeof focusVisualization>[1])
                }
                className="w-full text-left py-2 transition-colors hover:bg-signal-vision/5 rounded px-1 -mx-1"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[12px] font-medium text-signal-vision">{ref.label}</span>
                  {ref.tab &&
                    ref.label.toLowerCase() !== String(ref.tab).toLowerCase() && (
                      <span className="text-[10px] text-muted capitalize shrink-0">
                        {String(ref.tab).replace("_", " ")}
                      </span>
                    )}
                </div>
                {(ref.observation || ref.vlm_interpretation) && (
                  <p className="text-[11px] text-secondary mt-1 leading-relaxed italic">
                    {ref.observation || ref.vlm_interpretation}
                  </p>
                )}
              </button>
            ))}
          </div>
        </section>
      )}

      {showInterpretation && (
        <section className="border-l-2 border-accent-secondary/45 pl-3">
          <SectionHead tone="model" aside="Evidence reading">
            Model Interpretation
          </SectionHead>
          <p className="text-[12px] text-secondary leading-relaxed">{interpretation}</p>
        </section>
      )}

      {showUncertainty && (
        <section>
          <SectionHead tone="warn">Uncertainty / Limitations</SectionHead>
          <p className="text-[11px] text-muted leading-relaxed">{uncertainty}</p>
        </section>
      )}

      <div className="space-y-2 pt-1 border-t border-default/60">
        {answer.toolsUsed.length > 0 && (
          <Collapsible title="Tools / Provenance" compact>
            <div className="flex flex-wrap gap-1.5">
              {answer.toolsUsed.map((t) => (
                <Chip key={t} className="!rounded-md !text-[10px]">
                  {t}
                </Chip>
              ))}
            </div>
          </Collapsible>
        )}

        {showVerification && (
          <Collapsible title="Verification / Recovery" compact>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span
                className={clsx(
                  "font-mono text-[10px]",
                  answer.verification.status === "passed" && "text-accent",
                  (answer.verification.status === "triggered" ||
                    answer.verification.status === "recovered") &&
                    "text-signal-warning",
                  (answer.verification.status === "skipped" ||
                    answer.verification.status === "unavailable") &&
                    "text-muted",
                )}
              >
                {answer.verification.status}
              </span>
              {answer.verification.recoveryPerformed && (
                <span className="text-[10px] text-signal-warning">Recovery performed</span>
              )}
              {answer.verification.message && (
                <span className="text-[11px] text-secondary">{answer.verification.message}</span>
              )}
            </div>
          </Collapsible>
        )}

        {settings.showRawToolOutput && answer.rawToolOutput && (
          <Collapsible title="Raw tool output" compact>
            <pre className="text-[10px] font-mono bg-muted/40 p-2 rounded overflow-x-auto text-secondary">
              {answer.rawToolOutput}
            </pre>
          </Collapsible>
        )}

        {answer.timeline.length > 0 && (
          <Collapsible title="Execution timeline" compact defaultOpen={false}>
            <div className="mb-3 overflow-x-auto pb-1">
              <TimelinePipeline steps={answer.timeline} />
            </div>
            <ToolTimeline steps={answer.timeline} />
          </Collapsible>
        )}
      </div>
    </div>
  );
}
