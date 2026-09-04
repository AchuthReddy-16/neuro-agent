"use client";

import clsx from "clsx";
import type { ReactNode } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { Chip } from "@/components/ui/Chip";
import { Collapsible } from "@/components/ui/Collapsible";
import { ToolTimeline, TimelinePipeline } from "./ToolTimeline";
import type { AgentAnswer } from "@/lib/types";

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

function meaningfulUncertainty(raw: string | undefined): string | null {
  const u = (raw || "").trim();
  if (!u) return null;
  if (u.toLowerCase() === "none" || u.toLowerCase() === "n/a") return null;
  if (u.startsWith("raw_model_output=") || u.includes("raw_model_output={")) return null;
  return u;
}

/**
 * Task-aware answer view.
 * - chat: answer-first; engineering details collapsed
 * - workspace: same adaptive sections; richer defaults when evidence exists
 */
export function AgentResponseView({
  answer,
  variant = "workspace",
}: {
  answer: AgentAnswer;
  variant?: "chat" | "workspace";
}) {
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
    .replace(/\nUncertainty:\s*.+$/im, "")
    .trim();

  const uncertainty = settings.showUncertainty
    ? meaningfulUncertainty(answer.uncertainty)
    : null;

  const interpretation = (answer.modelInterpretation || "").trim();
  const showInterpretation =
    interpretation &&
    interpretation !== researchAnswer &&
    !interpretation.startsWith(researchAnswer);

  const toolsRan = answer.toolsUsed.length > 0;
  const verifyMeaningful =
    answer.verification.status === "passed" ||
    answer.verification.status === "triggered" ||
    answer.verification.status === "recovered" ||
    Boolean(answer.verification.recoveryPerformed);

  const isSimple =
    !toolsRan &&
    visual.length === 0 &&
    computed.length === 0 &&
    !showInterpretation &&
    !verifyMeaningful;

  const showResearchLabel = !isSimple && (toolsRan || visual.length > 0 || computed.length > 0);

  const hasDetails =
    toolsRan ||
    verifyMeaningful ||
    (answer.timeline?.length ?? 0) > 0 ||
    Boolean(settings.showRawToolOutput && answer.rawToolOutput) ||
    Boolean(answer.routeDetail);

  return (
    <div className="space-y-4 animate-fade-in">
      <section>
        {showResearchLabel && <SectionHead tone="accent">Research Answer</SectionHead>}
        {researchAnswer ? (
          <p
            className={clsx(
              "text-primary leading-relaxed whitespace-pre-wrap",
              variant === "chat" ? "text-[14px]" : "text-[13px]",
            )}
          >
            {researchAnswer}
          </p>
        ) : (
          <p className="text-[12px] text-muted leading-relaxed">
            No answer text was returned for this request.
          </p>
        )}
        {(answer.visionUsed || answer.route === "VISION") && answer.selectedImageName && (
          <p className="mt-2 text-[11px] text-signal-vision font-mono">
            Analyzed image: {answer.selectedImageName}
          </p>
        )}
      </section>

      {computed.length > 0 && (
        <section className="border-l-2 border-signal-eeg/50 pl-3">
          <SectionHead tone="computed">Computed Evidence</SectionHead>
          <table className="w-full text-[12px]">
            <tbody>
              {computed.map((item, idx) => (
                <tr
                  key={`${item.label}-${idx}`}
                  className="border-b border-default/50 last:border-0"
                >
                  <td className="py-1.5 pr-3 text-muted align-top">{item.label}</td>
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
          <SectionHead tone="visual">Visual Evidence</SectionHead>
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
                <span className="text-[12px] font-medium text-signal-vision">{ref.label}</span>
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
          <SectionHead tone="model">Interpretation</SectionHead>
          <p className="text-[12px] text-secondary leading-relaxed">{interpretation}</p>
        </section>
      )}

      {uncertainty && (
        <section>
          <SectionHead tone="warn">Limitations</SectionHead>
          <p className="text-[11px] text-muted leading-relaxed">{uncertainty}</p>
        </section>
      )}

      {verifyMeaningful && variant === "workspace" && (
        <p className="text-[10px] text-muted">
          Verification: {answer.verification.status}
          {answer.verification.recoveryPerformed ? " · recovery performed" : ""}
        </p>
      )}

      {hasDetails && (
        <Collapsible title="Analysis details" compact defaultOpen={false}>
          <div className="space-y-3 text-[11px] text-secondary">
            {toolsRan && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-muted mb-1">Tools</p>
                <div className="flex flex-wrap gap-1.5">
                  {answer.toolsUsed.map((t) => (
                    <Chip key={t} className="!rounded-md !text-[10px]">
                      {t}
                    </Chip>
                  ))}
                </div>
              </div>
            )}
            {answer.routeDetail?.reason && (
              <p>
                <span className="text-muted">Plan: </span>
                {answer.routeDetail.reason}
                {answer.routeDetail.components?.length
                  ? ` · ${answer.routeDetail.components.join(", ")}`
                  : ""}
              </p>
            )}
            {answer.timing.totalMs != null && (
              <p className="font-mono text-muted">{answer.timing.totalMs} ms total</p>
            )}
            {verifyMeaningful && (
              <p>
                Verifier: {answer.verification.status}
                {answer.verification.message ? ` — ${answer.verification.message}` : ""}
              </p>
            )}
            {answer.system?.textModel && (
              <p className="text-muted">
                {answer.system.textModel}
                {answer.system.precision ? ` · ${answer.system.precision}` : ""}
              </p>
            )}
            {settings.showRawToolOutput && answer.rawToolOutput && (
              <pre className="text-[10px] font-mono bg-muted/40 p-2 rounded overflow-x-auto">
                {answer.rawToolOutput}
              </pre>
            )}
            {answer.timeline.length > 0 && variant === "workspace" && (
              <div>
                <TimelinePipeline steps={answer.timeline} />
                <ToolTimeline steps={answer.timeline} compact />
              </div>
            )}
          </div>
        </Collapsible>
      )}
    </div>
  );
}
