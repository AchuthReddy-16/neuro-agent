"use client";

import { useEffect, useState } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { STARTER_PROMPTS } from "@/lib/constants";
import { describeBackendStatus } from "@/lib/config";
import { Button } from "@/components/ui/Button";
import { Collapsible } from "@/components/ui/Collapsible";
import { AgentResponseView } from "@/components/agent/AgentResponse";
import { ToolTimeline, TimelinePipeline } from "@/components/agent/ToolTimeline";
import { EmptyState } from "@/components/ui/EmptyState";
import { AgentSkeleton } from "@/components/ui/Skeleton";
import clsx from "clsx";

/**
 * Workspace Research Agent — question → answer → evidence first; secondary controls collapsed.
 */
export function ResearchAgentPanel() {
  const {
    experiment,
    currentAnswer,
    activeAnswerId,
    isAnalyzing,
    analyze,
    loadDemo,
    backendMode,
    analysisError,
    clearAnalysisError,
    selectedImage,
    workspaceEpoch,
    restoreAnalysis,
    healthInfo,
  } = useExperiment();
  const [question, setQuestion] = useState("");
  const [showExamples, setShowExamples] = useState(false);

  useEffect(() => {
    setQuestion("");
    setShowExamples(false);
  }, [workspaceEpoch]);

  const handleSend = () => {
    if (!question.trim()) return;
    void analyze(question);
  };

  const conversationStarted = !!currentAnswer || isAnalyzing;
  const history = experiment?.analysis_history ?? [];

  const contextBits = [
    { label: "EEG", ok: !!experiment?.modalities.eeg },
    { label: "Metadata", ok: !!experiment?.modalities.metadata },
    { label: "Figure", ok: !!experiment?.modalities.vision },
  ];

  const runningStage = currentAnswer?.timeline.find((s) => s.status === "running")?.name;
  const statusLine =
    runningStage === "Vision analysis"
      ? "Analyzing figure…"
      : runningStage === "Verification"
        ? "Verifying…"
        : runningStage === "Recovery"
          ? "Recovering…"
          : runningStage
            ? `${runningStage}…`
            : backendMode === "live" && healthInfo && !(healthInfo.agentLoaded ?? healthInfo.agent_loaded)
              ? "Preparing research model…"
              : "Analyzing…";

  return (
    <section className="h-full flex flex-col rounded-lg border border-default bg-elevated overflow-hidden">
      <header className="shrink-0 flex items-center justify-between px-3 py-2 border-b border-default/80">
        <h2 className="text-[11px] font-semibold tracking-[0.12em] text-secondary uppercase">
          Research Agent
        </h2>
        <span className="text-[10px] font-mono text-muted truncate max-w-[50%]">
          {isAnalyzing
            ? statusLine
            : backendMode === "live"
              ? describeBackendStatus(healthInfo)
              : backendMode === "unavailable"
                ? "Offline"
                : "Connecting…"}
        </span>
      </header>

      <div className="flex flex-col h-full min-h-0 p-3 gap-3">
        {/* Primary composer */}
        <div className="space-y-2 shrink-0">
          <div className="flex gap-2 items-end">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a research question…"
              rows={2}
              disabled={isAnalyzing}
              aria-label="Ask a research question"
              className="research-question-input flex-1 resize-none rounded-md border border-[var(--border)] bg-[var(--surface-canvas)] px-2.5 py-2 text-[13px] text-[var(--text-primary)] caret-[var(--accent)] placeholder:text-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[color-mix(in_srgb,var(--accent)_45%,transparent)] disabled:opacity-60"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSend();
              }}
            />
            <Button
              size="sm"
              className="shrink-0 h-[42px] px-3"
              onClick={handleSend}
              disabled={!experiment || isAnalyzing || !question.trim()}
            >
              Send
            </Button>
          </div>

          {experiment && (
            <p className="text-[10px] text-muted font-mono flex flex-wrap gap-x-3 gap-y-0.5">
              <span>Context:</span>
              {contextBits.map((b) => (
                <span key={b.label} className={b.ok ? "text-secondary" : "text-muted/70"}>
                  {b.label} {b.ok ? "✓" : "—"}
                </span>
              ))}
              {selectedImage && (
                <span className="text-accent truncate max-w-[160px]">
                  Selected: {selectedImage.name}
                </span>
              )}
            </p>
          )}

          {analysisError && (
            <div
              className="rounded border border-signal-warning/35 bg-signal-warning/8 px-2.5 py-2"
              role="alert"
            >
              <div className="flex items-start justify-between gap-2">
                <p className="text-[11px] text-signal-warning leading-relaxed">{analysisError}</p>
                <button
                  type="button"
                  onClick={clearAnalysisError}
                  className="text-[10px] text-muted shrink-0"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Empty: ≤3 subtle starters */}
        {!conversationStarted && experiment && (
          <div className="shrink-0 space-y-1">
            <p className="text-[10px] text-muted uppercase tracking-wide">Try asking</p>
            <ul>
              {STARTER_PROMPTS.slice(0, 3).map((q) => (
                <li key={q}>
                  <button
                    type="button"
                    onClick={() => setQuestion(q)}
                    className="w-full text-left text-[12px] text-muted hover:text-secondary py-1.5 border-b border-default/40 last:border-0 transition-colors"
                  >
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {!conversationStarted && !experiment && (
          <EmptyState
            title="No experiment loaded"
            description="Upload data in the experiment panel, or open the Interactive Demo."
            actions={
              <Button size="sm" variant="outline" onClick={() => void loadDemo()}>
                Load demo sample
              </Button>
            }
          />
        )}

        {conversationStarted && (
          <div className="shrink-0">
            <button
              type="button"
              onClick={() => setShowExamples((v) => !v)}
              className="text-[10px] text-muted hover:text-secondary"
            >
              {showExamples ? "Hide examples" : "Examples"}
            </button>
            {showExamples && (
              <ul className="mt-1 space-y-0.5">
                {STARTER_PROMPTS.map((q) => (
                  <li key={q}>
                    <button
                      type="button"
                      onClick={() => setQuestion(q)}
                      className="text-[11px] text-muted hover:text-secondary"
                    >
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Answer / evidence — primary scroll area */}
        <div className="flex-1 min-h-0 overflow-y-auto space-y-3">
          {isAnalyzing && currentAnswer && (
            <div className="space-y-2">
              <p className="text-[11px] text-accent">{statusLine}</p>
              <div className="overflow-x-auto pb-1">
                <TimelinePipeline steps={currentAnswer.timeline} />
              </div>
            </div>
          )}

          {isAnalyzing && !currentAnswer && <AgentSkeleton />}

          {!isAnalyzing && analysisError && !currentAnswer && (
            <div className="rounded-lg border border-signal-warning/30 bg-signal-warning/5 px-3 py-3 text-[12px] text-secondary">
              No answer returned. {analysisError}
            </div>
          )}

          {!isAnalyzing && currentAnswer && <AgentResponseView answer={currentAnswer} />}
        </div>

        {/* Secondary — collapsed */}
        <div className="shrink-0 space-y-1.5 border-t border-default/50 pt-2">
          {history.length > 0 && (
            <Collapsible title={`History (${history.length})`} compact>
              <ul className="space-y-1 max-h-36 overflow-y-auto">
                {history.map((h) => {
                  const answerId = h.answer?.id;
                  const active =
                    (!!answerId && activeAnswerId === answerId) ||
                    (!!answerId && currentAnswer?.id === answerId);
                  return (
                    <li key={h.id}>
                      <button
                        type="button"
                        onClick={() => restoreAnalysis(h.id)}
                        className={clsx(
                          "w-full text-left rounded-md px-2 py-1.5 border transition-colors",
                          active
                            ? "border-accent/40 bg-accent/8"
                            : "border-default/60 hover:border-accent/30",
                        )}
                      >
                        <p className="text-[11px] text-primary line-clamp-2">{h.question}</p>
                        <p className="text-[10px] text-muted line-clamp-1 mt-0.5">{h.answerPreview}</p>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </Collapsible>
          )}

          {currentAnswer && (
            <Collapsible title="Tools & verification" compact>
              <ToolTimeline steps={currentAnswer.timeline} active={false} compact />
              <p className="text-[11px] text-secondary mt-2">
                Verifier: {currentAnswer.verification.status}
                {currentAnswer.verification.message
                  ? ` — ${currentAnswer.verification.message}`
                  : ""}
              </p>
            </Collapsible>
          )}

          {currentAnswer && (
            <Collapsible title="System details" compact>
              <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-[10px]">
                <dt className="text-muted">Text</dt>
                <dd className="text-secondary truncate">{currentAnswer.system.textModel}</dd>
                <dt className="text-muted">Vision</dt>
                <dd className="text-secondary truncate">{currentAnswer.system.visionModel}</dd>
                <dt className="text-muted">Precision</dt>
                <dd className="text-secondary">{currentAnswer.system.precision}</dd>
                <dt className="text-muted">Serving</dt>
                <dd className="text-secondary">{currentAnswer.system.serving}</dd>
              </dl>
            </Collapsible>
          )}
        </div>
      </div>
    </section>
  );
}
