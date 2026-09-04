"use client";

import { useEffect, useState } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { EXAMPLE_QUESTIONS } from "@/lib/constants";
import { Button } from "@/components/ui/Button";
import { Collapsible } from "@/components/ui/Collapsible";
import { AgentResponseView } from "@/components/agent/AgentResponse";
import { ToolTimeline, TimelinePipeline } from "@/components/agent/ToolTimeline";
import { EmptyState } from "@/components/ui/EmptyState";
import { AgentSkeleton } from "@/components/ui/Skeleton";
import { inferNeedsVision } from "@/lib/routing";
import clsx from "clsx";

const EVIDENCE_ITEMS = [
  { key: "eeg", label: "EEG", modality: "eeg" as const },
  { key: "metadata", label: "Metadata", modality: "metadata" as const },
  { key: "images", label: "Figures", modality: "vision" as const },
];

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
  } = useExperiment();
  const [question, setQuestion] = useState("");

  useEffect(() => {
    setQuestion("");
  }, [workspaceEpoch]);

  const handleAnalyze = () => {
    if (!question.trim()) return;
    void analyze(question);
  };

  const availableEvidence = experiment
    ? EVIDENCE_ITEMS.filter((item) => {
        if (item.key === "eeg") return experiment.modalities.eeg;
        if (item.key === "metadata") return experiment.modalities.metadata;
        return experiment.modalities.vision;
      })
    : [];

  const showEmptyPrompt = !isAnalyzing && !currentAnswer;
  const history = experiment?.analysis_history ?? [];
  const previewRoute = question.trim()
    ? inferNeedsVision(question)
      ? "VISION + TOOLS"
      : "TEXT + TOOLS"
    : null;

  const runningStage = currentAnswer?.timeline.find((s) => s.status === "running")?.name;
  const statusLine =
    runningStage === "Vision analysis"
      ? "Vision-required processing…"
      : runningStage === "Verification"
        ? "Verifier running…"
        : runningStage === "Recovery"
          ? "Recovery in progress…"
          : runningStage
            ? `${runningStage}…`
            : "Analysis running…";

  return (
    <section className="h-full flex flex-col rounded-lg border border-default bg-elevated overflow-hidden">
      <header className="shrink-0 flex items-center justify-between px-3 py-2 border-b border-default/80">
        <h2 className="text-[11px] font-semibold tracking-[0.12em] text-secondary uppercase">
          Research Agent
        </h2>
        {currentAnswer && !isAnalyzing && (
          <span className="text-[10px] font-mono text-muted">
            {currentAnswer.route === "VISION" ? "VISION + TOOLS" : "TEXT + TOOLS"}
          </span>
        )}
      </header>

      <div className="flex flex-col h-full min-h-0 p-3 gap-3">
        <div className="space-y-2 shrink-0">
          <label className="block">
            <span className="text-[10px] font-medium text-muted uppercase tracking-wide">
              Ask about this experiment
            </span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Compare beta activity between left- and right-fist conditions…"
              rows={2}
              disabled={isAnalyzing}
              aria-label="Ask about this experiment"
              className="research-question-input mt-1.5 w-full resize-none rounded-md border border-[var(--border)] bg-[var(--surface-canvas)] px-2.5 py-2 text-[13px] text-[var(--text-primary)] caret-[var(--accent)] placeholder:text-[var(--text-muted)] selection:bg-[color-mix(in_srgb,var(--accent)_28%,transparent)] selection:text-[var(--text-primary)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[color-mix(in_srgb,var(--accent)_45%,transparent)] disabled:cursor-not-allowed disabled:border-[var(--border)] disabled:bg-[var(--surface)] disabled:text-[var(--text-muted)] disabled:opacity-60 disabled:placeholder:text-[var(--text-muted)]"
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleAnalyze();
              }}
            />
          </label>

          {previewRoute && (
            <p className="text-[10px] font-mono text-muted">
              Predicted route:{" "}
              <span
                className={
                  previewRoute.startsWith("VISION") ? "text-signal-vision" : "text-signal-text"
                }
              >
                {previewRoute}
              </span>
              {previewRoute.startsWith("VISION") && selectedImage && (
                <span className="text-muted"> · {selectedImage.name}</span>
              )}
            </p>
          )}

          {selectedImage && (
            <p className="text-[10px] text-muted">
              Vision context:{" "}
              <span className="font-mono text-signal-vision">{selectedImage.name}</span>
            </p>
          )}

          <Button
            className="w-full"
            size="sm"
            onClick={handleAnalyze}
            disabled={!experiment || isAnalyzing || !question.trim()}
          >
            {isAnalyzing ? "Running…" : "Analyze"}
          </Button>

          {analysisError && (
            <div className="rounded border border-signal-warning/35 bg-signal-warning/8 px-2.5 py-2" role="alert">
              <div className="flex items-start justify-between gap-2">
                <p className="text-[11px] text-signal-warning leading-relaxed">{analysisError}</p>
                <button type="button" onClick={clearAnalysisError} className="text-[10px] text-muted shrink-0">
                  Dismiss
                </button>
              </div>
            </div>
          )}

          {backendMode !== "live" && (
            <p className="text-[10px] text-muted leading-snug">
              Labeled demo fixtures until backend is connected.
            </p>
          )}
        </div>

        {history.length > 0 && (
          <Collapsible title={`Analysis History (${history.length})`} compact>
            <ul className="space-y-1.5 max-h-40 overflow-y-auto">
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
                          : "border-default/60 hover:border-accent/30 hover:bg-muted/30",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={clsx(
                            "text-[9px] font-mono",
                            h.route === "VISION" ? "text-signal-vision" : "text-signal-text",
                          )}
                        >
                          {h.route === "VISION" ? "VISION + TOOLS" : "TEXT + TOOLS"}
                        </span>
                        <span className="text-[9px] text-muted font-mono">
                          {h.timestamp
                            ? new Date(h.timestamp).toLocaleTimeString([], {
                                hour: "2-digit",
                                minute: "2-digit",
                              })
                            : "—"}
                        </span>
                      </div>
                      <p className="text-[11px] text-primary mt-0.5 line-clamp-2 leading-snug">
                        {h.question}
                      </p>
                      <p className="text-[10px] text-muted mt-0.5 line-clamp-1">
                        {h.answerPreview}
                      </p>
                    </button>
                  </li>
                );
              })}
            </ul>
          </Collapsible>
        )}

        {showEmptyPrompt && (
          <div className="shrink-0 space-y-3">
            {experiment ? (
              <>
                <div>
                  <p className="text-[10px] font-medium text-muted uppercase tracking-wide mb-1.5">
                    Available evidence
                  </p>
                  <div className="flex flex-wrap gap-x-3 gap-y-1">
                    {availableEvidence.map((item) => (
                      <span key={item.key} className="inline-flex items-center gap-1.5 text-[11px] text-secondary">
                        <span
                          className={`w-1 h-1 rounded-full ${
                            item.modality === "eeg"
                              ? "bg-signal-eeg"
                              : item.modality === "metadata"
                                ? "bg-signal-meta"
                                : "bg-signal-vision"
                          }`}
                          aria-hidden
                        />
                        {item.label}
                        {item.key === "images" && experiment.image_files.length > 0
                          ? ` (${experiment.image_files.length})`
                          : ""}
                      </span>
                    ))}
                    {availableEvidence.length === 0 && (
                      <span className="text-[11px] text-muted">No modalities loaded yet</span>
                    )}
                  </div>
                </div>

                <div>
                  <p className="text-[10px] font-medium text-muted uppercase tracking-wide mb-1.5">
                    Suggested
                  </p>
                  <div className="flex flex-col">
                    {EXAMPLE_QUESTIONS.map((q) => (
                      <button
                        key={q}
                        type="button"
                        onClick={() => setQuestion(q)}
                        className="text-left text-[11px] text-secondary py-1.5 border-b border-default/50 last:border-0 hover:text-primary transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <EmptyState
                title="No experiment loaded"
                description="Load data or try the demo to ask research questions."
                actions={
                  <Button size="sm" variant="primary" onClick={loadDemo}>
                    Try Demo
                  </Button>
                }
              />
            )}
          </div>
        )}

        <div className="flex-1 min-h-0 overflow-y-auto border-t border-default/60 pt-3">
          {isAnalyzing && currentAnswer && (
            <div className="mb-4 space-y-3">
              <div className="flex items-center justify-between gap-2">
                <p className="text-[11px] text-accent">{statusLine}</p>
                <span className="text-[10px] font-mono text-muted">
                  {currentAnswer.route === "VISION" ? "VISION + TOOLS" : "TEXT + TOOLS"}
                </span>
              </div>
              <div className="overflow-x-auto pb-1">
                <TimelinePipeline steps={currentAnswer.timeline} />
              </div>
              <ToolTimeline steps={currentAnswer.timeline} active compact />
            </div>
          )}

          {isAnalyzing && !currentAnswer && <AgentSkeleton />}

          {!isAnalyzing && currentAnswer && (
            <AgentResponseView answer={currentAnswer} />
          )}
        </div>
      </div>
    </section>
  );
}
