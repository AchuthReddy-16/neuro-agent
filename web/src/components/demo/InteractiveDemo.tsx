"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useExperiment } from "@/lib/store/experiment-context";
import {
  EEG_ACCEPT,
  FIGURE_ACCEPT,
  METADATA_ACCEPT,
  STARTER_PROMPTS,
} from "@/lib/constants";
import { describeBackendStatus } from "@/lib/config";
import { AgentResponseView } from "@/components/agent/AgentResponse";
import { TimelinePipeline } from "@/components/agent/ToolTimeline";
import { AgentSkeleton } from "@/components/ui/Skeleton";
import { Button } from "@/components/ui/Button";

type AttachKind = "eeg" | "figure" | "metadata";

/**
 * Conversational Interactive Demo — lightweight, isolated from the Research Workspace.
 */
export function InteractiveDemo() {
  const {
    beginDemoSession,
    experiment,
    answers,
    currentAnswer,
    isAnalyzing,
    analyze,
    uploadEEG,
    uploadFigure,
    uploadMetadata,
    removeFile,
    selectedImage,
    analysisError,
    clearAnalysisError,
    uploadError,
    clearUploadError,
    backendMode,
    healthInfo,
    explorerLoading,
  } = useExperiment();

  const [question, setQuestion] = useState("");
  const [attachOpen, setAttachOpen] = useState(false);
  const [showExamples, setShowExamples] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);
  const eegRef = useRef<HTMLInputElement>(null);
  const figRef = useRef<HTMLInputElement>(null);
  const metaRef = useRef<HTMLInputElement>(null);
  const booted = useRef(false);

  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    void beginDemoSession();
  }, [beginDemoSession]);

  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [answers, isAnalyzing, analysisError]);

  const conversationStarted = answers.length > 0 || isAnalyzing;

  // Built-in demo assets stay out of the composer; user attachments appear after fork
  const attachmentChips = experiment?.isDemo
    ? []
    : [
        ...(experiment?.eeg_files ?? []),
        ...(experiment?.metadata_files ?? []),
        ...(experiment?.image_files ?? []),
      ].filter((f) => f.status !== "error");

  const statusLabel = (() => {
    if (explorerLoading) return "Loading demo…";
    if (isAnalyzing) {
      const stage = currentAnswer?.timeline.find((s) => s.status === "running")?.name;
      if (stage === "Vision analysis") return "Analyzing figure…";
      if (stage) return `${stage}…`;
      return "Analyzing…";
    }
    if (backendMode === "live") return describeBackendStatus(healthInfo);
    if (backendMode === "unavailable") return "Offline — fixture responses only";
    return "Connecting…";
  })();

  const send = () => {
    const q = question.trim();
    if (!q || isAnalyzing) return;
    setQuestion("");
    setShowExamples(false);
    setAttachOpen(false);
    void analyze(q);
  };

  const onAttach = (kind: AttachKind, file: File | undefined) => {
    if (!file) return;
    setAttachOpen(false);
    if (kind === "eeg") void uploadEEG(file);
    else if (kind === "figure") void uploadFigure(file);
    else void uploadMetadata(file);
  };

  return (
    <div className="h-screen flex flex-col bg-surface overflow-hidden">
      <header className="shrink-0 flex items-center justify-between gap-3 px-4 sm:px-6 py-3 border-b border-default bg-elevated/80">
        <div className="flex items-center gap-2.5 min-w-0">
          <Link href="/" className="flex items-center gap-2.5 hover:opacity-85 transition-opacity">
            <div className="w-7 h-7 rounded-md bg-accent/12 border border-accent/25 flex items-center justify-center shrink-0">
              <span className="text-accent font-mono text-xs">Ψ</span>
            </div>
            <span className="text-sm font-medium text-primary tracking-tight truncate">
              Neuro Research Agent
            </span>
          </Link>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <span className="text-[10px] font-mono text-muted hidden sm:inline">{statusLabel}</span>
          <Link href="/workspace">
            <Button variant="ghost" size="sm">
              Workspace
            </Button>
          </Link>
        </div>
      </header>

      <div ref={threadRef} className="flex-1 min-h-0 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 sm:px-6 py-8 sm:py-12 space-y-6">
          {!conversationStarted && (
            <div className="text-center space-y-5 pt-8 sm:pt-16 animate-fade-in">
              <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-primary">
                Neuro Research Agent
              </h1>
              <p className="text-sm text-secondary max-w-md mx-auto leading-relaxed">
                Ask about EEG samples, metadata, or figures. Attach a file with + when you want
                to try your own inputs.
              </p>
              <ul className="pt-2 space-y-1.5 max-w-sm mx-auto">
                {STARTER_PROMPTS.slice(0, 3).map((p) => (
                  <li key={p}>
                    <button
                      type="button"
                      onClick={() => setQuestion(p)}
                      className="w-full text-left text-[13px] text-muted hover:text-secondary py-1.5 px-3 rounded-md hover:bg-muted/40 transition-colors"
                    >
                      {p}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {answers.map((a) => (
            <div key={a.id} className="space-y-3">
              {a.question && (
                <div className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl rounded-br-md bg-muted/70 border border-default px-3.5 py-2.5 text-[13px] text-primary leading-relaxed">
                    {a.question}
                  </div>
                </div>
              )}
              <div className="rounded-2xl rounded-bl-md border border-default bg-elevated px-4 py-3.5">
                {isAnalyzing && currentAnswer?.id === a.id && !a.answer ? (
                  <div className="space-y-3">
                    <p className="text-[11px] text-accent">{statusLabel}</p>
                    {a.timeline?.length > 0 && (
                      <div className="overflow-x-auto">
                        <TimelinePipeline steps={a.timeline} />
                      </div>
                    )}
                    <AgentSkeleton />
                  </div>
                ) : (
                  <AgentResponseView answer={a} />
                )}
              </div>
            </div>
          ))}

          {isAnalyzing && !currentAnswer && <AgentSkeleton />}

          {analysisError && (
            <div
              className="rounded-xl border border-signal-warning/40 bg-signal-warning/10 px-4 py-3"
              role="alert"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[11px] font-medium text-signal-warning mb-1">Analysis error</p>
                  <p className="text-[13px] text-secondary leading-relaxed">{analysisError}</p>
                </div>
                <button
                  type="button"
                  onClick={clearAnalysisError}
                  className="text-[10px] text-muted shrink-0 hover:text-primary"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="shrink-0 border-t border-default bg-elevated/90 backdrop-blur-sm">
        <div className="max-w-2xl mx-auto px-4 sm:px-6 py-3 space-y-2">
          {(uploadError || attachmentChips.length > 0 || (selectedImage && !experiment?.isDemo)) && (
            <div className="flex flex-wrap items-center gap-1.5">
              {uploadError && (
                <span className="text-[10px] text-signal-warning mr-1">
                  {uploadError}{" "}
                  <button type="button" className="underline" onClick={clearUploadError}>
                    dismiss
                  </button>
                </span>
              )}
              {attachmentChips.map((f) => (
                <span
                  key={f.id}
                  className="inline-flex items-center gap-1.5 rounded-full border border-default bg-canvas px-2.5 py-0.5 text-[11px] text-secondary"
                >
                  <span className="truncate max-w-[140px]">{f.name}</span>
                  <button
                    type="button"
                    aria-label={`Remove ${f.name}`}
                    onClick={() => removeFile(f.id)}
                    className="text-muted hover:text-primary leading-none"
                  >
                    ×
                  </button>
                </span>
              ))}
              {selectedImage && !experiment?.isDemo && (
                <span className="inline-flex items-center gap-1 rounded-full border border-accent/30 bg-accent/8 px-2.5 py-0.5 text-[11px] text-accent">
                  Selected: {selectedImage.name}
                </span>
              )}
            </div>
          )}

          {conversationStarted && (
            <div className="flex justify-start">
              <button
                type="button"
                onClick={() => setShowExamples((v) => !v)}
                className="text-[10px] text-muted hover:text-secondary"
              >
                {showExamples ? "Hide examples" : "Examples"}
              </button>
            </div>
          )}
          {showExamples && conversationStarted && (
            <ul className="flex flex-wrap gap-1.5">
              {STARTER_PROMPTS.map((p) => (
                <li key={p}>
                  <button
                    type="button"
                    onClick={() => setQuestion(p)}
                    className="text-[11px] text-muted border border-default/70 rounded-full px-2.5 py-0.5 hover:text-secondary hover:border-strong"
                  >
                    {p}
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="relative flex items-end gap-2">
            <div className="relative">
              <button
                type="button"
                aria-label="Attach file"
                aria-expanded={attachOpen}
                onClick={() => setAttachOpen((o) => !o)}
                className="h-10 w-10 shrink-0 rounded-xl border border-default bg-canvas text-secondary hover:text-primary hover:border-accent/40 flex items-center justify-center text-lg leading-none"
              >
                +
              </button>
              {attachOpen && (
                <div className="absolute bottom-full left-0 mb-2 w-48 rounded-lg border border-default bg-elevated shadow-lg py-1 z-20">
                  {(
                    [
                      ["eeg", "EEG / sample JSON", eegRef],
                      ["figure", "Figure / image", figRef],
                      ["metadata", "Metadata", metaRef],
                    ] as const
                  ).map(([kind, label, ref]) => (
                    <button
                      key={kind}
                      type="button"
                      className="w-full text-left px-3 py-2 text-[12px] text-secondary hover:bg-muted/50 hover:text-primary"
                      onClick={() => ref.current?.click()}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about EEG, metadata, or a figure…"
              rows={1}
              disabled={isAnalyzing}
              aria-label="Ask about EEG, metadata, or a figure"
              className="research-question-input flex-1 min-h-[40px] max-h-32 resize-none rounded-xl border border-[var(--border)] bg-[var(--surface-canvas)] px-3.5 py-2.5 text-[13px] text-[var(--text-primary)] caret-[var(--accent)] placeholder:text-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none focus:ring-1 focus:ring-[color-mix(in_srgb,var(--accent)_45%,transparent)] disabled:opacity-60"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />

            <Button
              size="sm"
              className="h-10 px-4 shrink-0"
              onClick={send}
              disabled={isAnalyzing || !question.trim()}
            >
              Send
            </Button>
          </div>

          <input
            ref={eegRef}
            type="file"
            accept={EEG_ACCEPT}
            className="hidden"
            onChange={(e) => {
              onAttach("eeg", e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <input
            ref={figRef}
            type="file"
            accept={FIGURE_ACCEPT}
            className="hidden"
            onChange={(e) => {
              onAttach("figure", e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <input
            ref={metaRef}
            type="file"
            accept={METADATA_ACCEPT}
            className="hidden"
            onChange={(e) => {
              onAttach("metadata", e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </div>
      </div>
    </div>
  );
}
