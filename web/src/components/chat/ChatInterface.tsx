"use client";

/**
 * Production Chat — live conversational neuroscience research assistant.
 * Shares experiment/analyze state with Workspace; no offline fixtures.
 */
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
import { AgentSkeleton } from "@/components/ui/Skeleton";
import { Button } from "@/components/ui/Button";

type AttachKind = "eeg" | "figure" | "metadata";

export function ChatInterface() {
  const {
    beginChatSession,
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
    void beginChatSession();
  }, [beginChatSession]);

  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [answers, isAnalyzing, analysisError]);

  const conversationStarted = answers.length > 0 || isAnalyzing;

  const attachmentChips = [
    ...(experiment?.eeg_files ?? []),
    ...(experiment?.metadata_files ?? []),
    ...(experiment?.image_files ?? []),
  ].filter((f) => f.status !== "error");

  const statusLabel = (() => {
    if (explorerLoading) return "Connecting…";
    if (isAnalyzing) return "Thinking…";
    if (backendMode === "live") return describeBackendStatus(healthInfo);
    if (backendMode === "unavailable") return "API unavailable";
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
    <div className="h-screen flex flex-col bg-surface">
      <header className="shrink-0 flex items-center justify-between gap-3 px-4 py-3 border-b border-default bg-elevated">
        <div className="flex items-center gap-2.5 min-w-0">
          <Link href="/" className="flex items-center gap-2 hover:opacity-85">
            <div className="w-7 h-7 rounded-md bg-accent/12 border border-accent/25 flex items-center justify-center">
              <span className="text-accent font-mono text-xs">Ψ</span>
            </div>
            <div className="min-w-0">
              <h1 className="text-[13px] font-semibold text-primary tracking-tight">
                Neuro Agent
              </h1>
              <p className="text-[10px] text-muted">Research chat</p>
            </div>
          </Link>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[10px] text-muted hidden sm:inline">{statusLabel}</span>
          <Link href="/workspace">
            <Button variant="outline" size="sm">
              Workspace
            </Button>
          </Link>
        </div>
      </header>

      {backendMode === "unavailable" && (
        <div className="shrink-0 px-4 py-2 bg-signal-error/10 border-b border-signal-error/30 text-[12px] text-signal-error">
          Live API is unavailable. Start the backend and refresh — the product does not use offline
          fixtures.
        </div>
      )}

      <div ref={threadRef} className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-2xl mx-auto space-y-6">
          {!conversationStarted && (
            <div className="text-center space-y-4 py-10 animate-fade-in">
              <h2 className="text-xl font-semibold text-primary tracking-tight">
                Neuroscience research assistant
              </h2>
              <p className="text-[13px] text-secondary max-w-md mx-auto leading-relaxed">
                Ask conceptual questions, analyze an uploaded EEG sample, or interpret a selected
                figure. Attachments are optional until the task needs them.
              </p>
              <div className="flex flex-wrap justify-center gap-2 pt-2">
                {STARTER_PROMPTS.slice(0, 4).map((p) => (
                  <button
                    key={p}
                    type="button"
                    className="text-[11px] px-3 py-1.5 rounded-lg border border-default text-secondary hover:border-accent/40 hover:text-primary transition-colors max-w-[240px] text-left"
                    onClick={() => {
                      setQuestion(p);
                      setShowExamples(false);
                    }}
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          )}

          {answers.map((a) => (
            <div key={a.id} className="space-y-3">
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent/15 border border-accent/25 px-3.5 py-2 text-[13px] text-primary">
                  {a.question}
                </div>
              </div>
              <div className="rounded-2xl rounded-tl-md border border-default bg-elevated/60 px-4 py-3">
                <AgentResponseView answer={a} variant="chat" />
              </div>
            </div>
          ))}

          {isAnalyzing && (
            <div className="rounded-2xl border border-default bg-elevated/60 px-4 py-3">
              <AgentSkeleton />
            </div>
          )}

          {analysisError && (
            <div className="rounded-lg border border-signal-error/40 bg-signal-error/10 px-3 py-2 text-[12px] text-signal-error flex justify-between gap-2">
              <span>{analysisError}</span>
              <button type="button" className="underline shrink-0" onClick={clearAnalysisError}>
                Dismiss
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="shrink-0 border-t border-default bg-elevated px-4 py-3">
        <div className="max-w-2xl mx-auto space-y-2">
          {(uploadError || attachmentChips.length > 0 || selectedImage) && (
            <div className="flex flex-wrap items-center gap-2">
              {uploadError && (
                <span className="text-[11px] text-signal-error">
                  {uploadError}{" "}
                  <button type="button" className="underline" onClick={clearUploadError}>
                    dismiss
                  </button>
                </span>
              )}
              {attachmentChips.map((f) => (
                <span
                  key={f.id}
                  className="inline-flex items-center gap-1.5 text-[10px] px-2 py-1 rounded-md bg-muted/50 text-secondary"
                >
                  {f.name}
                  <button
                    type="button"
                    className="text-muted hover:text-primary"
                    onClick={() => removeFile(f.id)}
                    aria-label={`Remove ${f.name}`}
                  >
                    ×
                  </button>
                </span>
              ))}
              {selectedImage && !attachmentChips.some((f) => f.id === selectedImage.id) && (
                <span className="text-[10px] text-signal-vision">Figure: {selectedImage.name}</span>
              )}
            </div>
          )}

          <div className="flex items-end gap-2">
            <div className="relative">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setAttachOpen((o) => !o)}
                aria-label="Attach"
              >
                +
              </Button>
              {attachOpen && (
                <div className="absolute bottom-full left-0 mb-1 w-44 rounded-lg border border-default bg-elevated shadow-lg py-1 z-20">
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
                      className="w-full text-left px-3 py-1.5 text-[12px] text-secondary hover:bg-muted/40"
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
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={1}
              placeholder={
                backendMode === "unavailable"
                  ? "API unavailable…"
                  : "Ask a neuroscience research question…"
              }
              disabled={backendMode === "unavailable" || isAnalyzing}
              className="flex-1 resize-none rounded-xl border border-default bg-canvas px-3 py-2.5 text-[13px] text-primary placeholder:text-muted focus:outline-none focus:border-accent/50 min-h-[42px] max-h-32"
            />
            <Button
              size="sm"
              onClick={send}
              disabled={!question.trim() || isAnalyzing || backendMode === "unavailable"}
            >
              Send
            </Button>
          </div>
          <div className="flex justify-between items-center">
            <button
              type="button"
              className="text-[10px] text-muted hover:text-secondary"
              onClick={() => setShowExamples((s) => !s)}
            >
              {showExamples ? "Hide examples" : "Examples"}
            </button>
            {currentAnswer?.timing.totalMs != null && (
              <span className="text-[10px] font-mono text-muted">
                {currentAnswer.timing.totalMs} ms
              </span>
            )}
          </div>
          {showExamples && (
            <div className="flex flex-wrap gap-1.5">
              {STARTER_PROMPTS.map((p) => (
                <button
                  key={p}
                  type="button"
                  className="text-[10px] px-2 py-1 rounded border border-default text-muted hover:text-secondary"
                  onClick={() => setQuestion(p)}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <input
        ref={eegRef}
        type="file"
        accept={EEG_ACCEPT}
        className="hidden"
        onChange={(e) => onAttach("eeg", e.target.files?.[0])}
      />
      <input
        ref={figRef}
        type="file"
        accept={FIGURE_ACCEPT}
        className="hidden"
        onChange={(e) => onAttach("figure", e.target.files?.[0])}
      />
      <input
        ref={metaRef}
        type="file"
        accept={METADATA_ACCEPT}
        className="hidden"
        onChange={(e) => onAttach("metadata", e.target.files?.[0])}
      />
    </div>
  );
}
