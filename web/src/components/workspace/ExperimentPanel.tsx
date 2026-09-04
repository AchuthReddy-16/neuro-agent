"use client";

import clsx from "clsx";
import { useState, type ReactNode } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { Button } from "@/components/ui/Button";
import { Collapsible } from "@/components/ui/Collapsible";
import { DropZone } from "@/components/ui/DropZone";
import { EEG_ACCEPT, FIGURE_ACCEPT, METADATA_ACCEPT } from "@/lib/constants";
import type { ExperimentFile, ExperimentStatus } from "@/lib/types";

function StatusPill({ status }: { status: ExperimentStatus }) {
  const map: Record<ExperimentStatus, { label: string; className: string }> = {
    empty: { label: "Empty", className: "text-muted" },
    ready: { label: "Ready", className: "text-accent" },
    processing: { label: "Processing", className: "text-signal-warning" },
    error: { label: "Error", className: "text-signal-error" },
  };
  const s = map[status];
  return (
    <span className={clsx("inline-flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wide", s.className)}>
      <span
        className={clsx(
          "w-1.5 h-1.5 rounded-full",
          status === "ready" && "bg-accent",
          status === "processing" && "bg-signal-warning animate-pulse",
          status === "error" && "bg-signal-error",
          status === "empty" && "bg-muted border border-default",
        )}
        aria-hidden
      />
      {s.label}
    </span>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function ExperimentPanel() {
  const {
    experiment,
    experimentStatus,
    uploadError,
    loadDemo,
    uploadEEG,
    uploadFigure,
    uploadMetadata,
    removeFile,
    selectImage,
    clearImageSelection,
    updateMetadata,
    clearExperiment,
    clearUploadError,
    backendMode,
  } = useExperiment();

  const [confirmClear, setConfirmClear] = useState(false);
  const meta = experiment?.metadata;
  const hasData = !!experiment;

  const eegFiles = experiment?.eeg_files ?? [];
  const metaFiles = experiment?.metadata_files ?? [];
  const imageFiles = experiment?.image_files ?? [];

  return (
    <section className="rounded-lg border border-default bg-elevated overflow-hidden">
      <header className="flex items-center justify-between px-3 py-2 border-b border-default/80">
        <h2 className="text-[11px] font-semibold tracking-[0.12em] text-secondary uppercase">
          Experiment
        </h2>
        <StatusPill status={experimentStatus} />
      </header>

      <div className="p-3 space-y-3">
        {backendMode === "unavailable" && !hasData && (
          <p className="text-[10px] text-signal-warning leading-relaxed">
            Backend offline — uploads stay local; analysis uses explicitly labeled offline fixtures.
          </p>
        )}
        {backendMode === "live" && !hasData && (
          <p className="text-[10px] text-muted leading-relaxed">
            Live API connected. Upload JSON with{" "}
            <span className="font-mono">sample_id</span> (e.g. S001_R01_E000), figures, or try the
            demo experiment.
          </p>
        )}

        {/* Current experiment summary */}
        {experiment && (
          <div className="space-y-1.5 pb-2 border-b border-default/60">
            <p className="text-[10px] font-medium text-muted uppercase tracking-wide">
              Current Experiment
              {experiment.isDemo && <span className="ml-1.5 normal-case text-muted/70">· demo</span>}
            </p>
            <dl className="space-y-1 text-[11px]">
              <SummaryRow
                label="EEG"
                value={eegFiles[0]?.name ?? experiment.eeg?.filename ?? "—"}
              />
              <SummaryRow
                label="Metadata"
                value={metaFiles[0]?.name ?? (meta?.subject ? "session fields" : "—")}
              />
              <SummaryRow label="Figures" value={String(imageFiles.length)} />
            </dl>
            {(meta?.subject || meta?.run) && (
              <p className="text-[10px] text-muted font-mono pt-0.5">
                {[meta.subject, meta.run].filter(Boolean).join(" · ")}
              </p>
            )}
          </div>
        )}

        {uploadError && (
          <div className="rounded border border-signal-error/30 bg-signal-error/8 px-2.5 py-2" role="alert">
            <div className="flex items-start justify-between gap-2">
              <p className="text-[11px] text-signal-error leading-relaxed">{uploadError}</p>
              <button type="button" onClick={clearUploadError} className="text-[10px] text-muted hover:text-primary shrink-0">
                Dismiss
              </button>
            </div>
          </div>
        )}

        {/* EEG / DATA */}
        <FileGroup title="EEG / Data">
          <DropZone
            label="Upload EEG / sample JSON"
            hint="Sample JSON (e.g. S001_R01_E000)"
            accept={EEG_ACCEPT}
            onFiles={(list) => {
              const f = Array.from(list)[0];
              if (f) void uploadEEG(f);
            }}
          >
            <p className="text-[9px] text-muted mt-1">
              Use a known sample JSON. Raw EDF/CSV are not supported yet.
            </p>
          </DropZone>
          {eegFiles.length > 0 && (
            <ul className="mt-1.5 space-y-0.5">
              {eegFiles.map((f) => (
                <FileRow key={f.id} file={f} tone="eeg" onRemove={() => removeFile(f.id)} />
              ))}
            </ul>
          )}
        </FileGroup>

        {/* FIGURES / IMAGES */}
        <FileGroup title={`Figures / Images${imageFiles.length ? ` (${imageFiles.length})` : ""}`}>
          <DropZone
            label="Upload figures"
            hint="Multiple images supported"
            accept={FIGURE_ACCEPT}
            multiple
            onFiles={(list) => {
              void (async () => {
                for (const f of Array.from(list)) {
                  await uploadFigure(f);
                }
              })();
            }}
          >
            <FormatHints formats={[{ label: "PNG", active: true }, { label: "JPG", active: true }, { label: "WEBP", active: true }]} />
          </DropZone>
          {imageFiles.length > 0 && (
            <ul className="mt-1.5 space-y-1">
              {imageFiles.map((f) => {
                const selected = experiment?.selected_image_id === f.id;
                return (
                  <li key={f.id}>
                    <div
                      className={clsx(
                        "flex items-center gap-2 rounded-md px-1.5 py-1.5 border",
                        selected
                          ? "border-signal-vision/50 bg-signal-vision/10"
                          : "border-transparent hover:bg-muted/40",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => selectImage(f.id)}
                        className="flex items-center gap-2 min-w-0 flex-1 text-left"
                      >
                        <div className="w-8 h-8 rounded bg-muted overflow-hidden shrink-0 border border-default">
                          {f.url ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img src={f.url} alt="" className="w-full h-full object-cover" />
                          ) : (
                            <span className="text-[9px] text-muted flex items-center justify-center h-full">img</span>
                          )}
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] font-mono text-primary truncate leading-tight">{f.name}</p>
                          <p className="text-[9px] text-muted">
                            {selected ? "Selected" : "Figure"}
                            {f.status === "uploading" && ` · ${f.progress ?? 0}%`}
                            {f.status === "error" && ` · ${f.error ?? "error"}`}
                          </p>
                        </div>
                      </button>
                      <button
                        type="button"
                        onClick={() => removeFile(f.id)}
                        className="text-[10px] text-muted hover:text-signal-error px-1"
                        aria-label={`Remove ${f.name}`}
                      >
                        ✕
                      </button>
                    </div>
                    {f.status === "uploading" && (
                      <div className="mt-1 h-0.5 rounded-full bg-muted overflow-hidden mx-1.5">
                        <div className="h-full bg-accent/70" style={{ width: `${f.progress ?? 0}%` }} />
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
          {imageFiles.length > 1 && experiment?.selected_image_id && (
            <button
              type="button"
              onClick={clearImageSelection}
              className="mt-1 text-[10px] text-muted hover:text-primary"
            >
              Clear figure selection
            </button>
          )}
        </FileGroup>
        <FileGroup title="Metadata">
          <DropZone
            label="Upload metadata"
            hint="JSON with sample_id (e.g. S001_R01_E000)"
            accept={METADATA_ACCEPT}
            multiple
            onFiles={(list) => {
              Array.from(list).forEach((f) => void uploadMetadata(f));
            }}
          />
          {metaFiles.length > 0 && (
            <ul className="mt-1.5 space-y-0.5">
              {metaFiles.map((f) => (
                <FileRow key={f.id} file={f} tone="meta" onRemove={() => removeFile(f.id)} />
              ))}
            </ul>
          )}
          {meta && (
            <div className="mt-2">
              <Collapsible title="Session fields" compact>
                <div className="space-y-2">
                  <MetadataField label="Subject" value={meta.subject} onChange={(v) => updateMetadata({ subject: v })} />
                  <MetadataField label="Run" value={meta.run} onChange={(v) => updateMetadata({ run: v })} />
                  <MetadataField label="Task Type" value={meta.taskType} onChange={(v) => updateMetadata({ taskType: v })} />
                  <MetadataField label="Condition" value={meta.movementCondition} onChange={(v) => updateMetadata({ movementCondition: v })} />
                  <MetadataField
                    label="Sampling Rate"
                    value={`${meta.samplingRateHz}`}
                    onChange={(v) => {
                      const n = parseInt(v, 10);
                      if (!isNaN(n)) updateMetadata({ samplingRateHz: n });
                    }}
                    hint={experiment?.eeg?.autoDetected ? "auto" : undefined}
                  />
                  <MetadataField
                    label="Channels"
                    value={String(meta.channels)}
                    onChange={(v) => {
                      const n = parseInt(v, 10);
                      if (!isNaN(n)) updateMetadata({ channels: n });
                    }}
                    hint={experiment?.eeg?.autoDetected ? "auto" : undefined}
                  />
                </div>
              </Collapsible>
            </div>
          )}
        </FileGroup>

        {!experiment?.isDemo && (
          <Button variant="secondary" size="sm" className="w-full" onClick={loadDemo}>
            Load S026 Demo
          </Button>
        )}

        {experiment && (
          <div className="pt-1 border-t border-default/60">
            {!confirmClear ? (
              <button
                type="button"
                onClick={() => setConfirmClear(true)}
                className="w-full text-[11px] text-signal-error/90 hover:text-signal-error py-1.5 transition-colors font-medium"
              >
                Clear Experiment
              </button>
            ) : (
              <div className="space-y-1.5">
                <p className="text-[10px] text-muted leading-relaxed">
                  Remove all files, analysis, and explorer state?
                </p>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1 !border-signal-error/40 !text-signal-error"
                    onClick={() => {
                      clearExperiment();
                      setConfirmClear(false);
                    }}
                  >
                    Confirm clear
                  </Button>
                  <Button size="sm" variant="ghost" className="flex-1" onClick={() => setConfirmClear(false)}>
                    Cancel
                  </Button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted shrink-0">{label}</dt>
      <dd className="font-mono text-primary truncate text-right">{value}</dd>
    </div>
  );
}

function FileGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="space-y-1.5">
      <p className="text-[10px] font-medium text-muted uppercase tracking-wide">{title}</p>
      {children}
    </div>
  );
}

function FormatHints({ formats }: { formats: { label: string; active: boolean }[] }) {
  return (
    <div className="flex gap-1 mt-1.5 flex-wrap">
      {formats.map((f) => (
        <span
          key={f.label}
          className={clsx(
            "text-[9px] px-1 py-px rounded font-mono",
            f.active ? "text-accent/90" : "text-muted/50",
          )}
        >
          {f.label}
        </span>
      ))}
    </div>
  );
}

function FileRow({
  file,
  tone,
  onRemove,
}: {
  file: ExperimentFile;
  tone: "eeg" | "meta";
  onRemove: () => void;
}) {
  return (
    <li className="flex items-center gap-2 py-1 px-0.5">
      <span
        className={clsx(
          "w-1.5 h-1.5 rounded-full shrink-0",
          tone === "eeg" ? "bg-signal-eeg" : "bg-signal-meta",
          file.status === "error" && "bg-signal-error",
        )}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <p className="text-[11px] font-mono text-primary truncate leading-tight">{file.name}</p>
        <p className="text-[9px] text-muted">
          {formatBytes(file.sizeBytes)}
          {file.status === "uploading" && ` · ${file.progress ?? 0}%`}
          {file.status === "ready" && " · ready"}
          {file.status === "error" && ` · ${file.error ?? "error"}`}
        </p>
        {file.status === "uploading" && (
          <div className="mt-1 h-0.5 rounded-full bg-muted overflow-hidden">
            <div className="h-full bg-accent/70" style={{ width: `${file.progress ?? 0}%` }} />
          </div>
        )}
      </div>
      <button type="button" onClick={onRemove} className="text-[10px] text-muted hover:text-signal-error" aria-label={`Remove ${file.name}`}>
        ✕
      </button>
    </li>
  );
}

function MetadataField({
  label,
  value,
  onChange,
  hint,
  readOnly,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  hint?: string;
  readOnly?: boolean;
}) {
  const id = `meta-${label.toLowerCase().replace(/\s+/g, "-")}`;
  return (
    <div>
      <label htmlFor={id} className="text-[10px] text-muted flex items-center gap-1.5 mb-0.5">
        {label}
        {hint && <span className="text-accent/80">({hint})</span>}
      </label>
      <input
        id={id}
        type="text"
        value={value}
        readOnly={readOnly}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-muted/40 border border-default rounded px-2 py-1 text-xs text-primary focus:outline-none focus:border-accent/50 read-only:opacity-70"
      />
    </div>
  );
}
