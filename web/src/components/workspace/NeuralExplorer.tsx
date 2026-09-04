"use client";

import clsx from "clsx";
import { useMemo, useRef, type ReactNode } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { EEG_ACCEPT, EXPLORER_TABS } from "@/lib/constants";
import type { Visualization, VisualizationTab } from "@/lib/types";
import {
  emptyStateMessage,
  type AnalysisResultsState,
  type TypedAnalysisResult,
} from "@/lib/analysis-results";
import { LiveWaveform } from "@/components/visualizations/LiveWaveform";
import { ImageViewer } from "@/components/visualizations/ImageViewer";
import { EmptyState, EmptyStateButton } from "@/components/ui/EmptyState";
import { ExplorerSkeleton } from "@/components/ui/Skeleton";
import { ExperimentSummaryStrip } from "./ExperimentSummaryStrip";

const BANDS = ["Delta", "Theta", "Alpha/Mu", "Beta"];
const CONDITIONS = ["Left Fist", "Right Fist", "Both Feet", "Rest"];

function SelectControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  const id = `viz-${label.toLowerCase().replace(/\s+/g, "-")}`;
  return (
    <label htmlFor={id} className="inline-flex items-center gap-1.5 text-[11px] text-muted">
      <span>{label}</span>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-transparent border-0 border-b border-default text-primary text-[11px] font-mono py-0.5 focus:outline-none focus:border-accent"
      >
        {options.map((o) => (
          <option key={o} value={o} className="bg-elevated text-primary">
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function VizToolbar({
  viz,
  onChange,
  viewControls,
}: {
  viz: Visualization;
  onChange: (patch: Partial<Visualization>) => void;
  viewControls?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 py-1.5 px-0.5">
      <SelectControl
        label="Channel"
        value={viz.channel ?? "C3"}
        options={["C3", "C4", "CZ", "FC3", "FC4", "CP3"]}
        onChange={(v) => onChange({ channel: v })}
      />
      <SelectControl
        label="Band"
        value={viz.band ?? "Beta"}
        options={BANDS}
        onChange={(v) => onChange({ band: v })}
      />
      <SelectControl
        label="Condition"
        value={viz.condition ?? "Left Fist"}
        options={CONDITIONS}
        onChange={(v) => onChange({ condition: v })}
      />
      {viz.tab === "comparison" && (
        <SelectControl
          label="Compare"
          value={viz.compareWith ?? "Right Fist"}
          options={CONDITIONS}
          onChange={(v) => onChange({ compareWith: v })}
        />
      )}
      {viewControls && <div className="ml-auto flex items-center gap-0.5">{viewControls}</div>}
    </div>
  );
}

function PlotMeta({
  title,
  channel,
  band,
  condition,
  compareWith,
  provenanceNote,
}: {
  title?: string | null;
  channel?: string | null;
  band?: string | null;
  condition?: string | null;
  compareWith?: string | null;
  provenanceNote?: string | null;
}) {
  const bits = [
    channel && `Ch ${channel}`,
    band,
    condition,
    compareWith && `vs ${compareWith}`,
  ].filter(Boolean);

  return (
    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[10px] text-muted px-0.5">
      {title && <span className="text-secondary">{title}</span>}
      {bits.map((b) => (
        <span
          key={String(b)}
          className="font-mono before:content-['·'] before:mr-2.5 before:text-border-strong"
        >
          {b}
        </span>
      ))}
      {provenanceNote && (
        <span className="text-muted/70 before:content-['·'] before:mr-2.5">{provenanceNote}</span>
      )}
    </div>
  );
}

function TabEmpty({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-6">
      <p className="text-sm text-primary mb-1">{title}</p>
      <p className="text-xs text-muted max-w-sm leading-relaxed">{body}</p>
    </div>
  );
}

function TabLoading({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-6">
      <p className="text-sm text-primary mb-1">{label}</p>
      <p className="text-xs text-muted">Computing…</p>
    </div>
  );
}

function TabError({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-6">
      <p className="text-sm text-signal-error mb-1">Analysis error</p>
      <p className="text-xs text-muted max-w-sm leading-relaxed">{message}</p>
    </div>
  );
}

function resultForTab(
  results: AnalysisResultsState,
  tab: VisualizationTab,
): TypedAnalysisResult<Record<string, unknown>> {
  const cast = (r: TypedAnalysisResult<unknown>) =>
    r as unknown as TypedAnalysisResult<Record<string, unknown>>;
  switch (tab) {
    case "waveform":
      return cast(results.waveform);
    case "spectrogram":
      return cast(results.spectrogram);
    case "psd":
      return cast(results.psd);
    case "band_power":
      return cast(results.bandPower);
    case "topomap":
      return cast(results.topomap);
    case "comparison":
      return cast(results.comparison);
    default:
      return cast(results.waveform);
  }
}

export function NeuralExplorer() {
  const {
    experiment,
    activeTab,
    setActiveTab,
    loadDemo,
    uploadEEG,
    uploadFigure,
    explorerLoading,
    explorerError,
    selectedImage,
    analysisResults,
    visionState,
  } = useExperiment();
  const eegInputRef = useRef<HTMLInputElement>(null);
  const figInputRef = useRef<HTMLInputElement>(null);

  const hasEeg = Boolean(experiment?.eeg_files.some((f) => f.status === "ready") || experiment?.eeg);
  const hasImages = Boolean(experiment?.image_files.some((f) => f.status === "ready"));
  const imageOnly = hasImages && !hasEeg;
  const tabResult = useMemo(
    () => resultForTab(analysisResults, activeTab),
    [analysisResults, activeTab],
  );

  const waveformStaticUrl =
    tabResult.status === "ready" &&
    tabResult.payload &&
    tabResult.payload.kind === "static_plot" &&
    typeof tabResult.payload.imageUrl === "string"
      ? (tabResult.payload.imageUrl as string)
      : undefined;

  if (explorerLoading) {
    return <ExplorerSkeleton />;
  }

  if (!experiment) {
    return (
      <EmptyState
        title="Load an experiment to begin"
        description="Upload sample JSON + optional figures, or load the linked live sample."
        actions={
          <>
            <EmptyStateButton onClick={() => eegInputRef.current?.click()}>
              Upload sample JSON
            </EmptyStateButton>
            <input
              ref={eegInputRef}
              type="file"
              accept={EEG_ACCEPT}
              className="hidden"
              aria-hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void uploadEEG(f);
              }}
            />
            <EmptyStateButton onClick={loadDemo} variant="primary">
              Load linked sample
            </EmptyStateButton>
          </>
        }
      />
    );
  }

  if (explorerError) {
    return (
      <EmptyState
        title="Visualization error"
        description={explorerError}
        actions={
          <EmptyStateButton onClick={loadDemo} variant="primary">
            Reload sample
          </EmptyStateButton>
        }
      />
    );
  }

  const hasMeta = experiment.metadata_files.length > 0;
  const onlyMeta =
    hasMeta && !hasEeg && !hasImages && Object.values(analysisResults).every((r) => r.status === "idle");

  if (onlyMeta) {
    return (
      <EmptyState
        title="Metadata loaded"
        description="Add EEG for explorer analysis tabs, or a figure for vision questions. Text-only questions still work."
        actions={
          <>
            <EmptyStateButton onClick={() => eegInputRef.current?.click()}>Upload EEG</EmptyStateButton>
            <EmptyStateButton onClick={() => figInputRef.current?.click()}>Upload figure</EmptyStateButton>
            <input
              ref={eegInputRef}
              type="file"
              accept={EEG_ACCEPT}
              className="hidden"
              aria-hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void uploadEEG(f);
              }}
            />
            <input
              ref={figInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp"
              className="hidden"
              aria-hidden
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void uploadFigure(f);
              }}
            />
          </>
        }
      />
    );
  }

  const emptyCopy = emptyStateMessage(activeTab, { hasEeg, hasImage: hasImages });

  return (
    <div className="flex flex-col h-full gap-2">
      <ExperimentSummaryStrip />

      {/* Vision context is informational only — never drives EEG tab plots */}
      {selectedImage && (
        <div className="space-y-0.5 px-0.5">
          <div className="flex items-baseline gap-2 text-[11px]">
            <span className="text-muted uppercase tracking-wide text-[10px]">
              {imageOnly ? "Uploaded figure — visual interpretation" : "Vision attachment"}
            </span>
            <span className="font-mono text-signal-vision">{selectedImage.name}</span>
            {visionState.interpretation.status === "ready" && (
              <span className="text-[10px] text-muted">· interpretation ready</span>
            )}
          </div>
          {imageOnly && (
            <p className="text-[11px] text-secondary leading-snug">
              Image loaded. You can ask questions about this figure. EEG-derived analyses require
              EEG/sample data.
            </p>
          )}
        </div>
      )}

      {!selectedImage && hasImages && (
        <p className="text-[10px] text-signal-warning px-0.5">
          Figures available — select one in the left panel for vision questions (not for EEG tabs).
        </p>
      )}

      <div
        className="flex gap-0 overflow-x-auto shrink-0 border-b border-default"
        role="tablist"
        aria-label="Analysis result type"
      >
        {EXPLORER_TABS.map((tab) => {
          const r = resultForTab(analysisResults, tab.id);
          const ready = r.status === "ready";
          const needsInput = r.status === "idle";
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                "px-3 py-2 text-[11px] font-medium whitespace-nowrap transition-colors duration-150 border-b-2 -mb-px",
                activeTab === tab.id
                  ? "text-accent border-accent"
                  : "text-muted border-transparent hover:text-secondary",
              )}
            >
              {tab.label}
              {ready && <span className="ml-1 text-[9px] text-accent/70">●</span>}
              {needsInput && !ready && activeTab !== tab.id && (
                <span className="ml-1 text-[9px] text-muted/50" title="Requires input">
                  ○
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="flex-1 min-h-0 flex flex-col">
        {tabResult.status === "loading" ? (
          <TabLoading
            label={
              activeTab === "psd"
                ? "Computing PSD…"
                : activeTab === "spectrogram"
                  ? "Computing spectrogram…"
                  : activeTab === "band_power"
                    ? "Computing band power…"
                    : `Loading ${activeTab.replace("_", " ")}`
            }
          />
        ) : tabResult.status === "error" ? (
          <TabError message={tabResult.error || "Result unavailable."} />
        ) : activeTab === "waveform" && waveformStaticUrl ? (
          <LiveWaveform staticImageUrl={waveformStaticUrl} enabled />
        ) : activeTab === "band_power" && tabResult.status === "ready" ? (
          <BandPowerPanel result={tabResult} />
        ) : activeTab === "comparison" && tabResult.status === "ready" ? (
          <ComparisonPanel result={tabResult} />
        ) : tabResult.status === "ready" &&
          tabResult.payload &&
          typeof tabResult.payload.imageUrl === "string" &&
          tabResult.payload.imageUrl ? (
          <ImageViewer
            src={tabResult.payload.imageUrl}
            alt={
              (typeof tabResult.payload.title === "string" && tabResult.payload.title) ||
              `${activeTab} plot`
            }
            toolbar={(viewControls) => {
              const p = tabResult.payload || {};
              const viz: Visualization = {
                id: (p.visualizationId as string) || `typed-${activeTab}`,
                tab: activeTab,
                title: (p.title as string) || activeTab,
                imageUrl: String(p.imageUrl),
                index: 0,
                channel: p.channel as string | undefined,
                band: p.band as string | undefined,
                condition: p.condition as string | undefined,
                compareWith: p.compareWith as string | undefined,
              };
              return (
                <VizToolbar
                  viz={viz}
                  onChange={() => {
                    /* display-only controls — do not mutate typed results */
                  }}
                  viewControls={viewControls}
                />
              );
            }}
          />
        ) : (
          <TabEmpty title={emptyCopy.title} body={emptyCopy.body} />
        )}
      </div>

      {tabResult.status === "ready" && (activeTab !== "waveform" || Boolean(waveformStaticUrl)) && (
        <PlotMeta
          title={
            tabResult.payload && typeof tabResult.payload.title === "string"
              ? tabResult.payload.title
              : activeTab.replace("_", " ")
          }
          channel={tabResult.provenance.channel}
          band={tabResult.provenance.band}
          condition={
            tabResult.provenance.conditionA ||
            (tabResult.payload && typeof tabResult.payload.condition === "string"
              ? tabResult.payload.condition
              : undefined)
          }
          compareWith={
            tabResult.provenance.conditionB ||
            (tabResult.payload && typeof tabResult.payload.compareWith === "string"
              ? tabResult.payload.compareWith
              : undefined)
          }
          provenanceNote={tabResult.provenance.source || undefined}
        />
      )}
    </div>
  );
}

function BandPowerPanel({
  result,
}: {
  result: TypedAnalysisResult<Record<string, unknown>>;
}) {
  const payload = result.payload;
  const imageUrl = typeof payload?.imageUrl === "string" ? payload.imageUrl : null;
  const rowsRaw = payload?.rows;
  const ranking = payload?.ranking as string[] | undefined;
  const values = payload?.values as Record<string, number | string> | undefined;
  const units = typeof payload?.units === "string" ? payload.units : undefined;

  if (imageUrl && (!Array.isArray(rowsRaw) || rowsRaw.length === 0)) {
    return (
      <ImageViewer
        src={imageUrl}
        alt="Band power"
        toolbar={(viewControls) => <div className="flex justify-end py-1">{viewControls}</div>}
      />
    );
  }

  const rows: { label: string; value: string; unit?: string; highlight?: boolean }[] =
    Array.isArray(rowsRaw) && rowsRaw.length
      ? (rowsRaw as { label: string; value: string; unit?: string; highlight?: boolean }[])
      : (ranking || []).map((ch, i) => ({
          label: `Rank ${i + 1} · ${ch}`,
          value: String(values?.[ch] ?? ch),
          unit: units,
          highlight: i === 0,
        }));

  if (!rows.length && !imageUrl) {
    return <TabEmpty title="No band-power rows" body="No computed band-power values yet." />;
  }

  return (
    <div className="flex flex-col h-full min-h-0 gap-2 p-2">
      {imageUrl && (
        <div className="max-h-[40%] min-h-0">
          <ImageViewer
            src={imageUrl}
            alt="Band power plot"
            toolbar={(viewControls) => <div className="flex justify-end py-1">{viewControls}</div>}
          />
        </div>
      )}
      <div className="overflow-auto border border-default/60 rounded-md">
        <table className="w-full text-[12px]">
          <tbody>
            {rows.map((row, idx) => (
              <tr key={`${row.label}-${idx}`} className="border-b border-default/50 last:border-0">
                <td className="py-1.5 px-2 text-muted">{row.label}</td>
                <td
                  className={clsx(
                    "py-1.5 px-2 text-right font-mono tabular-nums",
                    row.highlight ? "text-signal-eeg font-semibold" : "text-primary",
                  )}
                >
                  {row.value}
                  {row.unit && <span className="text-muted ml-1 font-sans text-[10px]">{row.unit}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonPanel({
  result,
}: {
  result: TypedAnalysisResult<Record<string, unknown>>;
}) {
  const payload = result.payload;
  const imageUrl = typeof payload?.imageUrl === "string" ? payload.imageUrl : null;
  const conditionA = typeof payload?.conditionA === "string" ? payload.conditionA : undefined;
  const conditionB = typeof payload?.conditionB === "string" ? payload.conditionB : undefined;
  const winner = typeof payload?.winner === "string" ? payload.winner : undefined;
  const summary = typeof payload?.summary === "string" ? payload.summary : undefined;

  return (
    <div className="flex flex-col h-full min-h-0 gap-2 p-2">
      {imageUrl && (
        <div className="max-h-[45%] min-h-0">
          <ImageViewer
            src={imageUrl}
            alt="Comparison"
            toolbar={(viewControls) => <div className="flex justify-end py-1">{viewControls}</div>}
          />
        </div>
      )}
      <div className="space-y-2 text-[12px]">
        {(conditionA || conditionB) && (
          <p className="text-secondary">
            {conditionA || "A"} vs {conditionB || "B"}
            {winner ? ` · higher: ${winner}` : ""}
          </p>
        )}
        {summary && <p className="text-primary leading-relaxed">{summary}</p>}
        {(payload?.valueA != null || payload?.valueB != null) && (
          <p className="font-mono text-[11px] text-muted">
            {conditionA || "A"}={String(payload?.valueA)} · {conditionB || "B"}=
            {String(payload?.valueB)}
          </p>
        )}
        {result.provenance.sampleIdA && (
          <p className="text-[10px] text-muted font-mono">
            A: {result.provenance.sampleIdA}
            {result.provenance.sampleIdB ? ` · B: ${result.provenance.sampleIdB}` : ""}
          </p>
        )}
      </div>
    </div>
  );
}
