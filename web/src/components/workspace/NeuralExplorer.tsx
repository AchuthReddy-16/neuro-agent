"use client";

import clsx from "clsx";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { EEG_ACCEPT, EXPLORER_TABS } from "@/lib/constants";
import type { Visualization } from "@/lib/types";
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

function PlotMeta({ viz, isDemo }: { viz: Visualization | null; isDemo?: boolean }) {
  if (!viz) return null;
  const bits = [
    viz.channel && `Ch ${viz.channel}`,
    viz.band && viz.band,
    viz.condition,
    viz.compareWith && `vs ${viz.compareWith}`,
  ].filter(Boolean);

  return (
    <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[10px] text-muted">
      <span className="text-secondary">{viz.title}</span>
      {bits.map((b) => (
        <span key={String(b)} className="font-mono before:content-['·'] before:mr-2.5 before:text-border-strong">
          {b}
        </span>
      ))}
      {isDemo && <span className="text-muted/70 before:content-['·'] before:mr-2.5">sample</span>}
    </div>
  );
}

export function NeuralExplorer() {
  const {
    experiment,
    activeTab,
    setActiveTab,
    focusedVizId,
    loadDemo,
    uploadEEG,
    uploadFigure,
    explorerLoading,
    explorerError,
    selectedImage,
  } = useExperiment();
  const [localViz, setLocalViz] = useState<Visualization | null>(null);
  const eegInputRef = useRef<HTMLInputElement>(null);
  const figInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setLocalViz(null);
  }, [activeTab, focusedVizId, selectedImage?.id]);

  const tabViz = useMemo(() => {
    if (!experiment) return [];
    return experiment.visualizations.filter((v) => v.tab === activeTab);
  }, [experiment, activeTab]);

  const currentViz = useMemo(() => {
    if (!tabViz.length) return null;
    if (focusedVizId) {
      const found = tabViz.find((v) => v.id === focusedVizId);
      if (found) return found;
    }
    return tabViz[0];
  }, [tabViz, focusedVizId]);

  const displayViz = localViz ?? currentViz;
  const currentIdx = displayViz ? tabViz.findIndex((v) => v.id === displayViz.id) : -1;

  // Prefer explicitly selected uploaded/demo figure for non-waveform views
  const selectedUrl = selectedImage?.url;
  const showSelectedFigure = !!selectedUrl && activeTab !== "waveform";

  if (explorerLoading) {
    return <ExplorerSkeleton />;
  }

  if (!experiment) {
    return (
      <EmptyState
        title="Load an experiment to begin"
        description="Upload sample JSON + figures — or open the live demo experiment."
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
              Try Demo
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
            Reload Demo
          </EmptyStateButton>
        }
      />
    );
  }

  const hasEeg = experiment.eeg_files.length > 0 || !!experiment.eeg;
  const hasImages = experiment.image_files.length > 0;
  const hasMeta = experiment.metadata_files.length > 0;
  const onlyMeta = hasMeta && !hasEeg && !hasImages && experiment.visualizations.length === 0;

  if (onlyMeta) {
    return (
      <EmptyState
        title="Metadata loaded"
        description="Add EEG or a figure to populate the Neural Data Explorer. Text-only tool questions can still be asked."
        actions={
          <>
            <EmptyStateButton onClick={() => eegInputRef.current?.click()}>Upload EEG</EmptyStateButton>
            <EmptyStateButton onClick={() => figInputRef.current?.click()}>Upload figure</EmptyStateButton>
            <input ref={eegInputRef} type="file" accept={EEG_ACCEPT} className="hidden" aria-hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void uploadEEG(f); }} />
            <input ref={figInputRef} type="file" accept=".png,.jpg,.jpeg,.webp" className="hidden" aria-hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void uploadFigure(f); }} />
          </>
        }
      />
    );
  }

  const hasVizForTab =
    activeTab === "waveform"
      ? hasEeg || !!displayViz?.imageUrl || !!selectedUrl
      : showSelectedFigure || !!displayViz?.imageUrl;

  const plotSrc = showSelectedFigure
    ? selectedUrl!
    : displayViz?.imageUrl;

  return (
    <div className="flex flex-col h-full gap-2">
      <ExperimentSummaryStrip />

      {selectedImage && (
        <div className="flex items-baseline gap-2 px-0.5 text-[11px]">
          <span className="text-muted uppercase tracking-wide text-[10px]">Selected figure</span>
          <span className="font-mono text-signal-vision">{selectedImage.name}</span>
          {experiment.image_files.length > 1 && (
            <span className="text-[10px] text-muted">
              ({experiment.image_files.length} figures)
            </span>
          )}
        </div>
      )}

      {!selectedImage && hasImages && (
        <p className="text-[10px] text-signal-warning px-0.5">
          Multiple figures available — select one in the left panel for vision context.
        </p>
      )}

      <div
        className="flex gap-0 overflow-x-auto shrink-0 border-b border-default"
        role="tablist"
        aria-label="Visualization type"
      >
        {EXPLORER_TABS.map((tab) => (
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
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0 flex flex-col">
        {!hasVizForTab ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-6">
            <p className="text-sm text-primary mb-1">
              {hasEeg && !hasImages
                ? "EEG loaded — no figure yet"
                : hasImages && !hasEeg
                  ? "Figure loaded"
                  : "No visualization for this tab"}
            </p>
            <p className="text-xs text-muted max-w-sm leading-relaxed">
              {hasEeg && !hasImages
                ? "Waveform is available; upload a figure for topomap/PSD/spectrogram views, or ask a text/tool question."
                : hasImages && !selectedImage
                  ? "Select a figure in the left panel to display it here."
                  : experiment.isDemo
                    ? "This demo fixture does not include this view."
                    : "Upload EEG or a figure, or load the S026 demo."}
            </p>
          </div>
        ) : activeTab === "waveform" ? (
          <LiveWaveform
            staticImageUrl={selectedUrl && !hasEeg ? selectedUrl : displayViz?.imageUrl}
            enabled={hasEeg}
          />
        ) : plotSrc ? (
          <ImageViewer
            src={plotSrc}
            alt={selectedImage?.name ?? displayViz?.title ?? "Figure"}
            hasPrev={!showSelectedFigure && currentIdx > 0}
            hasNext={!showSelectedFigure && currentIdx < tabViz.length - 1}
            onPrev={
              showSelectedFigure
                ? undefined
                : () => {
                    if (currentIdx > 0) setLocalViz(tabViz[currentIdx - 1]);
                  }
            }
            onNext={
              showSelectedFigure
                ? undefined
                : () => {
                    if (currentIdx < tabViz.length - 1) setLocalViz(tabViz[currentIdx + 1]);
                  }
            }
            toolbar={
              displayViz && !showSelectedFigure
                ? (viewControls) => (
                    <VizToolbar
                      viz={displayViz}
                      onChange={(patch) => setLocalViz({ ...displayViz, ...patch })}
                      viewControls={viewControls}
                    />
                  )
                : (viewControls) => (
                    <div className="flex justify-end py-1">{viewControls}</div>
                  )
            }
          />
        ) : null}
      </div>

      {showSelectedFigure ? (
        <div className="text-[10px] text-muted font-mono px-0.5">
          {selectedImage?.name}
          <span className="before:content-['·'] before:mx-2">active figure</span>
        </div>
      ) : (
        <PlotMeta viz={displayViz} isDemo={experiment.isDemo} />
      )}
    </div>
  );
}
