"use client";

import { useExperiment } from "@/lib/store/experiment-context";

export function ExperimentSummaryStrip() {
  const { experiment, currentAnswer, isAnalyzing } = useExperiment();

  if (!experiment) return null;

  const { metadata } = experiment;
  const isDemo = !!experiment.isDemo;

  const conditionLabel = metadata.taskType
    ? `${metadata.movementCondition} · ${metadata.taskType}`
    : metadata.movementCondition;

  const channelFromAnswer = currentAnswer?.computedEvidence?.find(
    (e) =>
      e.label.toLowerCase().includes("channel") ||
      e.label.toLowerCase().includes("discriminative") ||
      e.label.toLowerCase().includes("highest"),
  );

  const evidenceStatus = isAnalyzing
    ? "Analyzing"
    : currentAnswer
      ? "Answer ready"
      : "Loaded";

  const items = [
    { label: "Subject", value: metadata.subject || "—" },
    { label: "Run", value: metadata.run || "—" },
    { label: "Condition", value: conditionLabel || "—" },
    {
      label: "Channel",
      value: channelFromAnswer?.value ?? (isDemo ? "C3" : "—"),
    },
    { label: "Status", value: evidenceStatus },
  ];

  return (
    <div
      className="flex flex-wrap items-baseline gap-x-5 gap-y-1 px-0.5 shrink-0"
      role="region"
      aria-label="Experiment summary"
    >
      {items.map((item) => (
        <div key={item.label} className="flex items-baseline gap-1.5 min-w-0">
          <span className="text-[10px] text-muted uppercase tracking-wide">{item.label}</span>
          <span className="text-[11px] font-mono text-primary truncate">{item.value}</span>
        </div>
      ))}
      {isDemo && (
        <span className="text-[10px] text-muted/70 ml-auto">sample</span>
      )}
    </div>
  );
}
