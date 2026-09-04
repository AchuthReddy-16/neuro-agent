"use client";

import { useExperiment } from "@/lib/store/experiment-context";
import { Badge } from "@/components/ui/Badge";
import type { Modality } from "@/lib/types";

const MODALITIES: { key: Modality; label: string; color: "eeg" | "meta" | "vision" | "text" }[] = [
  { key: "eeg", label: "EEG", color: "eeg" },
  { key: "metadata", label: "Meta", color: "meta" },
  { key: "vision", label: "Vision", color: "vision" },
  { key: "text", label: "Text", color: "text" },
];

export function ModalityBadges() {
  const { experiment } = useExperiment();
  const mods = experiment?.modalities ?? {
    eeg: false,
    metadata: false,
    vision: false,
    text: true,
  };

  return (
    <div className="flex items-center gap-1.5 flex-wrap justify-end" aria-label="Active modalities">
      {MODALITIES.map((m) => (
        <Badge key={m.key} color={m.color} active={mods[m.key]} className="!px-2 !py-0.5 !text-[10px] !rounded">
          {m.label}
        </Badge>
      ))}
    </div>
  );
}
