"use client";

import { useState } from "react";
import { useExperiment } from "@/lib/store/experiment-context";
import { ANALYSIS_TOOLS } from "@/lib/constants";
import { Collapsible } from "@/components/ui/Collapsible";

export function AnalysisToolsPanel() {
  const { runTool } = useExperiment();
  const [selected, setSelected] = useState<string[]>([]);

  const toggle = (tool: string) => {
    setSelected((prev) =>
      prev.includes(tool) ? prev.filter((t) => t !== tool) : [...prev, tool],
    );
  };

  return (
    <Collapsible title="Analysis tools" compact>
      <p className="text-[10px] text-muted mb-2 leading-relaxed">
        Manual selection — the agent invokes tools automatically in production.
      </p>
      <div className="flex flex-col">
        {ANALYSIS_TOOLS.map((tool) => {
          const active = selected.includes(tool);
          return (
            <button
              key={tool}
              type="button"
              onClick={() => {
                toggle(tool);
                runTool(tool);
              }}
              className={`text-left text-[11px] py-1.5 border-b border-default/40 last:border-0 transition-colors ${
                active ? "text-accent" : "text-secondary hover:text-primary"
              }`}
            >
              {tool}
            </button>
          );
        })}
      </div>
    </Collapsible>
  );
}
