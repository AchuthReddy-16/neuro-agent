"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { useExperiment } from "@/lib/store/experiment-context";
import { ModalityBadges } from "./ModalityBadges";
import { ExperimentPanel } from "./ExperimentPanel";
import { NeuralExplorer } from "./NeuralExplorer";
import { ResearchAgentPanel } from "./ResearchAgentPanel";
import { AnalysisToolsPanel } from "./AnalysisToolsPanel";
import { SystemPanel } from "./SystemPanel";
import { SettingsPanel } from "./SettingsPanel";
import { SystemStatusStrip } from "./SystemStatusStrip";

export function WorkspaceLayout() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { beginWorkspaceSession } = useExperiment();
  const booted = useRef(false);

  // Legacy link /workspace?demo=1 → dedicated Interactive Demo
  useEffect(() => {
    if (searchParams.get("demo") === "1") {
      router.replace("/demo");
    }
  }, [searchParams, router]);

  useEffect(() => {
    if (booted.current) return;
    booted.current = true;
    beginWorkspaceSession();
  }, [beginWorkspaceSession]);

  return (
    <div className="h-screen flex flex-col bg-surface overflow-hidden">
      <header className="shrink-0 flex items-center justify-between gap-4 px-4 xl:px-5 py-2 border-b border-default bg-elevated">
        <div className="flex items-center gap-2.5 min-w-0">
          <Link
            href="/"
            className="flex items-center gap-2.5 hover:opacity-85 transition-opacity min-w-0"
          >
            <div className="w-7 h-7 rounded-md bg-accent/12 border border-accent/25 flex items-center justify-center shrink-0">
              <span className="text-accent font-mono text-xs">Ψ</span>
            </div>
            <div className="min-w-0">
              <h1 className="text-[13px] font-semibold text-primary leading-tight truncate tracking-tight">
                Research Workspace
              </h1>
              <p className="text-[10px] text-muted hidden xl:block leading-tight mt-0.5">
                Experiment context · neural explorer · research agent
              </p>
            </div>
          </Link>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Link href="/demo">
            <span className="text-[10px] text-muted hover:text-secondary hidden sm:inline px-2">
              Interactive Demo
            </span>
          </Link>
          <ModalityBadges />
        </div>
      </header>

      <SystemStatusStrip />

      <main className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(220px,20%)_minmax(0,1fr)_minmax(300px,26%)] xl:grid-cols-[minmax(240px,20%)_minmax(0,1fr)_minmax(320px,27%)] 2xl:grid-cols-[minmax(260px,21%)_minmax(0,1fr)_minmax(340px,26%)] gap-0 overflow-hidden">
        <aside className="border-b lg:border-b-0 lg:border-r border-default overflow-y-auto max-h-[38vh] lg:max-h-none order-2 lg:order-1 bg-elevated/40">
          <div className="p-2.5 space-y-2 h-full">
            <ExperimentPanel />
            <AnalysisToolsPanel />
            <SystemPanel />
            <SettingsPanel />
          </div>
        </aside>

        <section className="flex flex-col min-h-[50vh] lg:min-h-0 order-1 lg:order-2 border-b lg:border-b-0 lg:border-r border-default bg-canvas">
          <div className="shrink-0 flex items-center justify-between px-4 py-2 border-b border-default/80">
            <h2 className="text-[11px] font-semibold tracking-[0.14em] text-secondary uppercase">
              Neural Data Explorer
            </h2>
            <span className="text-[10px] text-muted hidden sm:inline">
              Subject · channel · band context
            </span>
          </div>
          <div className="flex-1 min-h-0 px-3 py-2.5 lg:px-4 lg:py-3">
            <NeuralExplorer />
          </div>
        </section>

        <aside className="overflow-y-auto min-h-[38vh] lg:min-h-0 order-3 bg-elevated/30">
          <div className="p-2.5 h-full">
            <ResearchAgentPanel />
          </div>
        </aside>
      </main>
    </div>
  );
}
