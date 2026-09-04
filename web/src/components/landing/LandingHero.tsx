"use client";

import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { WaveformBackground } from "./WaveformBackground";

export function LandingHero() {
  return (
    <div className="relative min-h-screen flex flex-col overflow-hidden bg-surface landing-vignette">
      <div className="absolute inset-0 bg-gradient-to-b from-accent/6 via-transparent to-surface pointer-events-none" />
      <WaveformBackground />

      <header className="relative z-10 flex items-center justify-between px-6 py-5 max-w-6xl mx-auto w-full">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-md bg-accent/12 border border-accent/25 flex items-center justify-center">
            <span className="text-accent font-mono text-xs">Ψ</span>
          </div>
          <span className="text-sm font-medium text-secondary tracking-tight">Neuro Agent</span>
        </div>
        <Link href="/workspace">
          <Button variant="ghost" size="sm">
            Workspace
          </Button>
        </Link>
      </header>

      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-6 pb-20">
        <div className="max-w-2xl text-center space-y-7 animate-fade-in">
          <p className="text-[11px] font-medium uppercase tracking-[0.18em] text-accent">
            Neuroscience research agent
          </p>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-primary leading-[1.1]">
            Multimodal Neuroscience
            <br />
            <span className="text-accent">Research Agent</span>
          </h1>
          <p className="text-base md:text-lg text-secondary max-w-xl mx-auto leading-relaxed">
            Deterministic EEG tools, evidence-grounded answers, and vision when needed —
            try it quickly, or open the full scientific workstation.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-1">
            <Link href="/demo">
              <Button size="lg">Open Interactive Demo</Button>
            </Link>
            <Link href="/workspace">
              <Button variant="outline" size="lg">
                Enter Workspace
              </Button>
            </Link>
          </div>

          <div className="pt-4 grid sm:grid-cols-2 gap-3 text-left max-w-lg mx-auto">
            <p className="text-[11px] text-muted leading-snug rounded-lg border border-default/60 px-3 py-2">
              <span className="text-secondary font-medium">Interactive Demo</span>
              <br />
              Conversational try-it flow. Attach files with +.
            </p>
            <p className="text-[11px] text-muted leading-snug rounded-lg border border-default/60 px-3 py-2">
              <span className="text-secondary font-medium">Workspace</span>
              <br />
              Full experiment controls, explorer, and evidence.
            </p>
          </div>

          <ul className="pt-4 flex flex-wrap items-center justify-center gap-x-5 gap-y-2 text-[11px] text-muted font-mono">
            <li>Text · Qwen3-4B</li>
            <li className="text-border-strong">·</li>
            <li>Vision · Qwen2.5-VL-3B</li>
            <li className="text-border-strong">·</li>
            <li>INT8 W8A8 · vLLM</li>
          </ul>
        </div>
      </main>

      <footer className="relative z-10 text-center pb-8 text-[11px] text-muted">
        Deployed API via NEXT_PUBLIC_API_BASE_URL
      </footer>
    </div>
  );
}
