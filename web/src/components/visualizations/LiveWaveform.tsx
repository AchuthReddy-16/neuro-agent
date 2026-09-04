"use client";

/**
 * Real EEG waveform display.
 * Production UI only shows a static plot image when available.
 * Simulated canvas animation is gated behind allowSimulated (dev/tests only).
 */

import { useCallback, useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { WAVEFORM_CHANNELS } from "@/lib/constants";
import { Button } from "@/components/ui/Button";

interface LiveWaveformProps {
  /** Real waveform plot from sample / analysis — required for production display */
  staticImageUrl?: string;
  enabled?: boolean;
  /**
   * DEV/TEST ONLY — synthetic canvas animation.
   * Must never be set from production Neural Explorer paths.
   */
  allowSimulated?: boolean;
}

export function LiveWaveform({
  staticImageUrl,
  enabled = true,
  allowSimulated = false,
}: LiveWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(true);
  const [selectedChannels, setSelectedChannels] = useState<string[]>(
    WAVEFORM_CHANNELS.slice(0, 4),
  );
  const offsetRef = useRef(0);

  const toggleChannel = (ch: string) => {
    setSelectedChannels((prev) =>
      prev.includes(ch)
        ? prev.length > 1
          ? prev.filter((c) => c !== ch)
          : prev
        : [...prev, ch],
    );
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !playing) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    const dpr = devicePixelRatio;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle =
      getComputedStyle(document.documentElement).getPropertyValue("--surface-elevated") ||
      "#141d2e";
    ctx.fillRect(0, 0, w, h);

    const labelW = 36;
    const plotW = w - labelW - 8;
    const chH = h / selectedChannels.length;

    ctx.strokeStyle =
      getComputedStyle(document.documentElement).getPropertyValue("--border") || "#2a3a52";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const x = labelW + (plotW * i) / 4;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    const colors = ["#4ade80", "#60a5fa", "#a78bfa", "#fbbf24", "#f87171", "#2dd4bf", "#818cf8"];

    selectedChannels.forEach((ch, idx) => {
      const yMid = chH * idx + chH / 2;
      const amp = chH * 0.32;
      const seed = ch.charCodeAt(0) + ch.charCodeAt(1);

      ctx.fillStyle =
        getComputedStyle(document.documentElement).getPropertyValue("--text-muted") || "#7a8fa6";
      ctx.font = "10px var(--font-geist-mono), monospace";
      ctx.textAlign = "right";
      ctx.fillText(ch, labelW - 4, yMid + 3);

      ctx.beginPath();
      ctx.strokeStyle = colors[idx % colors.length];
      ctx.lineWidth = 1.2;

      for (let x = 0; x < plotW; x++) {
        const t = (x + offsetRef.current) * 0.04 + seed;
        const y =
          yMid +
          Math.sin(t) * amp * 0.5 +
          Math.sin(t * 1.7 + seed) * amp * 0.25 +
          Math.sin(t * 3.1) * amp * 0.1;
        if (x === 0) ctx.moveTo(labelW + x, y);
        else ctx.lineTo(labelW + x, y);
      }
      ctx.stroke();

      ctx.strokeStyle =
        getComputedStyle(document.documentElement).getPropertyValue("--border") || "#2a3a52";
      ctx.beginPath();
      ctx.moveTo(0, chH * (idx + 1));
      ctx.lineTo(w, chH * (idx + 1));
      ctx.stroke();
    });

    ctx.fillStyle =
      getComputedStyle(document.documentElement).getPropertyValue("--text-muted") || "#7a8fa6";
    ctx.font = "9px var(--font-geist-mono), monospace";
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const sec = ((i / 4) * 2).toFixed(1);
      ctx.fillText(`${sec}s`, labelW + (plotW * i) / 4, h - 4);
    }

    offsetRef.current += 1.5;
  }, [playing, selectedChannels]);

  useEffect(() => {
    if (!enabled || staticImageUrl || !allowSimulated) return;
    let id: number;
    const loop = () => {
      draw();
      id = requestAnimationFrame(loop);
    };
    if (playing) loop();
    return () => cancelAnimationFrame(id);
  }, [draw, playing, enabled, staticImageUrl, allowSimulated]);

  // Production path: real static plot only
  if (staticImageUrl) {
    return (
      <div className="relative w-full h-full flex items-center justify-center bg-elevated rounded-lg overflow-hidden border border-strong">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={staticImageUrl}
          alt="EEG waveform"
          className="max-w-full max-h-full object-contain"
        />
      </div>
    );
  }

  // Simulated path — unreachable from production explorer (allowSimulated never set)
  if (!allowSimulated) {
    return null;
  }

  return (
    <div className="flex flex-col h-full gap-3">
      <div className="flex items-center justify-between gap-2 flex-wrap px-1">
        <div className="flex items-center gap-1.5 flex-wrap" role="group" aria-label="Channel selector">
          {WAVEFORM_CHANNELS.map((ch) => (
            <button
              key={ch}
              type="button"
              onClick={() => toggleChannel(ch)}
              aria-pressed={selectedChannels.includes(ch)}
              className={clsx(
                "px-2 py-0.5 text-[10px] font-mono rounded border transition-colors",
                selectedChannels.includes(ch)
                  ? "bg-accent/12 border-accent/40 text-accent"
                  : "border-default text-muted hover:text-secondary hover:border-strong",
              )}
            >
              {ch}
            </button>
          ))}
        </div>
        <Button size="sm" variant="ghost" onClick={() => setPlaying(!playing)}>
          {playing ? "Pause" : "Play"}
        </Button>
      </div>
      <div className="flex-1 min-h-[280px] relative overflow-hidden bg-elevated">
        <canvas ref={canvasRef} className="w-full h-full" aria-label="Simulated EEG waveform" />
      </div>
      <p className="text-[10px] text-muted text-center">
        Simulated multi-channel EEG — visualization only (dev)
      </p>
    </div>
  );
}
