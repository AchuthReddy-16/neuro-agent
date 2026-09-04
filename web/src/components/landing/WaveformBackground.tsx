"use client";

import { useEffect, useRef } from "react";

export function WaveformBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame = 0;
    let animId: number;

    const resize = () => {
      canvas.width = canvas.offsetWidth * devicePixelRatio;
      canvas.height = canvas.offsetHeight * devicePixelRatio;
      ctx.scale(devicePixelRatio, devicePixelRatio);
    };
    resize();
    window.addEventListener("resize", resize);

    const channels = 5;
    const colors = [
      "rgba(45, 212, 191, 0.15)",
      "rgba(96, 165, 250, 0.12)",
      "rgba(192, 132, 252, 0.10)",
      "rgba(74, 222, 128, 0.08)",
      "rgba(251, 191, 36, 0.06)",
    ];

    const draw = () => {
      const w = canvas.offsetWidth;
      const h = canvas.offsetHeight;
      ctx.clearRect(0, 0, w, h);

      for (let ch = 0; ch < channels; ch++) {
        const yBase = h * (0.2 + ch * 0.15);
        const amp = 12 + ch * 3;
        const freq = 0.02 + ch * 0.005;
        const phase = frame * 0.03 + ch * 1.2;

        ctx.beginPath();
        ctx.strokeStyle = colors[ch];
        ctx.lineWidth = 1.2;

        for (let x = 0; x < w; x++) {
          const t = x * freq + phase;
          const y =
            yBase +
            Math.sin(t) * amp +
            Math.sin(t * 2.3 + 1) * amp * 0.3 +
            Math.sin(t * 0.7 + 2) * amp * 0.15;
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      frame++;
      animId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full opacity-60"
      aria-hidden
    />
  );
}
