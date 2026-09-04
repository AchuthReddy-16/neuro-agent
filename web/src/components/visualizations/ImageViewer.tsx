"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import clsx from "clsx";
import { Button } from "@/components/ui/Button";

interface ImageViewerProps {
  src: string;
  alt: string;
  title?: string;
  onPrev?: () => void;
  onNext?: () => void;
  hasPrev?: boolean;
  hasNext?: boolean;
  /** @deprecated Prefer toolbar render prop */
  controls?: ReactNode;
  toolbar?: (viewControls: ReactNode) => ReactNode;
}

export function ImageViewer({
  src,
  alt,
  title,
  onPrev,
  onNext,
  hasPrev,
  hasNext,
  controls,
  toolbar,
}: ImageViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, ox: 0, oy: 0 });

  const resetView = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  const zoomIn = () => setScale((s) => Math.min(s + 0.25, 4));
  const zoomOut = () => setScale((s) => Math.max(s - 0.25, 0.5));

  const onWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();
    setScale((s) => Math.max(0.5, Math.min(4, s - e.deltaY * 0.001)));
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [onWheel]);

  const onMouseDown = (e: React.MouseEvent) => {
    setDragging(true);
    dragStart.current = { x: e.clientX, y: e.clientY, ox: offset.x, oy: offset.y };
  };

  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragging) return;
    setOffset({
      x: dragStart.current.ox + (e.clientX - dragStart.current.x),
      y: dragStart.current.oy + (e.clientY - dragStart.current.y),
    });
  };

  const toggleFullscreen = async () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      await containerRef.current.requestFullscreen();
      setFullscreen(true);
    } else {
      await document.exitFullscreen();
      setFullscreen(false);
    }
  };

  const viewControls = (
    <>
      <Button size="sm" variant="ghost" onClick={zoomOut} aria-label="Zoom out" className="!px-1.5 !py-0.5 text-muted">
        −
      </Button>
      <span className="text-[10px] text-muted w-9 text-center font-mono tabular-nums">
        {Math.round(scale * 100)}%
      </span>
      <Button size="sm" variant="ghost" onClick={zoomIn} aria-label="Zoom in" className="!px-1.5 !py-0.5 text-muted">
        +
      </Button>
      <Button size="sm" variant="ghost" onClick={resetView} className="!px-1.5 !py-0.5 text-[10px] text-muted">
        Reset
      </Button>
      <Button size="sm" variant="ghost" onClick={toggleFullscreen} className="!px-1.5 !py-0.5 text-[10px] text-muted">
        {fullscreen ? "Exit" : "Full"}
      </Button>
      {onPrev && (
        <Button size="sm" variant="ghost" onClick={onPrev} disabled={!hasPrev} aria-label="Previous" className="!px-1.5 !py-0.5 text-muted">
          ←
        </Button>
      )}
      {onNext && (
        <Button size="sm" variant="ghost" onClick={onNext} disabled={!hasNext} aria-label="Next" className="!px-1.5 !py-0.5 text-muted">
          →
        </Button>
      )}
    </>
  );

  return (
    <div className="flex flex-col h-full min-h-0 gap-1.5">
      {title && <p className="text-xs font-medium text-secondary px-0.5">{title}</p>}

      {toolbar ? toolbar(viewControls) : controls}

      {!toolbar && (
        <div className="flex items-center justify-end gap-0.5 px-0.5">
          {viewControls}
        </div>
      )}

      <div
        ref={containerRef}
        className={clsx(
          "flex-1 min-h-[320px] relative overflow-hidden bg-elevated cursor-grab",
          dragging && "cursor-grabbing",
        )}
        onMouseDown={onMouseDown}
        onMouseUp={() => setDragging(false)}
        onMouseLeave={() => setDragging(false)}
        onMouseMove={onMouseMove}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={alt}
          draggable={false}
          className="absolute top-1/2 left-1/2 select-none max-w-full max-h-full object-contain"
          style={{
            transform: `translate(calc(-50% + ${offset.x}px), calc(-50% + ${offset.y}px)) scale(${scale})`,
            transition: dragging ? "none" : "transform 0.12s ease-out",
          }}
        />
      </div>
    </div>
  );
}
