"use client";

import clsx from "clsx";
import { useId, useState, type ReactNode } from "react";

export function Collapsible({
  title,
  children,
  defaultOpen = false,
  compact,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();

  return (
    <div className="border border-default rounded-lg overflow-hidden bg-elevated/50">
      <button
        type="button"
        id={`${id}-trigger`}
        aria-expanded={open}
        aria-controls={`${id}-content`}
        onClick={() => setOpen(!open)}
        className={clsx(
          "w-full flex items-center justify-between text-left hover:bg-muted/60 transition-colors",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent/40",
          compact ? "px-3 py-2" : "px-4 py-2.5",
        )}
      >
        <span className="text-[11px] font-medium tracking-wide text-secondary">
          {title}
        </span>
        <span className="text-muted text-sm leading-none" aria-hidden>
          {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <div
          id={`${id}-content`}
          role="region"
          aria-labelledby={`${id}-trigger`}
          className={clsx("border-t border-default", compact ? "p-3" : "p-4")}
        >
          {children}
        </div>
      )}
    </div>
  );
}
