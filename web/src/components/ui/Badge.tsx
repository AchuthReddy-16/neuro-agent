import clsx from "clsx";
import type { ReactNode } from "react";

interface BadgeProps {
  children: ReactNode;
  active?: boolean;
  color?: "eeg" | "meta" | "vision" | "text" | "default";
  className?: string;
}

const activeMap = {
  eeg: "text-signal-eeg border-signal-eeg/40 bg-signal-eeg/12",
  meta: "text-signal-meta border-signal-meta/40 bg-signal-meta/12",
  vision: "text-signal-vision border-signal-vision/40 bg-signal-vision/12",
  text: "text-signal-text border-signal-text/40 bg-signal-text/12",
  default: "text-secondary border-default bg-muted",
};

const inactiveMap = {
  eeg: "text-signal-eeg/70 border-signal-eeg/25 bg-transparent",
  meta: "text-signal-meta/70 border-signal-meta/25 bg-transparent",
  vision: "text-signal-vision/70 border-signal-vision/25 bg-transparent",
  text: "text-signal-text/70 border-signal-text/25 bg-transparent",
  default: "text-muted border-default bg-transparent",
};

export function Badge({ children, active, color = "default", className }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md border transition-colors",
        active ? activeMap[color] : inactiveMap[color],
        className,
      )}
      aria-label={active ? `${children} loaded` : `${children} unavailable`}
    >
      <span
        className={clsx(
          "w-1 h-1 rounded-full",
          active ? "opacity-100" : "opacity-30",
          color === "eeg" && "bg-signal-eeg",
          color === "meta" && "bg-signal-meta",
          color === "vision" && "bg-signal-vision",
          color === "text" && "bg-signal-text",
          color === "default" && "bg-muted",
        )}
        aria-hidden
      />
      {children}
    </span>
  );
}
