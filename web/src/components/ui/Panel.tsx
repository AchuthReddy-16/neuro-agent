import clsx from "clsx";
import type { ReactNode } from "react";

export function Panel({
  title,
  children,
  className,
  headerRight,
  noPadding,
  variant = "default",
}: {
  title?: string;
  children: ReactNode;
  className?: string;
  headerRight?: ReactNode;
  noPadding?: boolean;
  variant?: "default" | "canvas" | "flat";
}) {
  return (
    <section
      className={clsx(
        "flex flex-col overflow-hidden",
        variant === "flat"
          ? "bg-transparent border-0"
          : variant === "canvas"
            ? "bg-canvas border border-strong rounded-xl panel-shadow"
            : "bg-elevated border border-default rounded-xl panel-shadow",
        className,
      )}
    >
      {title && (
        <header className="flex items-center justify-between px-4 py-3 border-b border-default shrink-0 bg-elevated/50">
          <h2 className="text-xs font-semibold tracking-wider text-secondary uppercase">
            {title}
          </h2>
          {headerRight}
        </header>
      )}
      <div className={clsx("flex-1 min-h-0", !noPadding && "p-4")}>{children}</div>
    </section>
  );
}
