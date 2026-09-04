import clsx from "clsx";
import type { CSSProperties } from "react";

export function Skeleton({
  className,
  style,
}: {
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      className={clsx(
        "rounded-md bg-muted/80 animate-pulse border border-default/40",
        className,
      )}
      style={style}
      aria-hidden
    />
  );
}

export function ExplorerSkeleton() {
  return (
    <div className="flex flex-col h-full gap-3 animate-fade-in" aria-busy aria-label="Loading visualization">
      <div className="flex gap-2">
        <Skeleton className="h-8 flex-1" />
        <Skeleton className="h-8 flex-1" />
        <Skeleton className="h-8 flex-1" />
        <Skeleton className="h-8 w-24 hidden sm:block" />
      </div>
      <Skeleton className="flex-1 min-h-[280px] rounded-xl" />
      <div className="flex gap-2">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-4 w-40" />
      </div>
    </div>
  );
}

export function AgentSkeleton() {
  return (
    <div className="space-y-3 animate-fade-in" aria-busy>
      <Skeleton className="h-4 w-24" />
      <Skeleton className="h-20 w-full" />
      <Skeleton className="h-4 w-32" />
      <Skeleton className="h-28 w-full" />
      <div className="space-y-2 pt-2">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="flex gap-3 items-center">
            <Skeleton className="w-5 h-5 rounded-full shrink-0" />
            <Skeleton className="h-3 flex-1" />
          </div>
        ))}
      </div>
    </div>
  );
}
