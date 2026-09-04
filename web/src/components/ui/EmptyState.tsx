import type { ReactNode } from "react";
import { Button } from "./Button";

export function EmptyState({
  icon,
  title,
  description,
  actions,
}: {
  icon?: ReactNode;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      {icon && (
        <div
          className="w-10 h-10 rounded-lg bg-muted/60 border border-default flex items-center justify-center mb-3 text-accent"
          aria-hidden
        >
          {icon}
        </div>
      )}
      {!icon && (
        <span className="text-accent/50 font-mono text-lg mb-3" aria-hidden>
          ∿
        </span>
      )}
      <h3 className="text-sm font-medium text-primary mb-1.5">{title}</h3>
      {description && (
        <p className="text-muted text-[13px] max-w-xs leading-relaxed mb-5">
          {description}
        </p>
      )}
      {actions && <div className="flex flex-wrap items-center justify-center gap-2">{actions}</div>}
    </div>
  );
}

export function EmptyStateButton({
  children,
  onClick,
  variant = "outline",
}: {
  children: ReactNode;
  onClick: () => void;
  variant?: "primary" | "outline" | "secondary";
}) {
  return (
    <Button size="sm" variant={variant} onClick={onClick}>
      {children}
    </Button>
  );
}
