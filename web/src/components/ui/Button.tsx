import clsx from "clsx";
import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "outline";
  size?: "sm" | "md" | "lg";
  icon?: ReactNode;
}

export function Button({
  className,
  variant = "primary",
  size = "md",
  children,
  icon,
  ...props
}: ButtonProps) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center gap-2 font-medium rounded-lg",
        "transition-all duration-200",
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 focus-visible:ring-offset-1 focus-visible:ring-offset-surface",
        "disabled:opacity-40 disabled:cursor-not-allowed",
        {
          "bg-accent text-slate-900 hover:brightness-110 active:brightness-95 shadow-sm shadow-accent/15":
            variant === "primary",
          "bg-muted text-primary hover:bg-surface-elevated border border-default hover:border-strong":
            variant === "secondary",
          "text-secondary hover:text-primary hover:bg-muted": variant === "ghost",
          "border border-default text-secondary hover:text-primary hover:border-accent/40 hover:bg-muted/50":
            variant === "outline",
          "text-xs px-2.5 py-1.5": size === "sm",
          "text-sm px-4 py-2": size === "md",
          "text-base px-6 py-3": size === "lg",
        },
        className,
      )}
      {...props}
    >
      {icon}
      {children}
    </button>
  );
}
