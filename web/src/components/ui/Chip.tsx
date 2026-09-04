import clsx from "clsx";
import type { ReactNode } from "react";

export function Chip({
  children,
  onClick,
  active,
  className,
}: {
  children: ReactNode;
  onClick?: () => void;
  active?: boolean;
  className?: string;
}) {
  const Tag = onClick ? "button" : "span";
  return (
    <Tag
      onClick={onClick}
      className={clsx(
        "inline-flex items-center px-2.5 py-1 text-xs font-medium rounded-md border transition-colors",
        active
          ? "bg-accent/15 border-accent/40 text-accent"
          : "bg-transparent border-default text-secondary",
        onClick && "cursor-pointer hover:border-accent/30 hover:text-primary",
        className,
      )}
    >
      {children}
    </Tag>
  );
}
