"use client";

import clsx from "clsx";
import { useCallback, useRef, useState, type DragEvent, type ReactNode } from "react";

export function DropZone({
  label,
  hint,
  accept,
  onFiles,
  disabled,
  multiple,
  children,
}: {
  label: string;
  hint: string;
  accept: string;
  onFiles: (files: FileList | File[]) => void;
  disabled?: boolean;
  multiple?: boolean;
  children?: ReactNode;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      if (e.dataTransfer.files?.length) onFiles(e.dataTransfer.files);
    },
    [disabled, onFiles],
  );

  return (
    <div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={clsx(
          "w-full rounded-md border border-dashed px-2.5 py-2 text-left transition-colors duration-150",
          "focus:outline-none focus-visible:ring-1 focus-visible:ring-accent/40",
          disabled && "opacity-50 cursor-not-allowed",
          dragging
            ? "border-accent bg-accent/8"
            : "border-default/80 bg-transparent hover:border-accent/35 hover:bg-muted/25",
        )}
      >
        <p className="text-[11px] font-medium text-primary leading-tight">{label}</p>
        <p className="text-[10px] text-muted mt-0.5 leading-tight">{hint}</p>
        {children}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        aria-label={label}
        disabled={disabled}
        onChange={(e) => {
          if (e.target.files?.length) onFiles(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
