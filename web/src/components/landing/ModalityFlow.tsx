const modalities = [
  { label: "EEG Signals", icon: "∿", color: "text-signal-eeg", border: "border-signal-eeg/25" },
  { label: "Behavioral Metadata", icon: "⊞", color: "text-signal-meta", border: "border-signal-meta/25" },
  { label: "Experimental Images", icon: "◈", color: "text-signal-vision", border: "border-signal-vision/25" },
  { label: "Natural Language", icon: "✦", color: "text-signal-text", border: "border-signal-text/25" },
];

export function ModalityFlow() {
  return (
    <div className="relative">
      <div
        className="hidden md:block absolute top-1/2 left-[12%] right-[12%] h-px bg-gradient-to-r from-transparent via-border-strong to-transparent -translate-y-1/2"
        aria-hidden
      />
      <div className="flex flex-wrap items-center justify-center gap-2 md:gap-0">
        {modalities.map((m, i) => (
          <div key={m.label} className="flex items-center">
            <div
              className={`relative z-10 flex flex-col items-center gap-2 px-5 py-3.5 rounded-xl bg-elevated/70 border ${m.border} backdrop-blur-sm panel-shadow transition-transform duration-200 hover:-translate-y-0.5`}
            >
              <span className={`text-2xl font-mono ${m.color}`} aria-hidden>
                {m.icon}
              </span>
              <span className="text-xs font-medium text-secondary whitespace-nowrap">
                {m.label}
              </span>
            </div>
            {i < modalities.length - 1 && (
              <span
                className="hidden md:flex items-center justify-center w-8 text-accent/50 text-sm font-light"
                aria-hidden
              >
                →
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
