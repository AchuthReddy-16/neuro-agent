"use client";

import { useExperiment } from "@/lib/store/experiment-context";
import { Collapsible } from "@/components/ui/Collapsible";
import { Toggle } from "@/components/ui/Toggle";
import { Button } from "@/components/ui/Button";
import type { AppSettings, FrequencyBand } from "@/lib/types";

const BANDS: { value: FrequencyBand; label: string }[] = [
  { value: "delta", label: "Delta" },
  { value: "theta", label: "Theta" },
  { value: "alpha_mu", label: "Alpha/Mu" },
  { value: "beta", label: "Beta" },
];

function OptionGroup<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  columns?: number;
}) {
  return (
    <fieldset>
      <legend className="section-label mb-2">{label}</legend>
      <div className={`grid gap-1.5 ${options.length > 2 ? "grid-cols-2" : "grid-cols-2"}`}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            aria-pressed={value === o.value}
            className={`px-2.5 py-1.5 text-xs rounded-md border transition-colors ${
              value === o.value
                ? "bg-accent/12 border-accent/40 text-accent"
                : "border-default text-muted hover:text-secondary hover:border-strong"
            }`}
          >
            {o.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function SettingsPanel() {
  const { settings, updateSettings, clearExperiment } = useExperiment();

  return (
    <Collapsible title="Settings">
      <div className="space-y-5">
        <section className="space-y-3">
          <h3 className="text-[10px] font-medium text-accent-secondary uppercase tracking-wide">
            Appearance
          </h3>
          <OptionGroup
            label="Theme"
            options={[
              { value: "dark" as const, label: "Dark" },
              { value: "light" as const, label: "Light" },
            ]}
            value={settings.theme}
            onChange={(t) => updateSettings({ theme: t })}
          />
        </section>

        <section className="space-y-3 pt-2 border-t border-default">
          <h3 className="text-[10px] font-medium text-accent-secondary uppercase tracking-wide">
            Analysis
          </h3>
          <OptionGroup
            label="Default Frequency Band"
            options={BANDS}
            value={settings.defaultBand}
            onChange={(b) => updateSettings({ defaultBand: b })}
          />
          <OptionGroup
            label="Answer Detail"
            options={[
              { value: "concise" as const, label: "Concise" },
              { value: "detailed" as const, label: "Detailed" },
            ]}
            value={settings.answerDetail}
            onChange={(d) => updateSettings({ answerDetail: d })}
          />
        </section>

        <section className="space-y-3 pt-2 border-t border-default">
          <h3 className="text-[10px] font-medium text-accent-secondary uppercase tracking-wide">
            Output
          </h3>
          <div className="space-y-3">
            <ToggleRow
              settings={settings}
              field="showRawToolOutput"
              label="Show Raw Tool Output"
              updateSettings={updateSettings}
            />
            <ToggleRow
              settings={settings}
              field="showUncertainty"
              label="Show Uncertainty"
              updateSettings={updateSettings}
            />
            <ToggleRow
              settings={settings}
              field="showSystemMetrics"
              label="Show System Metrics"
              updateSettings={updateSettings}
            />
            <ToggleRow
              settings={settings}
              field="autoGenerateVisuals"
              label="Auto-generate Visuals"
              updateSettings={updateSettings}
            />
          </div>
        </section>

        <section className="pt-2 border-t border-default">
          <h3 className="text-[10px] font-medium text-accent-secondary uppercase tracking-wide mb-3">
            Data
          </h3>
          <Button variant="outline" size="sm" className="w-full" onClick={clearExperiment}>
            Clear Uploaded Data
          </Button>
        </section>
      </div>
    </Collapsible>
  );
}

function ToggleRow({
  settings,
  field,
  label,
  updateSettings,
}: {
  settings: AppSettings;
  field: keyof Pick<
    AppSettings,
    "showRawToolOutput" | "showUncertainty" | "showSystemMetrics" | "autoGenerateVisuals"
  >;
  label: string;
  updateSettings: (p: Partial<AppSettings>) => void;
}) {
  return (
    <Toggle
      label={label}
      checked={settings[field]}
      onChange={(v) => updateSettings({ [field]: v })}
    />
  );
}
