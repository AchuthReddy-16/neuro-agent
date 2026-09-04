/**
 * Client storage versioning — clear obsolete demo/shared artifact keys
 * without wiping unrelated preferences (theme, etc.).
 */

export const NEURO_STORE_VERSION = 3;
export const NEURO_STORE_VERSION_KEY = "neuro-agent.store.version";

/** Obsolete keys from earlier demo/shared-session shapes. */
export const OBSOLETE_STORAGE_KEYS = [
  "neuro-agent.demo",
  "neuro-agent.demoExperiment",
  "neuro-agent.experiment",
  "neuro-agent.selectedImage",
  "neuro-agent.selectedFigure",
  "neuro-agent.currentExperiment",
  "neuro-agent.answers",
  "neuro-agent.analysisResults",
  "neuro-agent.visionState",
  "neuro-agent.sharedExperiment",
  "neuro-agent.exp_demo_s001",
  "neuro-agent.experienceMode",
] as const;

function removeKey(storage: Storage, key: string) {
  try {
    storage.removeItem(key);
  } catch {
    /* private mode / quota */
  }
}

function sweepPrefixed(storage: Storage, prefixes: string[]) {
  try {
    const toRemove: string[] = [];
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (!key) continue;
      if (prefixes.some((p) => key === p || key.startsWith(`${p}.`) || key.startsWith(`${p}:`))) {
        toRemove.push(key);
      }
    }
    toRemove.forEach((k) => removeKey(storage, k));
  } catch {
    /* ignore */
  }
}

/**
 * Run once on app boot. Invalidates incompatible persisted session/demo state.
 * Preserves keys that look like settings/preferences (theme, etc.).
 */
export function migrateClientStorage(): void {
  if (typeof window === "undefined") return;
  try {
    const raw = window.localStorage.getItem(NEURO_STORE_VERSION_KEY);
    const version = raw ? Number(raw) : 0;
    if (version >= NEURO_STORE_VERSION) return;

    for (const key of OBSOLETE_STORAGE_KEYS) {
      removeKey(window.localStorage, key);
      removeKey(window.sessionStorage, key);
    }

    // Sweep older experimental prefixes (not theme/settings)
    sweepPrefixed(window.localStorage, [
      "neuro-agent.demo",
      "neuro-agent.experiment",
      "neuro-agent.session",
      "neuro-agent.chat",
      "neuro-agent.workspace.experiment",
    ]);
    sweepPrefixed(window.sessionStorage, [
      "neuro-agent.demo",
      "neuro-agent.experiment",
      "neuro-agent.session",
      "neuro-agent.chat",
      "neuro-agent.workspace.experiment",
    ]);

    window.localStorage.setItem(NEURO_STORE_VERSION_KEY, String(NEURO_STORE_VERSION));
  } catch {
    /* ignore */
  }
}
