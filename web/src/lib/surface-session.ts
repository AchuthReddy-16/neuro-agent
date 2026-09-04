/**
 * Dual-surface session snapshots — Chat and Workspace must not share
 * the active experiment / selected image / answers.
 */

import type { AgentAnswer, Experiment, VisualizationTab } from "@/lib/types";
import {
  emptyAnalysisResults,
  emptyVisionState,
  type AnalysisResultsState,
  type VisionState,
} from "@/lib/analysis-results";

export type SurfaceKind = "chat" | "workspace";

export interface SurfaceSnapshot {
  sessionId: string | null;
  experiment: Experiment | null;
  answers: AgentAnswer[];
  activeAnswerId: string | null;
  analysisResults: AnalysisResultsState;
  visionState: VisionState;
  focusedVizId: string | null;
  activeTab: VisualizationTab;
}

export function emptySurfaceSnapshot(
  activeTab: VisualizationTab = "waveform",
): SurfaceSnapshot {
  return {
    sessionId: null,
    experiment: null,
    answers: [],
    activeAnswerId: null,
    analysisResults: emptyAnalysisResults(),
    visionState: emptyVisionState(),
    focusedVizId: null,
    activeTab,
  };
}

export function captureSurfaceSnapshot(input: {
  sessionId: string | null;
  experiment: Experiment | null;
  answers: AgentAnswer[];
  activeAnswerId: string | null;
  analysisResults: AnalysisResultsState;
  visionState: VisionState;
  focusedVizId: string | null;
  activeTab: VisualizationTab;
}): SurfaceSnapshot {
  return {
    sessionId: input.sessionId,
    experiment: input.experiment
      ? {
          ...input.experiment,
          // Deep-ish copy of mutable collections so later mutations don't leak
          eeg_files: [...input.experiment.eeg_files],
          metadata_files: [...input.experiment.metadata_files],
          image_files: [...input.experiment.image_files],
          visualizations: [...input.experiment.visualizations],
          analysis_history: [...input.experiment.analysis_history],
          modalities: { ...input.experiment.modalities },
          metadata: { ...input.experiment.metadata },
        }
      : null,
    answers: [...input.answers],
    activeAnswerId: input.activeAnswerId,
    analysisResults: { ...input.analysisResults },
    visionState: { ...input.visionState },
    focusedVizId: input.focusedVizId,
    activeTab: input.activeTab,
  };
}

/** Linked-sample load: EEG/metadata + viz tabs only — never keep user figure selection. */
export function stripUserFiguresForLinkedSample(exp: Experiment): Experiment {
  return {
    ...exp,
    image_files: [],
    selected_image_id: null,
    figure: undefined,
    modalities: {
      ...exp.modalities,
      // vision true only if sample provides visualizations (explorer tabs), not user figures
      vision: exp.visualizations.length > 0,
    },
  };
}
