#!/usr/bin/env python3
"""Train and evaluate classical EEG movement-state baselines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.evaluation.eeg_classifier import run_baseline_classifier
from neuro_agent.paths import RESULTS_DIR


def main() -> None:
    summary = run_baseline_classifier(RESULTS_DIR / "baseline_classifier")
    print(json.dumps(summary.__dict__, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_DIR / 'baseline_classifier'}")


if __name__ == "__main__":
    main()
