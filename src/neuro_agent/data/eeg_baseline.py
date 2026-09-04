"""EEG movement-state baseline classifier utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuro_agent.paths import PROJECT_ROOT

BASELINE_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "baseline"
SPLIT_REPORT_PATH = PROJECT_ROOT / "data" / "metadata" / "splits" / "split_report.json"

MOVEMENT_LABELS = [
    "rest",
    "left_fist",
    "right_fist",
    "both_fists",
    "both_feet",
]

LABEL_DISPLAY = {
    "rest": "Rest",
    "left_fist": "Left fist",
    "right_fist": "Right fist",
    "both_fists": "Both fists",
    "both_feet": "Both feet",
}

EXPECTED_SPLITS = {
    "train": [f"S{i:03d}" for i in range(1, 21)],
    "validation": [f"S{i:03d}" for i in range(21, 26)],
    "test": [f"S{i:03d}" for i in range(26, 31)],
}


@dataclass
class EegSplit:
    """One data split with features, labels, and metadata."""

    name: str
    X: np.ndarray
    y: np.ndarray
    sample_ids: np.ndarray
    feature_names: np.ndarray
    metadata: pd.DataFrame
    subjects: list[str]


def load_split(name: str) -> EegSplit:
    """Load a prepared baseline split from disk."""
    npz = np.load(BASELINE_DATA_DIR / f"{name}.npz", allow_pickle=True)
    metadata = pd.read_parquet(BASELINE_DATA_DIR / f"{name}_metadata.parquet")
    subjects = sorted(metadata["subject_id"].unique().tolist())
    return EegSplit(
        name=name,
        X=npz["X"].astype(np.float64),
        y=npz["y"].astype(str),
        sample_ids=npz["sample_ids"].astype(str),
        feature_names=npz["feature_names"].astype(str),
        metadata=metadata,
        subjects=subjects,
    )


def verify_subject_splits(splits: dict[str, EegSplit]) -> dict[str, Any]:
    """Fail fast if subjects overlap across splits or deviate from expected."""
    report: dict[str, Any] = {"subject_safe": True, "splits": {}, "errors": []}

    all_subjects: dict[str, str] = {}
    for name, split in splits.items():
        report["splits"][name] = {
            "subjects": split.subjects,
            "n_samples": len(split.y),
            "n_features": split.X.shape[1],
        }
        expected = set(EXPECTED_SPLITS.get(name, []))
        actual = set(split.subjects)
        if expected and actual != expected:
            report["errors"].append(
                f"{name}: subject mismatch expected={sorted(expected)} actual={sorted(actual)}"
            )
        for subject in split.subjects:
            if subject in all_subjects:
                report["errors"].append(
                    f"Subject leakage: {subject} appears in both {all_subjects[subject]} and {name}"
                )
                report["subject_safe"] = False
            all_subjects[subject] = name

    if report["errors"]:
        raise RuntimeError("Subject split validation failed:\n" + "\n".join(report["errors"]))

    return report


def load_all_splits() -> dict[str, EegSplit]:
    """Load train, validation, and test splits."""
    return {
        "train": load_split("train"),
        "validation": load_split("validation"),
        "test": load_split("test"),
    }
