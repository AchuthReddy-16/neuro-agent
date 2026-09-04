#!/usr/bin/env python3
"""Audit multimodal RLVR training data (Stage F.2 Step 1)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.multimodal.dataset import load_multimodal_examples
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT


def _subject(example: dict) -> str:
    for field in ("image_path", "id", "image_id", "source_samples"):
        value = example.get(field, "")
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        match = re.search(r"(S\d{3})", str(value))
        if match:
            return match.group(1)
    return "unknown"


def _image_family(example: dict) -> str:
    path = str(example.get("image_path", "")).lower()
    for family in ("waveform", "spectrogram", "topomap", "psd", "band_power", "comparison", "channel_band"):
        if family in path:
            return family
    return "unknown"


def _target_format(example: dict) -> str:
    return example.get("task_class", example.get("verification_type", "unknown"))


def build_audit_report(examples: list[dict], forbidden: set[str]) -> dict:
    subjects = Counter(_subject(ex) for ex in examples)
    leakage = {s: c for s, c in subjects.items() if s in forbidden}

    return {
        "total_examples": len(examples),
        "image_family": dict(Counter(_image_family(ex) for ex in examples)),
        "task_family": dict(Counter(ex.get("task_family", "unknown") for ex in examples)),
        "verification_type": dict(Counter(ex.get("verification_type", "unknown") for ex in examples)),
        "target_format": dict(Counter(_target_format(ex) for ex in examples)),
        "subject_split": dict(sorted(subjects.items())),
        "held_out_leakage": leakage,
        "subject_safe": len(leakage) == 0,
        "forbidden_subjects": sorted(forbidden),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit multimodal RLVR training data")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "multimodal_rlvr.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/multimodal_rlvr_training/data_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    path = PROJECT_ROOT / data_cfg["train_path"]
    forbidden = set(data_cfg.get("forbidden_subjects", []))

    examples = load_multimodal_examples(path)
    report = build_audit_report(examples, forbidden)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    if not report["subject_safe"]:
        print("ERROR: held-out subject leakage detected", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
