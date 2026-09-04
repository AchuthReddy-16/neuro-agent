#!/usr/bin/env python3
"""Stage F.1 Step 3: Audit multimodal SFT training data coverage and answer formats."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.paths import PROJECT_ROOT

SUBJECT_RE = re.compile(r"^(S\d{3})")
HELD_OUT = {f"S{i:03d}" for i in range(26, 31)}

VERBOSE_PATTERNS = (
    r"is the grounded result",
    r"their values are",
    r"computed across the supplied values",
    r"is higher:",
    r"the computed result is",
    r"the grounded descending order is",
)

ANSWER_ONLY_PATTERNS = {
    "channel_only": re.compile(r"^[A-Z][A-Z0-9]{1,3}$"),
    "label_only": re.compile(r"^[a-z_]+$"),
    "numeric_only": re.compile(r"^[-+]?\d"),
    "ranking_list": re.compile(r"^[A-Z][A-Z0-9]{1,3}(,\s*[A-Z][A-Z0-9]{1,3})+$"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_subjects(example: dict[str, Any]) -> set[str]:
    subjects: set[str] = set()
    for key in ("source_samples", "subjects"):
        for sample in example.get(key, []):
            match = SUBJECT_RE.match(str(sample))
            if match:
                subjects.add(match.group(1))
    subject = example.get("subject")
    if subject:
        match = SUBJECT_RE.match(str(subject))
        if match:
            subjects.add(match.group(1))
    return subjects


def image_family(image_path: str) -> str:
    path = image_path.lower()
    for family in ("waveform", "spectrogram", "psd", "topomap", "band", "comparison", "condition"):
        if family in path:
            if family == "band":
                return "band_power"
            if family in ("comparison", "condition"):
                return "condition_comparison"
            return family
    return "other"


def classify_answer_format(answer: str, task_class: str) -> str:
    text = answer.strip()
    if not text:
        return "empty"
    if any(re.search(p, text.lower()) for p in VERBOSE_PATTERNS):
        return "verbose_grounded_template"
    if ANSWER_ONLY_PATTERNS["channel_only"].match(text):
        return "channel_only"
    if ANSWER_ONLY_PATTERNS["ranking_list"].match(text):
        return "ranking_list"
    if ANSWER_ONLY_PATTERNS["numeric_only"].match(text):
        return "numeric_only"
    if ANSWER_ONLY_PATTERNS["label_only"].match(text):
        return "label_only"
    if task_class == "ranking":
        return "ranking_other"
    if task_class == "numeric":
        return "numeric_other"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit multimodal SFT training data")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/vision/multimodal_sft_train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/multimodal_error_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.train_path)
    train_only = [r for r in rows if not (extract_subjects(r) & HELD_OUT)]

    by_image: Counter[str] = Counter()
    by_task: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    by_answer_fmt: Counter[str] = Counter()
    by_target: Counter[str] = Counter()
    family_answer_fmt: dict[str, Counter[str]] = defaultdict(Counter)

    waveform_categorical = 0
    spectrogram_categorical = 0

    for row in train_only:
        img_fam = image_family(row.get("image_path", ""))
        task_fam = row.get("task_family", "unknown")
        task_class = row.get("task_class", "unknown")
        answer = (row.get("grounded_answer") or row.get("answer") or "").strip()
        gt = row.get("ground_truth", "")

        by_image[img_fam] += 1
        by_task[task_fam] += 1
        by_class[task_class] += 1
        fmt = classify_answer_format(answer, task_class)
        by_answer_fmt[fmt] += 1
        family_answer_fmt[task_fam][fmt] += 1
        by_target[str(gt)] += 1

        if img_fam == "waveform" and task_class == "categorical":
            waveform_categorical += 1
        if img_fam == "spectrogram" and task_class == "categorical":
            spectrogram_categorical += 1

    focus_families = [
        "waveform_highest_rms",
        "waveform_max_rms_numeric",
        "spectrogram_dominant_band",
        "spectrogram_strongest_vs_weakest",
        "topomap_strongest_alpha_mu",
        "band_power_weakest_alpha_mu",
        "psd_dominant_band",
    ]

    focus_coverage = {}
    for fam in focus_families:
        fam_rows = [r for r in train_only if r.get("task_family") == fam]
        samples = [r.get("grounded_answer", "")[:120] for r in fam_rows[:3]]
        focus_coverage[fam] = {
            "count": len(fam_rows),
            "answer_formats": dict(family_answer_fmt.get(fam, {})),
            "sample_answers": samples,
        }

    report = {
        "total_examples": len(rows),
        "train_subjects_only": len(train_only),
        "held_out_leakage": len(rows) - len(train_only),
        "by_image_family": dict(by_image.most_common()),
        "by_task_family": dict(by_task.most_common()),
        "by_task_class": dict(by_class.most_common()),
        "by_answer_format": dict(by_answer_fmt.most_common()),
        "by_ground_truth_top20": dict(by_target.most_common(20)),
        "waveform_categorical_count": waveform_categorical,
        "spectrogram_categorical_count": spectrogram_categorical,
        "verbose_answer_fraction": round(
            by_answer_fmt["verbose_grounded_template"] / len(train_only), 4
        ),
        "answer_only_fraction": round(
            sum(
                by_answer_fmt[k]
                for k in ("channel_only", "label_only", "numeric_only", "ranking_list")
            )
            / len(train_only),
            4,
        ),
        "focus_family_coverage": focus_coverage,
        "findings": [
            "Training targets overwhelmingly use verbose grounded_answer templates.",
            "Waveform categorical answers teach 'CHANNEL is the grounded result (...)' not channel-only labels.",
            "Ranking tasks teach 'The grounded descending order is CH1, CH2, CH3; their values are ...'.",
            "Eval verifiers expect concise parseable answers (channel name, label, number, or ranking).",
        ],
    }

    out_path = args.output_dir / "training_data_audit.json"
    with out_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    meta_dir = PROJECT_ROOT / "data/metadata/vision_corrective"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / "training_data_audit.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
