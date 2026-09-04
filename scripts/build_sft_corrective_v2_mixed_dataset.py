#!/usr/bin/env python3
"""Build mixed corrective SFT v2 dataset with replay buffer."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.paths import PROJECT_ROOT

SFT_QUESTION_FAMILIES = {
    "movement_classification": "Which normalized movement condition is assigned to this sample?",
    "band_power_ranking": "Which EEG channel has the highest beta-band power in this epoch?",
    "numeric_rms": "What is the RMS amplitude for channel F3?",
    "threshold_set": "Which channels have beta power strictly above the supplied median threshold?",
}

REPLAY_FAMILY_TO_TASK = {
    "movement_classification": "movement_task_classification",
    "band_power_ranking": "band_power_analysis",
    "numeric_rms": "numerical_reasoning",
    "threshold_set": "channel_ranking",
}

SUBJECT_RE = re.compile(r"^(S\d{3})")
HELD_OUT_SUBJECTS = {f"S{i:03d}" for i in range(26, 31)}
VALID_LABELS = {"baseline", "execution", "imagery"}


def extract_subjects(example: dict) -> set[str]:
    subjects: set[str] = set()
    for sample in example.get("source_samples", []):
        match = SUBJECT_RE.match(str(sample))
        if match:
            subjects.add(match.group(1))
    return subjects


def classify_sft_family(question: str) -> str:
    for family, text in SFT_QUESTION_FAMILIES.items():
        if question == text:
            return family
    return "other"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_corrective_answer(example: dict) -> dict:
    label = str(example.get("ground_truth", "")).strip().lower()
    if label not in VALID_LABELS:
        first = example["answer"].split(".")[0].strip().lower()
        if first in VALID_LABELS:
            label = first
        else:
            raise ValueError(f"Cannot normalize corrective label for {example['id']}: {example['answer']}")
    out = dict(example)
    out["answer"] = label
    out["category"] = "execution_vs_imagery"
    out["dataset_role"] = "corrective"
    return out


def sample_replay(
    sft_examples: list[dict],
    *,
    seed: int,
    per_family: int,
) -> list[dict]:
    by_family: dict[str, list[dict]] = {family: [] for family in SFT_QUESTION_FAMILIES}
    for example in sft_examples:
        subjects = extract_subjects(example)
        if subjects & HELD_OUT_SUBJECTS:
            continue
        family = classify_sft_family(example["question"])
        if family in by_family:
            by_family[family].append(example)

    rng = random.Random(seed)
    selected: list[dict] = []
    for family, pool in by_family.items():
        if len(pool) < per_family:
            raise ValueError(
                f"Insufficient replay pool for {family}: need {per_family}, have {len(pool)}"
            )
        chosen = rng.sample(pool, per_family)
        for row in chosen:
            out = dict(row)
            out["category"] = REPLAY_FAMILY_TO_TASK[family]
            out["dataset_role"] = "replay"
            out["replay_family"] = family
            selected.append(out)
    return selected


def build_mixed_dataset(
    corrective_path: Path,
    sft_train_path: Path,
    *,
    seed: int,
    replay_total: int,
) -> tuple[list[dict], dict]:
    corrective = [normalize_corrective_answer(row) for row in load_jsonl(corrective_path)]
    if len(corrective) != 200:
        raise ValueError(f"Expected 200 corrective examples, got {len(corrective)}")

    per_family = replay_total // len(SFT_QUESTION_FAMILIES)
    if per_family * len(SFT_QUESTION_FAMILIES) != replay_total:
        raise ValueError("replay_total must be divisible by number of replay families")

    replay = sample_replay(load_jsonl(sft_train_path), seed=seed, per_family=per_family)

    rng = random.Random(seed)
    mixed = corrective + replay
    rng.shuffle(mixed)

    task_counts = Counter(row.get("category", "unknown") for row in mixed)
    role_counts = Counter(row.get("dataset_role", "unknown") for row in mixed)
    replay_family_counts = Counter(row.get("replay_family", "corrective") for row in mixed)

    report = {
        "seed": seed,
        "total_examples": len(mixed),
        "corrective_examples": len(corrective),
        "replay_examples": len(replay),
        "replay_per_family": per_family,
        "task_counts": dict(sorted(task_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "replay_family_counts": dict(sorted(replay_family_counts.items())),
        "held_out_subjects_excluded": sorted(HELD_OUT_SUBJECTS),
        "corrective_answer_format": "strict label only: baseline | execution | imagery",
    }
    return mixed, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corrective SFT v2 mixed dataset")
    parser.add_argument(
        "--corrective-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sft_corrective_execution_imagery.jsonl",
    )
    parser.add_argument(
        "--sft-train-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sft_train.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/sft_corrective_v2_mixed.jsonl",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=PROJECT_ROOT / "data/metadata/corrective/sft_corrective_v2_mixed_report.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-total", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mixed, report = build_mixed_dataset(
        args.corrective_path,
        args.sft_train_path,
        seed=args.seed,
        replay_total=args.replay_total,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    with args.output_path.open("w") as handle:
        for row in mixed:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with args.report_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
