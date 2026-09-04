#!/usr/bin/env python3
"""Stage F.1 Step 4: Build corrective multimodal SFT dataset with answer-only targets."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.paths import PROJECT_ROOT

SUBJECT_RE = re.compile(r"^(S\d{3})")
HELD_OUT = {f"S{i:03d}" for i in range(26, 31)}
TRAIN_SUBJECTS = {f"S{i:03d}" for i in range(1, 21)}

REGRESSED_FAMILIES = {
    "waveform_highest_rms",
    "waveform_max_rms_numeric",
    "spectrogram_strongest_vs_weakest",
    "spectrogram_dominant_band",
    "topomap_strongest_alpha_mu",
    "band_power_weakest_alpha_mu",
    "psd_dominant_band",
}

REPLAY_FAMILIES = {
    "psd_band_order",
    "psd_peak_frequency",
    "spectrogram_peak_frequency",
    "waveform_rms_order",
    "condition_higher_mean",
    "band_power_beta_top3",
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


def strict_answer(example: dict[str, Any]) -> str:
    gt = example.get("ground_truth")
    task_class = example.get("task_class", "")

    if task_class == "ranking" or example.get("verification_type") == "ranking":
        if isinstance(gt, list):
            return ", ".join(str(x).upper() for x in gt)
        return str(gt).upper()

    if task_class == "set_membership" or example.get("verification_type") == "set":
        if isinstance(gt, list):
            return ", ".join(str(x).upper() for x in gt)
        return str(gt).upper()

    if task_class == "numeric" or example.get("verification_type") == "numeric":
        if isinstance(gt, (int, float)):
            return str(gt)
        return str(gt)

    if isinstance(gt, str):
        return gt.lower().replace(" ", "_")
    return str(gt)


def to_corrective_row(example: dict[str, Any]) -> dict[str, Any]:
    row = dict(example)
    row["grounded_answer"] = strict_answer(example)
    row["answer"] = row["grounded_answer"]
    row["dataset_role"] = "corrective"
    row["answer_format"] = "answer_only"
    return row


def to_replay_row(example: dict[str, Any]) -> dict[str, Any]:
    row = dict(example)
    row["grounded_answer"] = strict_answer(example)
    row["answer"] = row["grounded_answer"]
    row["dataset_role"] = "replay"
    row["answer_format"] = "answer_only"
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimodal corrective SFT dataset")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=PROJECT_ROOT / "data/processed/vision/multimodal_sft_train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/vision/corrective",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corrective-fraction", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = PROJECT_ROOT / "data/metadata/vision_corrective"
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.train_path)
    train_rows = [r for r in rows if extract_subjects(r) <= TRAIN_SUBJECTS and not (extract_subjects(r) & HELD_OUT)]

    corrective_pool: list[dict[str, Any]] = []
    replay_pool: list[dict[str, Any]] = []
    for row in train_rows:
        fam = row.get("task_family", "")
        if fam in REGRESSED_FAMILIES:
            corrective_pool.append(row)
        elif fam in REPLAY_FAMILIES:
            replay_pool.append(row)

    rng = random.Random(args.seed)
    target_total = len(train_rows)
    n_corrective = int(target_total * args.corrective_fraction)
    n_replay = target_total - n_corrective

    if len(replay_pool) < n_replay:
        print(
            f"Warning: replay pool {len(replay_pool)} < {n_replay}; sampling with replacement"
        )
        selected_replay = [rng.choice(replay_pool) for _ in range(n_replay)]
    else:
        selected_replay = rng.sample(replay_pool, n_replay)

    if len(corrective_pool) < n_corrective:
        print(
            f"Warning: corrective pool {len(corrective_pool)} < {n_corrective}; sampling with replacement"
        )
        selected_corrective = [rng.choice(corrective_pool) for _ in range(n_corrective)]
    else:
        selected_corrective = rng.sample(corrective_pool, n_corrective)

    mixed = [to_corrective_row(r) for r in selected_corrective] + [to_replay_row(r) for r in selected_replay]
    rng.shuffle(mixed)

    out_path = args.output_dir / "multimodal_sft_corrective_train.jsonl"
    with out_path.open("w") as handle:
        for row in mixed:
            handle.write(json.dumps(row) + "\n")

    by_role = Counter(r["dataset_role"] for r in mixed)
    by_family = Counter(r.get("task_family", "?") for r in mixed)
    by_image: Counter[str] = Counter()
    for row in mixed:
        p = row.get("image_path", "").lower()
        if "waveform" in p:
            by_image["waveform"] += 1
        elif "spectrogram" in p:
            by_image["spectrogram"] += 1
        elif "topomap" in p:
            by_image["topomap"] += 1
        elif "psd" in p:
            by_image["psd"] += 1
        else:
            by_image["other"] += 1

    report = {
        "total_examples": len(mixed),
        "corrective_fraction": round(by_role["corrective"] / len(mixed), 4),
        "replay_fraction": round(by_role["replay"] / len(mixed), 4),
        "by_dataset_role": dict(by_role),
        "by_task_family_top20": dict(by_family.most_common(20)),
        "by_image_family": dict(by_image),
        "corrective_families": sorted(REGRESSED_FAMILIES),
        "replay_families": sorted(REPLAY_FAMILIES),
        "output_path": str(out_path),
        "sample_corrective_answers": [
            r["grounded_answer"] for r in mixed if r["dataset_role"] == "corrective"
        ][:10],
    }

    with (meta_dir / "corrective_dataset_report.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    with (args.output_dir / "dataset_manifest.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
