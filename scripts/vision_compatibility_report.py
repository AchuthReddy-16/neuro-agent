#!/usr/bin/env python3
"""Generate vision multimodal compatibility report (Stage D prep only)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.paths import PROJECT_ROOT

SUBJECT_RE = re.compile(r"^(S\d{3})")

DATASETS = {
    "multimodal_sft_train": PROJECT_ROOT / "data/processed/vision/multimodal_sft_train.jsonl",
    "multimodal_rlvr_train": PROJECT_ROOT / "data/processed/vision/multimodal_rlvr_train.jsonl",
    "multimodal_eval_heldout": PROJECT_ROOT / "data/processed/vision/multimodal_eval_heldout.jsonl",
}

TRAIN_SUBJECTS = {f"S{i:03d}" for i in range(1, 21)}
VAL_SUBJECTS = {f"S{i:03d}" for i in range(21, 26)}
TEST_SUBJECTS = {f"S{i:03d}" for i in range(26, 31)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _schema_keys(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return sorted(rows[0].keys())


def _extract_subjects(row: dict[str, Any]) -> set[str]:
    subjects: set[str] = set()
    for sample in row.get("source_samples", []):
        match = SUBJECT_RE.match(str(sample))
        if match:
            subjects.add(match.group(1))
    context = row.get("context") or row.get("relevant_context") or {}
    if "subject_id" in context:
        subjects.add(str(context["subject_id"]))
    return subjects


def _validate_images(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing: list[str] = []
    resolved: list[str] = []
    for row in rows:
        image_path = row.get("image_path")
        if not image_path:
            missing.append(row.get("id", "unknown"))
            continue
        full_path = PROJECT_ROOT / image_path
        if full_path.exists():
            resolved.append(image_path)
        else:
            missing.append(image_path)
    return {
        "total_with_image_path": len(rows),
        "resolved_count": len(resolved),
        "missing_count": len(missing),
        "missing_samples": missing[:20],
        "all_resolved": len(missing) == 0,
    }


def _subject_leakage(rows: list[dict[str, Any]], allowed: set[str]) -> dict[str, Any]:
    violations: list[dict[str, str]] = []
    subjects = Counter()
    for row in rows:
        row_subjects = _extract_subjects(row)
        subjects.update(row_subjects)
        for subject in row_subjects:
            if subject not in allowed:
                violations.append(
                    {
                        "id": row.get("id", ""),
                        "subject": subject,
                        "reason": "subject_not_allowed_for_split",
                    }
                )
    return {
        "subjects": dict(sorted(subjects.items())),
        "violations": violations[:20],
        "violation_count": len(violations),
        "subject_safe": len(violations) == 0,
    }


def main() -> None:
    output_dir = PROJECT_ROOT / "results/vision_integration"
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "datasets": {},
        "future_multimodal_requirements": {
            "vision_encoder": "SigLIP or CLIP encoder required for image embeddings",
            "projector": "Linear/MLP projector to map vision tokens into LLM hidden size",
            "input_format": "Interleave image tokens with chat template user message",
            "image_fields": ["image_id", "image_path"],
            "context_fields": ["relevant_context", "context", "supporting_numeric_evidence", "source_values"],
            "answer_fields": ["grounded_answer", "ground_truth"],
            "chat_template": "Extend Qwen chat template with vision placeholders",
            "training_stages": ["SFT with image+text pairs", "RLVR with verifier rewards"],
        },
        "blockers": [],
    }

    for name, path in DATASETS.items():
        rows = _load_jsonl(path)
        schema = _schema_keys(rows)
        image_validation = _validate_images(rows)
        if name.endswith("train"):
            allowed = TRAIN_SUBJECTS
        else:
            allowed = TEST_SUBJECTS
        leakage = _subject_leakage(rows, allowed)

        report["datasets"][name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "count": len(rows),
            "schema_keys": schema,
            "image_validation": image_validation,
            "subject_integrity": leakage,
        }

        if not image_validation["all_resolved"]:
            report["blockers"].append(f"{name}: {image_validation['missing_count']} image paths missing")
        if not leakage["subject_safe"]:
            report["blockers"].append(f"{name}: subject leakage detected ({leakage['violation_count']})")

    report["train_test_separation"] = {
        "train_subjects": sorted(TRAIN_SUBJECTS),
        "validation_subjects": sorted(VAL_SUBJECTS),
        "held_out_subjects": sorted(TEST_SUBJECTS),
        "sft_train_uses_train_only": report["datasets"]["multimodal_sft_train"]["subject_integrity"]["subject_safe"],
        "rlvr_train_uses_train_only": report["datasets"]["multimodal_rlvr_train"]["subject_integrity"]["subject_safe"],
        "eval_uses_held_out_only": report["datasets"]["multimodal_eval_heldout"]["subject_integrity"]["subject_safe"],
    }

    output_path = output_dir / "compatibility_report.json"
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
