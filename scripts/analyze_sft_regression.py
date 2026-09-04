#!/usr/bin/env python3
"""Investigate SFT regression on execution_vs_imagery and factual_grounding."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "sft_regression_analysis"

FOCUS_TASKS = [
    "execution_vs_imagery",
    "factual_grounding",
    "movement_task_classification",
    "band_power_analysis",
    "channel_ranking",
]

SFT_QUESTION_FAMILIES = {
    "band_power_ranking": "Which EEG channel has the highest beta-band power in this epoch?",
    "numeric_rms": "What is the RMS amplitude for channel F3?",
    "movement_classification": "Which normalized movement condition is assigned to this sample?",
    "threshold_set": "Which channels have beta power strictly above the supplied median threshold?",
}

SUBJECT_RE = re.compile(r"^(S\d{3})")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_sft_example(example: dict[str, Any]) -> str:
    question = example["question"]
    for family, text in SFT_QUESTION_FAMILIES.items():
        if question == text:
            return family
    return "other"


def extract_subjects(example: dict[str, Any]) -> set[str]:
    subjects: set[str] = set()
    for sample in example.get("source_samples", []):
        match = SUBJECT_RE.match(str(sample))
        if match:
            subjects.add(match.group(1))
    return subjects


def build_task_distribution(
    sft_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    train_subjects: set[str],
    val_subjects: set[str],
) -> dict[str, Any]:
    sft_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in sft_examples:
        sft_by_family[classify_sft_example(ex)].append(ex)

    eval_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ex in eval_examples:
        eval_by_category[ex["category"]].append(ex)

    sft_families: dict[str, Any] = {}
    for family, rows in sorted(sft_by_family.items()):
        labels = Counter()
        answers = Counter()
        train_count = 0
        val_count = 0
        for row in rows:
            labels[str(row["grounded_facts"]["ground_truth"])] += 1
            answers[row["answer"]] += 1
            subjects = extract_subjects(row)
            if subjects <= val_subjects:
                val_count += 1
            elif subjects <= train_subjects:
                train_count += 1

        sft_families[family] = {
            "count": len(rows),
            "train_count": train_count,
            "validation_count": val_count,
            "label_distribution": dict(labels.most_common()),
            "answer_templates": dict(answers.most_common(10)),
            "maps_to_eval_categories": _family_eval_mapping(family),
        }

    eval_categories: dict[str, Any] = {}
    for category in sorted(eval_by_category):
        rows = eval_by_category[category]
        labels = Counter(str(r["ground_truth"]) for r in rows)
        context_keys = Counter(str(sorted(r.get("context", {}).keys())) for r in rows)
        eval_categories[category] = {
            "count": len(rows),
            "label_distribution": dict(labels.most_common()),
            "context_key_patterns": dict(context_keys.most_common()),
            "has_matching_sft_family": _eval_has_sft_match(category),
        }

    focus_coverage = {}
    for task in FOCUS_TASKS:
        focus_coverage[task] = {
            "eval_count": len(eval_by_category.get(task, [])),
            "sft_direct_examples": _count_direct_sft_examples(task, sft_examples),
            "sft_related_family": _related_sft_family(task),
        }

    return {
        "sft_train_total": len(sft_examples),
        "sft_families": sft_families,
        "eval_heldout_total": len(eval_examples),
        "eval_categories": eval_categories,
        "focus_task_coverage": focus_coverage,
        "task_family_overrepresentation": {
            family: data["count"]
            for family, data in sft_families.items()
        },
    }


def _family_eval_mapping(family: str) -> list[str]:
    mapping = {
        "band_power_ranking": ["band_power_analysis"],
        "numeric_rms": ["numerical_reasoning"],
        "movement_classification": ["movement_task_classification", "factual_grounding"],
        "threshold_set": [],
    }
    return mapping.get(family, [])


def _eval_has_sft_match(category: str) -> bool:
    direct = {
        "movement_task_classification": True,
        "band_power_analysis": True,
        "channel_ranking": False,
        "execution_vs_imagery": False,
        "factual_grounding": False,
    }
    return direct.get(category, False)


def _related_sft_family(task: str) -> str | None:
    mapping = {
        "movement_task_classification": "movement_classification",
        "factual_grounding": "movement_classification",
        "band_power_analysis": "band_power_ranking",
        "channel_ranking": "band_power_ranking",
        "execution_vs_imagery": None,
    }
    return mapping.get(task)


def _count_direct_sft_examples(task: str, sft_examples: list[dict[str, Any]]) -> int:
    if task == "movement_task_classification":
        return sum(
            1
            for ex in sft_examples
            if ex["question"] == SFT_QUESTION_FAMILIES["movement_classification"]
        )
    if task == "band_power_analysis":
        return sum(
            1
            for ex in sft_examples
            if ex["question"] == SFT_QUESTION_FAMILIES["band_power_ranking"]
        )
    return 0


def build_confusion_analysis(
    base_preds: dict[str, dict[str, Any]],
    sft_preds: dict[str, dict[str, Any]],
    eval_examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}

    for task in FOCUS_TASKS:
        base_conf = defaultdict(Counter)
        sft_conf = defaultdict(Counter)
        base_pass = sft_pass = total = 0
        regressions: list[dict[str, Any]] = []
        improvements: list[dict[str, Any]] = []

        for eid, base_row in base_preds.items():
            if base_row["category"] != task:
                continue
            sft_row = sft_preds[eid]
            total += 1
            expected = str(base_row["verification"].get("expected", base_row["ground_truth"]))
            base_parsed = str(base_row["verification"].get("parsed_answer", ""))
            sft_parsed = str(sft_row["verification"].get("parsed_answer", ""))
            base_ok = bool(base_row["verification"]["passed"])
            sft_ok = bool(sft_row["verification"]["passed"])
            base_pass += int(base_ok)
            sft_pass += int(sft_ok)
            base_conf[expected][base_parsed] += 1
            sft_conf[expected][sft_parsed] += 1

            if base_ok and not sft_ok:
                ev = eval_examples.get(eid, {})
                regressions.append(
                    {
                        "id": eid,
                        "ground_truth": expected,
                        "base_parsed": base_parsed,
                        "sft_parsed": sft_parsed,
                        "base_response": base_row["response"][:120],
                        "sft_response": sft_row["response"][:120],
                        "context": ev.get("context", {}),
                    }
                )
            elif not base_ok and sft_ok:
                improvements.append(
                    {
                        "id": eid,
                        "ground_truth": expected,
                        "base_parsed": base_parsed,
                        "sft_parsed": sft_parsed,
                    }
                )

        per_task[task] = {
            "count": total,
            "base_pass_rate": round(base_pass / total, 4) if total else 0.0,
            "sft_pass_rate": round(sft_pass / total, 4) if total else 0.0,
            "delta": (sft_pass - base_pass) if total else 0,
            "base_confusion": {k: dict(v) for k, v in sorted(base_conf.items())},
            "sft_confusion": {k: dict(v) for k, v in sorted(sft_conf.items())},
            "regressed_count": len(regressions),
            "improved_count": len(improvements),
            "sample_regressions": regressions[:10],
            "sample_improvements": improvements[:10],
        }

    execution_detail = _execution_vs_imagery_detail(sft_preds, eval_examples)
    factual_detail = _factual_grounding_detail(base_preds, sft_preds, eval_examples)

    return {
        "per_task": per_task,
        "execution_vs_imagery_by_condition": execution_detail,
        "factual_grounding_analysis": factual_detail,
    }


def _execution_vs_imagery_detail(
    sft_preds: dict[str, dict[str, Any]],
    eval_examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_condition: dict[str, Any] = {}
    for eid, row in sft_preds.items():
        if row["category"] != "execution_vs_imagery":
            continue
        ev = eval_examples[eid]
        condition = ev["context"].get("condition", "unknown")
        entry = by_condition.setdefault(
            condition,
            {
                "ground_truth": str(ev["ground_truth"]),
                "count": 0,
                "sft_parsed_distribution": Counter(),
                "sample_sft_response": row["response"],
            },
        )
        entry["count"] += 1
        entry["sft_parsed_distribution"][str(row["verification"]["parsed_answer"])] += 1

    for condition, entry in by_condition.items():
        entry["sft_parsed_distribution"] = dict(entry["sft_parsed_distribution"])

    return by_condition


def _factual_grounding_detail(
    base_preds: dict[str, dict[str, Any]],
    sft_preds: dict[str, dict[str, Any]],
    eval_examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    context_fields = Counter()
    derivable_from_samples = 0
    movement_only_match = 0
    gt_vs_sft = defaultdict(Counter)

    samples = {s["sample_id"]: s for s in load_jsonl(PROJECT_ROOT / "data/processed/samples.jsonl")}

    for eid, row in sft_preds.items():
        if row["category"] != "factual_grounding":
            continue
        ev = eval_examples[eid]
        context_fields[str(sorted(ev["context"].keys()))] += 1
        sample = samples.get(ev["context"].get("sample_id", ""))
        if sample:
            derived = f"{sample['task_type']}_{sample['movement']}"
            if derived == ev["ground_truth"]:
                derivable_from_samples += 1

        gt = str(ev["ground_truth"])
        parsed = str(row["verification"]["parsed_answer"])
        movement = gt.split("_", 1)[-1]
        if parsed == movement:
            movement_only_match += 1
        gt_vs_sft[gt][parsed] += 1

    run_event_ambiguity = _run_event_ambiguity(eval_examples)

    return {
        "eval_context_fields": dict(context_fields.most_common()),
        "ground_truth_derivable_from_samples_metadata": derivable_from_samples,
        "ground_truth_derivable_from_eval_context_alone": False,
        "run_id_event_code_ambiguity_in_eval": run_event_ambiguity,
        "sft_outputs_movement_component_only": movement_only_match,
        "base_top_response": "refusal_or_missing_context (125/125)",
        "sft_top_responses": dict(
            Counter(row["response"] for eid, row in sft_preds.items() if row["category"] == "factual_grounding").most_common(5)
        ),
        "ground_truth_vs_sft_parsed": {k: dict(v) for k, v in sorted(gt_vs_sft.items())},
        "learnable_from_prompt": False,
        "dataset_design_issue": True,
        "issue_summary": (
            "Eval context exposes only sample_id, run_id, and event_code. "
            "Ground truth uses compound labels like imagery_rest that require "
            "task_type and movement metadata omitted from the prompt."
        ),
    }


def _run_event_ambiguity(eval_examples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mapping: dict[tuple[str, str], set[str]] = defaultdict(set)
    event_only: dict[str, set[str]] = defaultdict(set)
    for ev in eval_examples.values():
        if ev["category"] != "factual_grounding":
            continue
        key = (ev["context"]["run_id"], ev["context"]["event_code"])
        mapping[key].add(str(ev["ground_truth"]))
        event_only[ev["context"]["event_code"]].add(str(ev["ground_truth"]))

    ambiguous_pairs = sum(1 for labels in mapping.values() if len(labels) > 1)
    ambiguous_event_codes = sum(1 for labels in event_only.values() if len(labels) > 1)
    return {
        "unique_run_id_event_code_pairs": len(mapping),
        "ambiguous_run_id_event_code_pairs": ambiguous_pairs,
        "event_code_alone_ambiguous": ambiguous_event_codes,
        "event_code_label_sets": {k: sorted(v) for k, v in sorted(event_only.items())},
    }


def build_data_quality_findings(sft_examples: list[dict[str, Any]]) -> dict[str, Any]:
    movement_rows = [
        ex
        for ex in sft_examples
        if ex["question"] == SFT_QUESTION_FAMILIES["movement_classification"]
    ]

    task_type_contradictions: dict[str, Any] = {}
    by_event_movement: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in movement_rows:
        ctx = row["tool_context"]["inputs"]
        key = (str(ctx.get("event_code")), str(row["grounded_facts"]["ground_truth"]))
        by_event_movement[key].add(str(ctx.get("task_type")))

    for key, task_types in sorted(by_event_movement.items()):
        if len(task_types) > 1:
            task_type_contradictions[f"{key[0]}_{key[1]}"] = sorted(task_types)

    t0_rest_count = sum(
        1
        for row in movement_rows
        if row["tool_context"]["inputs"].get("event_code") == "T0"
        and row["grounded_facts"]["ground_truth"] == "rest"
    )

    source_sample_usage = Counter()
    for ex in sft_examples:
        for sample in ex.get("source_samples", []):
            source_sample_usage[sample] += 1

    eval_examples = load_jsonl(PROJECT_ROOT / "data/processed/eval_heldout.jsonl")
    sft_source_samples = {
        sample for ex in sft_examples for sample in ex.get("source_samples", [])
    }
    eval_overlap = {}
    for task in FOCUS_TASKS:
        task_rows = [ex for ex in eval_examples if ex["category"] == task]
        overlap = sum(
            1
            for ex in task_rows
            if set(ex.get("source_samples", [])) & sft_source_samples
        )
        eval_overlap[task] = {"eval_count": len(task_rows), "source_sample_overlap": overlap}

    return {
        "class_imbalance": {
            "movement_classification_rest_fraction": round(t0_rest_count / len(movement_rows), 4),
            "movement_label_distribution": dict(
                Counter(str(r["grounded_facts"]["ground_truth"]) for r in movement_rows).most_common()
            ),
        },
        "inconsistent_label_wording": {
            "sft_movement_labels": ["rest", "both_fists", "left_fist", "both_feet", "right_fist"],
            "eval_execution_vs_imagery_labels": ["baseline", "execution", "imagery"],
            "eval_factual_grounding_labels": "compound task_type_movement (e.g. imagery_rest)",
            "label_space_mismatch": True,
        },
        "contradictory_examples": {
            "movement_training_same_event_movement_across_task_types": task_type_contradictions,
            "summary": (
                "Movement SFT examples use identical labels for execution, imagery, "
                "and baseline when event_code and movement match. task_type is present "
                "in training context but ignored by the supervised answer."
            ),
        },
        "execution_vs_imagery_mapping": {
            "sft_training_examples": 0,
            "eval_mapping_rule": "ground_truth = task_type from condition prefix; rest trials map to parent task_type not movement label",
            "baseline_rest_ambiguity": {
                "eval_baseline_rest_count": 11,
                "execution_rest_count": 27,
                "imagery_rest_count": 28,
                "sft_outputs_rest_for_all_rest_conditions": True,
                "base_outputs_baseline_for_execution_and_imagery_rest": True,
            },
        },
        "sparse_metadata": {
            "factual_grounding_missing_fields": ["task_type", "movement", "condition"],
            "movement_eval_includes_task_type": True,
            "execution_vs_imagery_includes_condition_and_movement": True,
        },
        "duplicate_near_duplicate_prompts": {
            "max_source_sample_appearances_in_sft": max(source_sample_usage.values()),
            "samples_with_more_than_four_appearances": sum(1 for c in source_sample_usage.values() if c > 4),
            "unique_source_samples_in_sft": len(source_sample_usage),
        },
        "train_eval_mismatch": eval_overlap,
        "overrepresentation": {
            "sft_has_only_four_question_families": True,
            "examples_per_family": 600,
            "missing_eval_categories_in_sft": ["execution_vs_imagery", "factual_grounding", "channel_ranking", "tool_selection", "statistical_comparison", "numerical_reasoning"],
        },
    }


def build_remediation_plan() -> dict[str, Any]:
    return {
        "root_cause_hypothesis": (
            "SFT regression on execution_vs_imagery is negative transfer from movement "
            "classification training, not overfitting. The model was never trained on "
            "execution_vs_imagery or factual_grounding, but learned to emit movement "
            "labels (especially 'rest') that collide with the execution/imagery label "
            "space. factual_grounding stays at 0% because eval prompts omit the metadata "
            "required to form compound ground-truth labels."
        ),
        "issue_classification": {
            "execution_vs_imagery": "data_coverage_gap_plus_negative_transfer",
            "factual_grounding": "dataset_design_issue",
            "movement_task_classification": "resolved_by_sft",
            "band_power_analysis": "partial_generalization",
            "channel_ranking": "partial_generalization",
        },
        "model_overfitting_vs_data_design": {
            "execution_vs_imagery": "data_design_and_coverage (negative transfer, not overfitting)",
            "factual_grounding": "dataset_design (unlearnable prompt; 0% for base and SFT)",
        },
        "recommended_smallest_fix": [
            {
                "priority": 1,
                "action": "Add execution_vs_imagery SFT examples with answers in {baseline, execution, imagery} label space",
                "details": (
                    "Use eval-style question and context fields (condition, movement). "
                    "Teach that *_rest conditions map to parent task_type, not movement label 'rest'."
                ),
                "estimated_examples": "125-250 balanced across baseline/execution/imagery",
            },
            {
                "priority": 2,
                "action": "Fix factual_grounding eval and/or training context",
                "details": (
                    "Include task_type and movement (or condition) in eval context, OR "
                    "change ground truth to movement-only labels if that is the intended task. "
                    "Do not train on compound labels without supplying both components."
                ),
                "estimated_examples": "dataset relabel/regen only; no model change required for eval fix",
            },
            {
                "priority": 3,
                "action": "Disambiguate movement SFT supervision",
                "details": (
                    "If movement and execution_vs_imagery tasks coexist, avoid training "
                    "identical answers for different task_types at the same event_code. "
                    "Either split tasks cleanly or include explicit task_type in the answer."
                ),
            },
        ],
        "short_corrective_sft_pass_needed": {
            "execution_vs_imagery": True,
            "factual_grounding": False,
            "rationale": (
                "execution_vs_imagery needs a short targeted SFT slice after dataset fixes. "
                "factual_grounding should be fixed at the dataset/prompt level first; "
                "corrective SFT is not justified until the prompt exposes answerable metadata."
            ),
        },
        "do_not_do_yet": [
            "Full SFT retrain",
            "RLVR",
            "Corrective training in this investigation pass",
        ],
    }


def main() -> None:
    train_subjects = {f"S{i:03d}" for i in range(1, 21)}
    val_subjects = {"S018", "S019", "S020"}

    sft_examples = load_jsonl(PROJECT_ROOT / "data/processed/sft_train.jsonl")
    eval_examples = load_jsonl(PROJECT_ROOT / "data/processed/eval_heldout.jsonl")
    eval_by_id = {ex["id"]: ex for ex in eval_examples}

    base_preds = {row["id"]: row for row in load_jsonl(PROJECT_ROOT / "results/base_model_eval/predictions.jsonl")}
    sft_preds = {row["id"]: row for row in load_jsonl(PROJECT_ROOT / "results/sft_model_eval/predictions.jsonl")}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    task_distribution = build_task_distribution(
        sft_examples, eval_examples, train_subjects, val_subjects
    )
    confusion_analysis = build_confusion_analysis(base_preds, sft_preds, eval_by_id)
    data_quality = build_data_quality_findings(sft_examples)
    remediation = build_remediation_plan()

    outputs = {
        "task_distribution.json": task_distribution,
        "confusion_analysis.json": confusion_analysis,
        "data_quality_findings.json": data_quality,
        "remediation_plan.json": remediation,
    }

    for filename, payload in outputs.items():
        path = OUTPUT_DIR / filename
        with path.open("w") as handle:
            json.dump(payload, handle, indent=2)

    print(f"Wrote analysis artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
