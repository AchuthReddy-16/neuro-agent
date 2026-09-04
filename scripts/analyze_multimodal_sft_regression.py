#!/usr/bin/env python3
"""Stage F.1 Step 1: Compare base vs multimodal SFT predictions and classify regressions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.evaluation.verifiers import normalize_token, verify_example
from neuro_agent.paths import PROJECT_ROOT

VERBOSE_PATTERNS = (
    r"is the grounded result",
    r"their values are",
    r"computed across the supplied values",
    r"is higher:",
    r"the computed result is",
    r"the grounded descending order is",
)

REGRESSION_GROUPS = ("A", "B", "C", "D", "E", "F")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(path)}


def is_verbose_response(response: str) -> bool:
    lower = response.lower()
    return any(re.search(p, lower) for p in VERBOSE_PATTERNS)


def _gt_tokens(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, list):
        return [normalize_token(str(x)) for x in ground_truth]
    return [normalize_token(str(ground_truth))]


def is_semantically_correct(example: dict[str, Any], response: str) -> bool:
    """Heuristic: correct answer appears in response before verifier parsing."""
    vtype = example["verification_type"]
    gt_tokens = _gt_tokens(example["ground_truth"])
    resp_norm = normalize_token(response)

    if vtype == "categorical":
        gt = gt_tokens[0]
        if gt in KNOWN_SHORT_LABELS and gt in resp_norm:
            return True
        # EEG channel labels (C3, CZ, AF7, etc.)
        raw_gt = str(example["ground_truth"]).upper()
        if re.search(rf"\b{re.escape(raw_gt)}\b", response.upper()):
            return True
        return gt in resp_norm.split("_")

    if vtype == "numeric":
        gt = float(example["ground_truth"])
        nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)", response)
        for n in nums:
            try:
                if abs(float(n) - gt) < max(0.05, abs(gt) * 0.02):
                    return True
            except ValueError:
                continue
        return False

    if vtype == "ranking":
        expected = [str(x).upper() for x in example["ground_truth"]]
        text = response.upper()
        return any(ch in text for ch in expected)

    if vtype == "set":
        expected = {str(x).upper() for x in example["ground_truth"]}
        text = response.upper()
        return all(any(ch in text for ch in [e]) for e in expected)

    return False


KNOWN_SHORT_LABELS = {
    "rest",
    "movement",
    "beta_power",
    "alpha_power",
    "delta_power",
    "theta_power",
    "gamma_power",
    "mu_power",
    "left_fist",
    "right_fist",
    "both_fists",
    "both_feet",
    "baseline",
    "execution",
    "imagery",
}


def classify_regression(
    example: dict[str, Any],
    base_row: dict[str, Any],
    sft_row: dict[str, Any],
) -> str:
    """Assign regression bucket A-F for base-pass / sft-fail cases."""
    base_resp = base_row["response"]
    sft_resp = sft_row["response"]
    sft_ver = sft_row["verification"]
    base_tokens = base_row.get("generated_tokens", 0)
    sft_tokens = sft_row.get("generated_tokens", 0)

    if is_semantically_correct(example, sft_resp) and not sft_ver.get("passed"):
        return "A"

    if is_verbose_response(sft_resp):
        return "B"

    parsed = str(sft_ver.get("parsed_answer", ""))
    expected = normalize_token(str(example["ground_truth"]))
    if parsed and parsed != expected and not is_semantically_correct(example, sft_resp):
        # Wrong label vocab / wrong channel picked
        if example["verification_type"] == "categorical":
            return "C"

    if sft_tokens > base_tokens * 1.5 and sft_tokens > 20:
        return "E"

    if not is_semantically_correct(example, sft_resp):
        return "D"

    return "F"


def build_comparison_row(
    example: dict[str, Any],
    base_row: dict[str, Any],
    sft_row: dict[str, Any],
) -> dict[str, Any]:
    base_pass = bool(base_row["verification"]["passed"])
    sft_pass = bool(sft_row["verification"]["passed"])
    regressed = base_pass and not sft_pass
    improved = (not base_pass) and sft_pass

    row: dict[str, Any] = {
        "id": example["id"],
        "task_family": example.get("category", example.get("task_family")),
        "verification_type": example["verification_type"],
        "ground_truth": example["ground_truth"],
        "base_pass": base_pass,
        "sft_pass": sft_pass,
        "regressed": regressed,
        "improved": improved,
        "base_parsed": base_row["verification"].get("parsed_answer"),
        "sft_parsed": sft_row["verification"].get("parsed_answer"),
        "base_response": base_row["response"],
        "sft_response": sft_row["response"],
        "base_tokens": base_row.get("generated_tokens"),
        "sft_tokens": sft_row.get("generated_tokens"),
        "base_reason": base_row["verification"].get("reason"),
        "sft_reason": sft_row["verification"].get("reason"),
        "sft_semantically_correct": is_semantically_correct(example, sft_row["response"]),
        "sft_verbose": is_verbose_response(sft_row["response"]),
    }
    if regressed:
        row["regression_group"] = classify_regression(example, base_row, sft_row)
    return row


def summarize_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regressed = [r for r in rows if r.get("regressed")]
    by_group = Counter(r.get("regression_group", "F") for r in regressed)
    by_task = Counter(r["task_family"] for r in regressed)
    by_verifier = Counter(r["verification_type"] for r in regressed)

    representatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in regressed:
        grp = r.get("regression_group", "F")
        if len(representatives[grp]) < 5:
            representatives[grp].append(
                {
                    "id": r["id"],
                    "task_family": r["task_family"],
                    "verification_type": r["verification_type"],
                    "ground_truth": r["ground_truth"],
                    "base_response": r["base_response"][:200],
                    "sft_response": r["sft_response"][:200],
                    "sft_parsed": r["sft_parsed"],
                }
            )

    return {
        "total_regressions": len(regressed),
        "by_group": dict(by_group),
        "by_task_family": dict(by_task.most_common()),
        "by_verifier_type": dict(by_verifier.most_common()),
        "representative_examples": dict(representatives),
        "group_definitions": {
            "A": "semantically correct but parser failed",
            "B": "wrong format (verbose grounded-answer template)",
            "C": "wrong label vocab",
            "D": "genuine visual reasoning failure",
            "E": "overly verbose",
            "F": "other",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal SFT regression error audit")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT / "results/multimodal_base_eval",
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=PROJECT_ROOT / "results/multimodal_sft_eval",
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

    base_preds = load_predictions(args.base_dir / "predictions.jsonl")
    sft_preds = load_predictions(args.sft_dir / "predictions.jsonl")
    common_ids = sorted(set(base_preds) & set(sft_preds))
    if len(common_ids) != len(base_preds):
        raise ValueError(f"Prediction ID mismatch: {len(common_ids)} common vs {len(base_preds)} base")

    comparison_rows: list[dict[str, Any]] = []
    for eid in common_ids:
        base_row = base_preds[eid]
        example = {
            "id": eid,
            "category": base_row["category"],
            "verification_type": base_row["verification_type"],
            "ground_truth": base_row["ground_truth"],
            "question": base_row.get("question", ""),
        }
        comparison_rows.append(build_comparison_row(example, base_row, sft_preds[eid]))

    regressed = [r for r in comparison_rows if r["regressed"]]
    improved = [r for r in comparison_rows if r["improved"]]
    unchanged_pass = sum(1 for r in comparison_rows if r["base_pass"] and r["sft_pass"])
    unchanged_fail = sum(1 for r in comparison_rows if not r["base_pass"] and not r["sft_pass"])

    summary = {
        "total_examples": len(comparison_rows),
        "base_pass_count": sum(1 for r in comparison_rows if r["base_pass"]),
        "sft_pass_count": sum(1 for r in comparison_rows if r["sft_pass"]),
        "regressed_count": len(regressed),
        "improved_count": len(improved),
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
        "avg_base_tokens": sum(r["base_tokens"] or 0 for r in comparison_rows) / len(comparison_rows),
        "avg_sft_tokens": sum(r["sft_tokens"] or 0 for r in comparison_rows) / len(comparison_rows),
        "regression_analysis": summarize_groups(comparison_rows),
        "per_task_delta": _per_task_delta(comparison_rows),
        "per_verifier_delta": _per_verifier_delta(comparison_rows),
    }

    with (args.output_dir / "comparison.jsonl").open("w") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "regressions.jsonl").open("w") as handle:
        for row in regressed:
            handle.write(json.dumps(row) + "\n")
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


def _per_task_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_family"]].append(row)
    out: dict[str, Any] = {}
    for task, task_rows in sorted(by_task.items()):
        n = len(task_rows)
        base_rate = sum(1 for r in task_rows if r["base_pass"]) / n
        sft_rate = sum(1 for r in task_rows if r["sft_pass"]) / n
        out[task] = {
            "count": n,
            "base_pass_rate": round(base_rate, 4),
            "sft_pass_rate": round(sft_rate, 4),
            "delta": round(sft_rate - base_rate, 4),
            "regressed": sum(1 for r in task_rows if r["regressed"]),
            "improved": sum(1 for r in task_rows if r["improved"]),
        }
    return out


def _per_verifier_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_v: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_v[row["verification_type"]].append(row)
    out: dict[str, Any] = {}
    for vtype, v_rows in sorted(by_v.items()):
        n = len(v_rows)
        base_rate = sum(1 for r in v_rows if r["base_pass"]) / n
        sft_rate = sum(1 for r in v_rows if r["sft_pass"]) / n
        out[vtype] = {
            "count": n,
            "base_pass_rate": round(base_rate, 4),
            "sft_pass_rate": round(sft_rate, 4),
            "delta": round(sft_rate - base_rate, 4),
            "regressed": sum(1 for r in v_rows if r["regressed"]),
            "verbose_sft_failures": sum(
                1 for r in v_rows if not r["sft_pass"] and r.get("sft_verbose")
            ),
        }
    return out


if __name__ == "__main__":
    main()
