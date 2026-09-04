#!/usr/bin/env python3
"""Compare base and SFT evaluation summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def build_four_way_comparison(
    base_dir: Path,
    sft_dir: Path,
    corrected_dir: Path,
    rlvr_dir: Path,
) -> dict[str, Any]:
    base_summary = _load_json(base_dir / "summary.json")
    sft_summary = _load_json(sft_dir / "summary.json")
    corrected_summary = _load_json(corrected_dir / "summary.json")
    rlvr_summary = _load_json(rlvr_dir / "summary.json")
    base_tasks = _load_json(base_dir / "per_task_metrics.json")
    sft_tasks = _load_json(sft_dir / "per_task_metrics.json")
    corrected_tasks = _load_json(corrected_dir / "per_task_metrics.json")
    rlvr_tasks = _load_json(rlvr_dir / "per_task_metrics.json")

    per_task: dict[str, Any] = {}
    all_tasks = sorted(set(base_tasks) | set(sft_tasks) | set(corrected_tasks) | set(rlvr_tasks))
    for task in all_tasks:
        base_rate = base_tasks.get(task, {}).get("verifier_pass_rate")
        sft_rate = sft_tasks.get(task, {}).get("verifier_pass_rate")
        corrected_rate = corrected_tasks.get(task, {}).get("verifier_pass_rate")
        rlvr_rate = rlvr_tasks.get(task, {}).get("verifier_pass_rate")
        per_task[task] = {
            "base_pass_rate": base_rate,
            "original_sft_pass_rate": sft_rate,
            "corrected_sft_pass_rate": corrected_rate,
            "rlvr_pass_rate": rlvr_rate,
            "original_sft_vs_base_delta": _delta(base_rate, sft_rate),
            "corrected_sft_vs_base_delta": _delta(base_rate, corrected_rate),
            "corrected_vs_original_sft_delta": _delta(sft_rate, corrected_rate),
            "rlvr_vs_corrected_v2_delta": _delta(corrected_rate, rlvr_rate),
            "rlvr_vs_base_delta": _delta(base_rate, rlvr_rate),
            "count": rlvr_tasks.get(task, {}).get("count")
            or corrected_tasks.get(task, {}).get("count")
            or sft_tasks.get(task, {}).get("count")
            or base_tasks.get(task, {}).get("count"),
        }

    strong_tasks = [
        "numerical_reasoning",
        "tool_selection",
        "statistical_comparison",
        "execution_vs_imagery",
    ]
    preserve_tasks = [
        "movement_task_classification",
        "band_power_analysis",
        "channel_ranking",
    ]

    strong_regressed = [
        task
        for task in strong_tasks
        if per_task.get(task, {}).get("rlvr_vs_corrected_v2_delta") is not None
        and per_task[task]["rlvr_vs_corrected_v2_delta"] < -0.05
    ]
    preserve_regressed = [
        task
        for task in preserve_tasks
        if per_task.get(task, {}).get("rlvr_vs_corrected_v2_delta") is not None
        and per_task[task]["rlvr_vs_corrected_v2_delta"] < -0.05
    ]

    return {
        "base_variant": base_summary.get("variant"),
        "original_sft_variant": sft_summary.get("variant"),
        "corrected_sft_variant": corrected_summary.get("variant"),
        "rlvr_variant": rlvr_summary.get("variant"),
        "eval_note": (
            "Base and original SFT evaluated on eval_heldout.jsonl; "
            "corrected SFT v2 and RLVR on eval_heldout_corrected.jsonl."
        ),
        "overall": {
            "base_pass_rate": base_summary.get("verifier_pass_rate"),
            "original_sft_pass_rate": sft_summary.get("verifier_pass_rate"),
            "corrected_sft_pass_rate": corrected_summary.get("verifier_pass_rate"),
            "rlvr_pass_rate": rlvr_summary.get("verifier_pass_rate"),
            "original_sft_vs_base_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                sft_summary.get("verifier_pass_rate"),
            ),
            "corrected_sft_vs_base_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                corrected_summary.get("verifier_pass_rate"),
            ),
            "corrected_vs_original_sft_delta": _delta(
                sft_summary.get("verifier_pass_rate"),
                corrected_summary.get("verifier_pass_rate"),
            ),
            "rlvr_vs_corrected_v2_delta": _delta(
                corrected_summary.get("verifier_pass_rate"),
                rlvr_summary.get("verifier_pass_rate"),
            ),
            "rlvr_vs_base_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                rlvr_summary.get("verifier_pass_rate"),
            ),
        },
        "per_task": per_task,
        "quality_check": {
            "strong_tasks_substantially_regressed_vs_corrected_v2": strong_regressed,
            "preserve_tasks_substantially_regressed_vs_corrected_v2": preserve_regressed,
            "factual_grounding_rlvr_rate": per_task.get("factual_grounding", {}).get(
                "rlvr_pass_rate"
            ),
            "pass": not strong_regressed and not preserve_regressed,
        },
    }


def build_three_way_comparison(
    base_dir: Path,
    sft_dir: Path,
    corrected_dir: Path,
) -> dict[str, Any]:
    base_summary = _load_json(base_dir / "summary.json")
    sft_summary = _load_json(sft_dir / "summary.json")
    corrected_summary = _load_json(corrected_dir / "summary.json")
    base_tasks = _load_json(base_dir / "per_task_metrics.json")
    sft_tasks = _load_json(sft_dir / "per_task_metrics.json")
    corrected_tasks = _load_json(corrected_dir / "per_task_metrics.json")

    per_task: dict[str, Any] = {}
    all_tasks = sorted(set(base_tasks) | set(sft_tasks) | set(corrected_tasks))
    for task in all_tasks:
        base_rate = base_tasks.get(task, {}).get("verifier_pass_rate")
        sft_rate = sft_tasks.get(task, {}).get("verifier_pass_rate")
        corrected_rate = corrected_tasks.get(task, {}).get("verifier_pass_rate")
        per_task[task] = {
            "base_pass_rate": base_rate,
            "original_sft_pass_rate": sft_rate,
            "corrected_sft_pass_rate": corrected_rate,
            "original_sft_vs_base_delta": _delta(base_rate, sft_rate),
            "corrected_sft_vs_base_delta": _delta(base_rate, corrected_rate),
            "corrected_vs_original_sft_delta": _delta(sft_rate, corrected_rate),
            "count": corrected_tasks.get(task, {}).get("count")
            or sft_tasks.get(task, {}).get("count")
            or base_tasks.get(task, {}).get("count"),
        }

    strong_tasks = [
        "numerical_reasoning",
        "tool_selection",
        "statistical_comparison",
    ]
    preserve_tasks = [
        "movement_task_classification",
        "band_power_analysis",
        "channel_ranking",
    ]
    target_task = "execution_vs_imagery"

    strong_regressed_vs_original = [
        task
        for task in strong_tasks
        if per_task.get(task, {}).get("corrected_vs_original_sft_delta") is not None
        and per_task[task]["corrected_vs_original_sft_delta"] < -0.05
    ]
    preserve_regressed = [
        task
        for task in preserve_tasks
        if per_task.get(task, {}).get("corrected_vs_original_sft_delta") is not None
        and per_task[task]["corrected_vs_original_sft_delta"] < -0.05
    ]

    execution_imagery = per_task.get(target_task, {})
    execution_improved = (
        execution_imagery.get("corrected_sft_pass_rate") is not None
        and execution_imagery.get("original_sft_pass_rate") is not None
        and execution_imagery["corrected_sft_pass_rate"]
        > execution_imagery["original_sft_pass_rate"] + 0.05
    )

    return {
        "base_variant": base_summary.get("variant"),
        "original_sft_variant": sft_summary.get("variant"),
        "corrected_sft_variant": corrected_summary.get("variant"),
        "eval_note": (
            "Base and original SFT evaluated on eval_heldout.jsonl; "
            "corrected SFT on eval_heldout_corrected.jsonl "
            "(875/1000 examples identical; 125 factual_grounding prompts differ)."
        ),
        "overall": {
            "base_pass_rate": base_summary.get("verifier_pass_rate"),
            "original_sft_pass_rate": sft_summary.get("verifier_pass_rate"),
            "corrected_sft_pass_rate": corrected_summary.get("verifier_pass_rate"),
            "original_sft_vs_base_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                sft_summary.get("verifier_pass_rate"),
            ),
            "corrected_sft_vs_base_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                corrected_summary.get("verifier_pass_rate"),
            ),
            "corrected_vs_original_sft_delta": _delta(
                sft_summary.get("verifier_pass_rate"),
                corrected_summary.get("verifier_pass_rate"),
            ),
        },
        "per_task": per_task,
        "success_criteria": {
            "execution_vs_imagery_substantially_improved": execution_improved,
            "execution_vs_imagery_corrected_rate": execution_imagery.get(
                "corrected_sft_pass_rate"
            ),
            "execution_vs_imagery_original_sft_rate": execution_imagery.get(
                "original_sft_pass_rate"
            ),
            "strong_tasks_substantially_regressed": strong_regressed_vs_original,
            "preserve_tasks_substantially_regressed": preserve_regressed,
            "factual_grounding_corrected_rate": per_task.get("factual_grounding", {}).get(
                "corrected_sft_pass_rate"
            ),
            "pass": execution_improved
            and not strong_regressed_vs_original
            and not preserve_regressed,
        },
    }


def build_comparison(base_dir: Path, sft_dir: Path) -> dict[str, Any]:
    base_summary = _load_json(base_dir / "summary.json")
    sft_summary = _load_json(sft_dir / "summary.json")
    base_tasks = _load_json(base_dir / "per_task_metrics.json")
    sft_tasks = _load_json(sft_dir / "per_task_metrics.json")
    base_verifiers = _load_json(base_dir / "verifier_summary.json")
    sft_verifiers = _load_json(sft_dir / "verifier_summary.json")

    per_task: dict[str, Any] = {}
    all_tasks = sorted(set(base_tasks) | set(sft_tasks))
    for task in all_tasks:
        base_rate = base_tasks.get(task, {}).get("verifier_pass_rate")
        sft_rate = sft_tasks.get(task, {}).get("verifier_pass_rate")
        per_task[task] = {
            "base_pass_rate": base_rate,
            "sft_pass_rate": sft_rate,
            "delta": _delta(base_rate, sft_rate),
            "base_count": base_tasks.get(task, {}).get("count"),
            "sft_count": sft_tasks.get(task, {}).get("count"),
        }

    per_verifier: dict[str, Any] = {}
    all_verifiers = sorted(set(base_verifiers) | set(sft_verifiers))
    for vtype in all_verifiers:
        base_rate = base_verifiers.get(vtype, {}).get("verifier_pass_rate")
        sft_rate = sft_verifiers.get(vtype, {}).get("verifier_pass_rate")
        per_verifier[vtype] = {
            "base_pass_rate": base_rate,
            "sft_pass_rate": sft_rate,
            "delta": _delta(base_rate, sft_rate),
        }

    weak_tasks = [
        "movement_task_classification",
        "factual_grounding",
        "execution_vs_imagery",
        "band_power_analysis",
        "channel_ranking",
    ]
    strong_tasks = [
        "numerical_reasoning",
        "tool_selection",
        "statistical_comparison",
    ]

    weak_improved = [
        task
        for task in weak_tasks
        if per_task.get(task, {}).get("delta") is not None and per_task[task]["delta"] > 0
    ]
    strong_regressed = [
        task
        for task in strong_tasks
        if per_task.get(task, {}).get("delta") is not None and per_task[task]["delta"] < -0.05
    ]

    return {
        "base_variant": base_summary.get("variant"),
        "sft_variant": sft_summary.get("variant"),
        "overall": {
            "base_pass_rate": base_summary.get("verifier_pass_rate"),
            "sft_pass_rate": sft_summary.get("verifier_pass_rate"),
            "delta": _delta(
                base_summary.get("verifier_pass_rate"),
                sft_summary.get("verifier_pass_rate"),
            ),
        },
        "parse_refusal": {
            "base_invalid_parse_rate": base_summary.get("invalid_parse_rate"),
            "sft_invalid_parse_rate": sft_summary.get("invalid_parse_rate"),
            "invalid_parse_delta": _delta(
                base_summary.get("invalid_parse_rate"),
                sft_summary.get("invalid_parse_rate"),
            ),
            "base_empty_refusal_rate": base_summary.get("empty_refusal_rate"),
            "sft_empty_refusal_rate": sft_summary.get("empty_refusal_rate"),
            "empty_refusal_delta": _delta(
                base_summary.get("empty_refusal_rate"),
                sft_summary.get("empty_refusal_rate"),
            ),
        },
        "runtime": {
            "base_runtime_s": base_summary.get("runtime_s"),
            "sft_runtime_s": sft_summary.get("runtime_s"),
            "base_peak_vram_mb": base_summary.get("peak_torch_allocated_mb"),
            "sft_peak_vram_mb": sft_summary.get("peak_torch_allocated_mb"),
        },
        "unsupported_claim": {
            "base_rate": base_summary.get("unsupported_claim_rate"),
            "sft_rate": sft_summary.get("unsupported_claim_rate"),
            "delta": _delta(
                base_summary.get("unsupported_claim_rate"),
                sft_summary.get("unsupported_claim_rate"),
            ),
        },
        "per_task": per_task,
        "per_verifier_type": per_verifier,
        "quality_check": {
            "weak_tasks_improved": weak_improved,
            "strong_tasks_substantially_regressed": strong_regressed,
            "tradeoff_flag": bool(weak_improved and strong_regressed),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs SFT evaluation results")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "eval_sft.yaml",
        help="Path to eval config containing comparison paths",
    )
    parser.add_argument(
        "--four-way",
        action="store_true",
        help="Compare base, original SFT, corrected SFT v2, and RLVR",
    )
    parser.add_argument(
        "--three-way",
        action="store_true",
        help="Compare base, original SFT, and corrected SFT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    if args.four_way or "rlvr_dir" in cfg.get("comparison", {}):
        comparison_cfg = cfg["comparison"]
        base_dir = PROJECT_ROOT / comparison_cfg["base_dir"]
        sft_dir = PROJECT_ROOT / comparison_cfg.get("sft_dir", cfg["output"]["dir"])
        corrected_dir = PROJECT_ROOT / comparison_cfg["corrected_dir"]
        rlvr_dir = PROJECT_ROOT / comparison_cfg["rlvr_dir"]
        output_path = PROJECT_ROOT / comparison_cfg["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison = build_four_way_comparison(base_dir, sft_dir, corrected_dir, rlvr_dir)
    elif args.three_way or "corrected_dir" in cfg.get("comparison", {}):
        comparison_cfg = cfg["comparison"]
        base_dir = PROJECT_ROOT / comparison_cfg["base_dir"]
        sft_dir = PROJECT_ROOT / comparison_cfg.get("sft_dir", cfg["output"]["dir"])
        corrected_dir = PROJECT_ROOT / comparison_cfg["corrected_dir"]
        output_path = PROJECT_ROOT / comparison_cfg["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison = build_three_way_comparison(base_dir, sft_dir, corrected_dir)
    else:
        comparison_cfg = cfg["comparison"]
        base_dir = PROJECT_ROOT / comparison_cfg["base_dir"]
        sft_dir = PROJECT_ROOT / cfg["output"]["dir"]
        output_path = PROJECT_ROOT / comparison_cfg["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison = build_comparison(base_dir, sft_dir)

    with output_path.open("w") as handle:
        json.dump(comparison, handle, indent=2)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
