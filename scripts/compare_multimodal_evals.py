#!/usr/bin/env python3
"""Compare base vs multimodal SFT evaluation results."""

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


def build_multimodal_comparison(base_dir: Path, sft_dir: Path) -> dict[str, Any]:
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
        base_count = base_tasks.get(task, {}).get("count", 0)
        sft_count = sft_tasks.get(task, {}).get("count", 0)
        per_task[task] = {
            "base_pass_rate": base_rate,
            "sft_pass_rate": sft_rate,
            "delta": _delta(base_rate, sft_rate),
            "count": sft_count or base_count,
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

    base_overall = base_summary.get("verifier_pass_rate", 0.0)
    sft_overall = sft_summary.get("verifier_pass_rate", 0.0)
    overall_delta = _delta(base_overall, sft_overall)

    large_families = [
        task
        for task, metrics in per_task.items()
        if metrics.get("count", 0) >= 50
    ]
    collapsed = [
        task
        for task in large_families
        if per_task[task].get("delta") is not None and per_task[task]["delta"] < -0.10
    ]

    meaningful_improvement = (
        overall_delta is not None and overall_delta >= 0.03 and not collapsed
    )

    return {
        "base_variant": base_summary.get("variant"),
        "sft_variant": sft_summary.get("variant"),
        "overall": {
            "base_pass_rate": base_overall,
            "sft_pass_rate": sft_overall,
            "delta": overall_delta,
        },
        "runtime": {
            "base_runtime_s": base_summary.get("runtime_s"),
            "sft_runtime_s": sft_summary.get("runtime_s"),
            "base_peak_vram_mb": base_summary.get("peak_torch_allocated_mb"),
            "sft_peak_vram_mb": sft_summary.get("peak_torch_allocated_mb"),
        },
        "per_task": per_task,
        "per_verifier_type": per_verifier,
        "gate": {
            "meaningful_improvement": meaningful_improvement,
            "large_family_collapse": collapsed,
            "pass": meaningful_improvement,
        },
    }


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
    base_verifiers = _load_json(base_dir / "verifier_summary.json")
    sft_verifiers = _load_json(sft_dir / "verifier_summary.json")
    corrected_verifiers = _load_json(corrected_dir / "verifier_summary.json")
    rlvr_verifiers = _load_json(rlvr_dir / "verifier_summary.json")

    per_task: dict[str, Any] = {}
    all_tasks = sorted(set(base_tasks) | set(sft_tasks) | set(corrected_tasks) | set(rlvr_tasks))
    for task in all_tasks:
        base_rate = base_tasks.get(task, {}).get("verifier_pass_rate")
        sft_rate = sft_tasks.get(task, {}).get("verifier_pass_rate")
        corrected_rate = corrected_tasks.get(task, {}).get("verifier_pass_rate")
        rlvr_rate = rlvr_tasks.get(task, {}).get("verifier_pass_rate")
        count = (
            rlvr_tasks.get(task, {}).get("count")
            or corrected_tasks.get(task, {}).get("count")
            or sft_tasks.get(task, {}).get("count", 0)
        )
        per_task[task] = {
            "base_pass_rate": base_rate,
            "sft_pass_rate": sft_rate,
            "corrected_pass_rate": corrected_rate,
            "rlvr_pass_rate": rlvr_rate,
            "sft_delta": _delta(base_rate, sft_rate),
            "corrected_delta": _delta(base_rate, corrected_rate),
            "rlvr_delta": _delta(base_rate, rlvr_rate),
            "rlvr_vs_corrected_delta": _delta(corrected_rate, rlvr_rate),
            "count": count,
        }

    per_verifier: dict[str, Any] = {}
    all_verifiers = sorted(
        set(base_verifiers) | set(sft_verifiers) | set(corrected_verifiers) | set(rlvr_verifiers)
    )
    for vtype in all_verifiers:
        per_verifier[vtype] = {
            "base_pass_rate": base_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            "sft_pass_rate": sft_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            "corrected_pass_rate": corrected_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            "rlvr_pass_rate": rlvr_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            "corrected_delta": _delta(
                base_verifiers.get(vtype, {}).get("verifier_pass_rate"),
                corrected_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            ),
            "rlvr_delta": _delta(
                base_verifiers.get(vtype, {}).get("verifier_pass_rate"),
                rlvr_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            ),
            "rlvr_vs_corrected_delta": _delta(
                corrected_verifiers.get(vtype, {}).get("verifier_pass_rate"),
                rlvr_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            ),
        }

    return {
        "base_variant": base_summary.get("variant"),
        "sft_variant": sft_summary.get("variant"),
        "corrected_variant": corrected_summary.get("variant"),
        "rlvr_variant": rlvr_summary.get("variant"),
        "overall": {
            "base_pass_rate": base_summary.get("verifier_pass_rate"),
            "sft_pass_rate": sft_summary.get("verifier_pass_rate"),
            "corrected_pass_rate": corrected_summary.get("verifier_pass_rate"),
            "rlvr_pass_rate": rlvr_summary.get("verifier_pass_rate"),
            "corrected_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                corrected_summary.get("verifier_pass_rate"),
            ),
            "rlvr_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                rlvr_summary.get("verifier_pass_rate"),
            ),
            "rlvr_vs_corrected_delta": _delta(
                corrected_summary.get("verifier_pass_rate"),
                rlvr_summary.get("verifier_pass_rate"),
            ),
        },
        "runtime": {
            "base_runtime_s": base_summary.get("runtime_s"),
            "sft_runtime_s": sft_summary.get("runtime_s"),
            "corrected_runtime_s": corrected_summary.get("runtime_s"),
            "rlvr_runtime_s": rlvr_summary.get("runtime_s"),
            "base_peak_vram_mb": base_summary.get("peak_torch_allocated_mb"),
            "sft_peak_vram_mb": sft_summary.get("peak_torch_allocated_mb"),
            "corrected_peak_vram_mb": corrected_summary.get("peak_torch_allocated_mb"),
            "rlvr_peak_vram_mb": rlvr_summary.get("peak_torch_allocated_mb"),
        },
        "per_task": per_task,
        "per_verifier_type": per_verifier,
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
    base_verifiers = _load_json(base_dir / "verifier_summary.json")
    sft_verifiers = _load_json(sft_dir / "verifier_summary.json")
    corrected_verifiers = _load_json(corrected_dir / "verifier_summary.json")

    per_task: dict[str, Any] = {}
    all_tasks = sorted(set(base_tasks) | set(sft_tasks) | set(corrected_tasks))
    for task in all_tasks:
        base_rate = base_tasks.get(task, {}).get("verifier_pass_rate")
        sft_rate = sft_tasks.get(task, {}).get("verifier_pass_rate")
        corrected_rate = corrected_tasks.get(task, {}).get("verifier_pass_rate")
        count = corrected_tasks.get(task, {}).get("count") or sft_tasks.get(task, {}).get("count", 0)
        per_task[task] = {
            "base_pass_rate": base_rate,
            "sft_pass_rate": sft_rate,
            "corrected_pass_rate": corrected_rate,
            "sft_delta": _delta(base_rate, sft_rate),
            "corrected_delta": _delta(base_rate, corrected_rate),
            "count": count,
        }

    per_verifier: dict[str, Any] = {}
    all_verifiers = sorted(set(base_verifiers) | set(sft_verifiers) | set(corrected_verifiers))
    for vtype in all_verifiers:
        per_verifier[vtype] = {
            "base_pass_rate": base_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            "sft_pass_rate": sft_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            "corrected_pass_rate": corrected_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            "sft_delta": _delta(
                base_verifiers.get(vtype, {}).get("verifier_pass_rate"),
                sft_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            ),
            "corrected_delta": _delta(
                base_verifiers.get(vtype, {}).get("verifier_pass_rate"),
                corrected_verifiers.get(vtype, {}).get("verifier_pass_rate"),
            ),
        }

    return {
        "base_variant": base_summary.get("variant"),
        "sft_variant": sft_summary.get("variant"),
        "corrected_variant": corrected_summary.get("variant"),
        "overall": {
            "base_pass_rate": base_summary.get("verifier_pass_rate"),
            "sft_pass_rate": sft_summary.get("verifier_pass_rate"),
            "corrected_pass_rate": corrected_summary.get("verifier_pass_rate"),
            "sft_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                sft_summary.get("verifier_pass_rate"),
            ),
            "corrected_delta": _delta(
                base_summary.get("verifier_pass_rate"),
                corrected_summary.get("verifier_pass_rate"),
            ),
        },
        "runtime": {
            "base_runtime_s": base_summary.get("runtime_s"),
            "sft_runtime_s": sft_summary.get("runtime_s"),
            "corrected_runtime_s": corrected_summary.get("runtime_s"),
            "base_peak_vram_mb": base_summary.get("peak_torch_allocated_mb"),
            "sft_peak_vram_mb": sft_summary.get("peak_torch_allocated_mb"),
            "corrected_peak_vram_mb": corrected_summary.get("peak_torch_allocated_mb"),
        },
        "per_task": per_task,
        "per_verifier_type": per_verifier,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base vs multimodal SFT evals")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "multimodal_sft.yaml",
        help="Path to multimodal SFT YAML config",
    )
    parser.add_argument(
        "--sft-dir",
        type=Path,
        default=None,
        help="Override SFT eval results directory",
    )
    parser.add_argument(
        "--corrected-dir",
        type=Path,
        default=None,
        help="Include corrected SFT eval results for three-way comparison",
    )
    parser.add_argument(
        "--rlvr-dir",
        type=Path,
        default=None,
        help="Include RLVR eval results for four-way comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    comparison_cfg = cfg.get("comparison", {})
    base_dir = PROJECT_ROOT / comparison_cfg.get("base_dir", "results/multimodal_base_eval")
    sft_dir = args.sft_dir or PROJECT_ROOT / comparison_cfg.get(
        "sft_dir", "results/multimodal_sft_eval"
    )
    output_path = PROJECT_ROOT / comparison_cfg.get(
        "output_path",
        "results/model_comparison/multimodal_base_vs_sft.json",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corrected_dir = args.corrected_dir or (
        PROJECT_ROOT / comparison_cfg["corrected_dir"]
        if comparison_cfg.get("corrected_dir")
        else None
    )
    rlvr_dir = args.rlvr_dir or (
        PROJECT_ROOT / comparison_cfg["rlvr_dir"]
        if comparison_cfg.get("rlvr_dir")
        else None
    )
    if rlvr_dir and rlvr_dir.exists() and corrected_dir and corrected_dir.exists():
        comparison = build_four_way_comparison(base_dir, sft_dir, corrected_dir, rlvr_dir)
    elif corrected_dir and corrected_dir.exists():
        comparison = build_three_way_comparison(base_dir, sft_dir, corrected_dir)
    else:
        comparison = build_multimodal_comparison(base_dir, sft_dir)
    with output_path.open("w") as handle:
        json.dump(comparison, handle, indent=2)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
