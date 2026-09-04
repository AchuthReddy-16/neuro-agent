#!/usr/bin/env python3
"""Stage F.1 Step 2: Decode-format ablation on existing multimodal SFT checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.evaluation.llm_eval import load_eval_examples, verify_heldout_integrity
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.multimodal.eval import MultimodalEvalConfig, run_multimodal_evaluation
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs

TASK_SUFFIXES: dict[str, str] = {
    "categorical": "Reply with only the final label or channel name. No explanation.",
    "ranking": "Reply with comma-separated channel names only, highest first. No values or explanation.",
    "numeric": "Reply with a single number only. No units or explanation.",
    "set": "Reply with comma-separated channel names only. No explanation.",
    "comparison": "Reply with only the final label. No explanation.",
}

ABLATION_CONDITIONS = [
    {
        "name": "current",
        "max_new_tokens": 128,
        "system_prompt": None,
        "task_suffix": False,
    },
    {
        "name": "low_tokens",
        "max_new_tokens": 16,
        "system_prompt": None,
        "task_suffix": False,
    },
    {
        "name": "answer_only",
        "max_new_tokens": 32,
        "system_prompt": (
            "You are a neuroscience research assistant analyzing EEG-derived plots. "
            "Use the provided image and context. Respond with ONLY the direct answer — "
            "a single label, channel name, number, or comma-separated list. "
            "Never add explanation, units, or supporting values."
        ),
        "task_suffix": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal SFT format ablation")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "multimodal_sft.yaml",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Subset of condition names to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/multimodal_format_ablation",
    )
    return parser.parse_args()


def _apply_task_suffix(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for ex in examples:
        copy = dict(ex)
        vtype = ex.get("verification_type", ex.get("task_class", "categorical"))
        suffix = TASK_SUFFIXES.get(vtype, TASK_SUFFIXES["categorical"])
        q = (copy.get("question") or copy.get("researcher_question") or "").strip()
        copy["question"] = f"{q}\n\n{suffix}"
        augmented.append(copy)
    return augmented


def run_condition(
    *,
    condition: dict[str, Any],
    examples: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    model_cfg = cfg["model"]
    inf_cfg = cfg.get("inference", {})
    eval_cfg = cfg.get("evaluation", {})
    prompt_cfg = cfg["prompt"]
    adapter_path = PROJECT_ROOT / cfg["output"]["checkpoint_dir"] / "final"

    system_prompt = condition["system_prompt"] or prompt_cfg["system"]
    eval_examples = _apply_task_suffix(examples) if condition["task_suffix"] else examples

    out_dir = output_root / condition["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        print(f"Skipping {condition['name']}: existing results at {out_dir}")
        summary = json.loads(summary_path.read_text())
        return {
            "condition": condition["name"],
            "max_new_tokens": condition["max_new_tokens"],
            "task_suffix": condition["task_suffix"],
            "verifier_pass_rate": summary.get("verifier_pass_rate"),
            "invalid_parse_rate": summary.get("invalid_parse_rate"),
            "avg_generated_tokens": summary.get("avg_generated_tokens"),
            "runtime_s": summary.get("runtime_s"),
            "peak_torch_allocated_mb": summary.get("peak_torch_allocated_mb"),
            "output_dir": str(out_dir),
            "skipped": True,
        }

    config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=inf_cfg.get("dtype", "bfloat16"),
        seed=inf_cfg.get("seed", 42),
        do_sample=inf_cfg.get("do_sample", False),
        max_new_tokens=condition["max_new_tokens"],
        use_cache=inf_cfg.get("use_cache", True),
        temperature=inf_cfg.get("temperature", 0.0),
        top_p=inf_cfg.get("top_p", 1.0),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        adapter_path=str(adapter_path),
    )

    print(f"\n=== Ablation condition: {condition['name']} ===")
    summary = run_multimodal_evaluation(
        eval_examples,
        config,
        MultimodalEvalConfig(
            system_prompt=system_prompt,
            model_name=model_cfg["name"],
            variant=f"multimodal_sft_{condition['name']}",
            output_dir=out_dir,
        ),
    )
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return {
        "condition": condition["name"],
        "max_new_tokens": condition["max_new_tokens"],
        "task_suffix": condition["task_suffix"],
        "verifier_pass_rate": summary.verifier_pass_rate,
        "invalid_parse_rate": summary.invalid_parse_rate,
        "avg_generated_tokens": summary.avg_generated_tokens,
        "runtime_s": summary.runtime_s,
        "peak_torch_allocated_mb": summary.peak_torch_allocated_mb,
        "output_dir": str(out_dir),
    }


def build_ablation_summary(results: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    from neuro_agent.config import load_yaml as _load

    per_condition: dict[str, Any] = {}
    for result in results:
        cond = result["condition"]
        per_task_path = Path(result["output_dir"]) / "per_task_metrics.json"
        per_verifier_path = Path(result["output_dir"]) / "verifier_summary.json"
        per_task = json.loads(per_task_path.read_text()) if per_task_path.exists() else {}
        per_verifier = json.loads(per_verifier_path.read_text()) if per_verifier_path.exists() else {}
        per_condition[cond] = {
            **result,
            "per_task": per_task,
            "per_verifier": per_verifier,
        }

    current = per_condition.get("current", {})
    answer_only = per_condition.get("answer_only", {})
    low_tokens = per_condition.get("low_tokens", {})

    def _rate(metrics: dict[str, Any], key: str) -> float | None:
        entry = metrics.get(key, {})
        return entry.get("verifier_pass_rate")

    waveform_tasks = ["waveform_highest_rms", "waveform_max_rms_numeric", "waveform_rms_order"]
    waveform_recovery = {}
    for task in waveform_tasks:
        waveform_recovery[task] = {
            "current": _rate(current.get("per_task", {}), task),
            "low_tokens": _rate(low_tokens.get("per_task", {}), task),
            "answer_only": _rate(answer_only.get("per_task", {}), task),
        }

    cat_current = current.get("per_verifier", {}).get("categorical", {}).get("verifier_pass_rate")
    cat_answer = answer_only.get("per_verifier", {}).get("categorical", {}).get("verifier_pass_rate")
    cat_low = low_tokens.get("per_verifier", {}).get("categorical", {}).get("verifier_pass_rate")

    formatting_primary = False
    if cat_answer is not None and cat_current is not None:
        if cat_answer >= 0.08 and cat_answer > cat_current * 2:
            formatting_primary = True
    wf_current_rates = [waveform_recovery[t]["current"] for t in waveform_tasks if waveform_recovery[t]["current"] is not None]
    wf_answer_rates = [waveform_recovery[t]["answer_only"] for t in waveform_tasks if waveform_recovery[t]["answer_only"] is not None]
    if wf_current_rates and wf_answer_rates:
        if sum(wf_answer_rates) / len(wf_answer_rates) >= 0.4:
            formatting_primary = True

    return {
        "conditions": per_condition,
        "waveform_recovery": waveform_recovery,
        "categorical_by_condition": {
            "current": cat_current,
            "low_tokens": cat_low,
            "answer_only": cat_answer,
        },
        "formatting_primary_issue": formatting_primary,
        "retrain_recommended": not formatting_primary,
        "recommendation": (
            "Formatting/decoding is the primary issue; do not retrain."
            if formatting_primary
            else "Corrective retraining with answer-only targets is justified."
        ),
    }


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})
    dataset_path = PROJECT_ROOT / eval_cfg.get(
        "dataset", "data/processed/vision/multimodal_eval_heldout.jsonl"
    )
    held_out = set(eval_cfg.get("held_out_subjects", ["S026", "S027", "S028", "S029", "S030"]))
    forbidden = set(eval_cfg.get("train_subjects", [])) | set(eval_cfg.get("validation_subjects", []))

    examples = load_eval_examples(dataset_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    integrity = verify_heldout_integrity(examples, held_out, forbidden)
    print(f"Held-out integrity: {integrity['example_count']} examples")

    conditions = ABLATION_CONDITIONS
    if args.conditions:
        names = set(args.conditions)
        conditions = [c for c in ABLATION_CONDITIONS if c["name"] in names]

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for condition in conditions:
        results.append(
            run_condition(
                condition=condition,
                examples=examples,
                cfg=cfg,
                output_root=args.output_root,
            )
        )

    summary = build_ablation_summary(results, args.output_root)
    with (args.output_root / "ablation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nAblation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
