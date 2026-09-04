#!/usr/bin/env python3
"""Stage H.1 quality eval for text model under BF16 / INT8 / INT4."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.evaluation.llm_eval import (
    load_eval_examples,
    run_llm_evaluation,
    verify_heldout_integrity,
)
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs
from neuro_agent.quantization import normalize_quantization


def _filter_categories(examples: list[dict], categories: list[str] | None) -> list[dict]:
    if not categories:
        return examples
    allowed = set(categories)
    return [ex for ex in examples if ex.get("category") in allowed]


def _balanced_subset(examples: list[dict], categories: list[str], per_category: int) -> list[dict]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        cat = ex.get("category")
        if cat in categories and len(buckets[cat]) < per_category:
            buckets[cat].append(ex)
    ordered: list[dict] = []
    for cat in categories:
        ordered.extend(buckets.get(cat, []))
    return ordered


def _evaluate_gate(per_task: dict[str, dict], gate_cfg: dict, summary_pass: float, invalid_parse: float) -> dict:
    checks: dict[str, dict] = {}
    passed = True
    reference = gate_cfg.get("reference", {})

    for key, minimum in gate_cfg.items():
        if key in {
            "reference",
            "catastrophic_overall_drop",
            "catastrophic_invalid_parse_rate",
        }:
            continue
        task = key.removesuffix("_min") if key.endswith("_min") else key
        rate = per_task.get(task, {}).get("verifier_pass_rate")
        ref = reference.get(task)
        ok = rate is not None and rate >= float(minimum)
        checks[task] = {
            "rate": rate,
            "minimum": float(minimum),
            "reference": ref,
            "passed": ok,
        }
        if not ok:
            passed = False

    cat_drop = float(gate_cfg.get("catastrophic_overall_drop", 0.15))
    ref_overall = sum(float(v) for v in reference.values()) / max(len(reference), 1)
    catastrophic = False
    reasons: list[str] = []
    if summary_pass < (ref_overall - cat_drop):
        catastrophic = True
        reasons.append(
            f"overall_pass {summary_pass:.3f} < reference_mean {ref_overall:.3f} - {cat_drop}"
        )
    max_invalid = float(gate_cfg.get("catastrophic_invalid_parse_rate", 0.05))
    if invalid_parse > max_invalid:
        catastrophic = True
        reasons.append(f"invalid_parse_rate {invalid_parse:.3f} > {max_invalid}")
    if not passed:
        failed = [k for k, v in checks.items() if not v["passed"]]
        reasons.append(f"task_floors_failed={failed}")

    return {
        "passed": passed and not catastrophic,
        "catastrophic": catastrophic,
        "checks": checks,
        "reasons": reasons,
        "overall_pass_rate": summary_pass,
        "invalid_parse_rate": invalid_parse,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text quantization quality evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to eval_quant_text_{bf16,int8,int4}.yaml",
    )
    parser.add_argument(
        "--targeted-only",
        action="store_true",
        help="Evaluate balanced subset across all targeted categories",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    inf_cfg = cfg["inference"]
    eval_cfg = cfg["evaluation"]
    prompt_cfg = cfg["prompt"]
    output_cfg = cfg["output"]

    dataset_path = PROJECT_ROOT / eval_cfg["dataset"]
    output_dir = PROJECT_ROOT / (
        output_cfg["targeted_dir"] if args.targeted_only else output_cfg["dir"]
    )

    held_out = set(eval_cfg["held_out_subjects"])
    forbidden = set(eval_cfg.get("train_subjects", [])) | set(
        eval_cfg.get("validation_subjects", [])
    )

    examples = load_eval_examples(dataset_path)
    categories = eval_cfg.get("targeted_categories") if args.targeted_only else None
    if args.targeted_only and categories:
        per_cat = int(eval_cfg.get("targeted_per_category", 10))
        examples = _balanced_subset(examples, categories, per_cat)
    elif categories:
        examples = _filter_categories(examples, categories)
    if args.limit is not None:
        examples = examples[: args.limit]

    integrity = verify_heldout_integrity(examples, held_out, forbidden)
    print(
        f"Held-out integrity passed: {integrity['example_count']} examples, "
        f"subjects={integrity['confirmed_subjects']}"
    )
    if args.targeted_only and categories:
        print(f"Targeted categories ({len(categories)}): {categories}")

    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        adapter_path = str(PROJECT_ROOT / adapter_path)

    quant = normalize_quantization(model_cfg.get("quantization", "none")).value
    config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        seed=inf_cfg["seed"],
        do_sample=inf_cfg["do_sample"],
        max_new_tokens=inf_cfg["max_new_tokens"],
        use_cache=inf_cfg["use_cache"],
        temperature=inf_cfg.get("temperature", 0.0),
        top_p=inf_cfg.get("top_p", 1.0),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        adapter_path=adapter_path,
        quantization=quant,
    )

    print(
        f"Evaluating {model_cfg['name']} ({model_cfg.get('variant')}) "
        f"quant={quant} adapter={adapter_path} on {len(examples)} examples..."
    )
    summary = run_llm_evaluation(
        examples,
        config,
        system_prompt=prompt_cfg["system"],
        model_name=model_cfg["name"],
        variant=model_cfg.get("variant", f"sft_corrected_v2_{quant.upper()}"),
        output_dir=output_dir,
    )

    per_task_path = output_dir / "per_task_metrics.json"
    with per_task_path.open() as handle:
        per_task = json.load(handle)

    gate_result = None
    if args.targeted_only and "gate" in cfg:
        gate_result = _evaluate_gate(
            per_task,
            cfg["gate"],
            summary.verifier_pass_rate,
            summary.invalid_parse_rate,
        )
        with (output_dir / "gate_result.json").open("w") as handle:
            json.dump(gate_result, handle, indent=2)

    payload = {
        "model": summary.model_name,
        "variant": summary.variant,
        "quantization": quant,
        "total_examples": summary.total_examples,
        "verifier_pass_rate": summary.verifier_pass_rate,
        "invalid_parse_rate": summary.invalid_parse_rate,
        "empty_refusal_rate": summary.empty_refusal_rate,
        "avg_generated_tokens": summary.avg_generated_tokens,
        "runtime_s": summary.runtime_s,
        "peak_torch_allocated_mb": summary.peak_torch_allocated_mb,
        "peak_torch_reserved_mb": summary.peak_torch_reserved_mb,
        "nvidia_smi_peak_mb": summary.nvidia_smi_peak_mb,
        "output_dir": str(output_dir),
        "per_task": {
            task: per_task[task]["verifier_pass_rate"] for task in sorted(per_task)
        },
    }
    if gate_result is not None:
        payload["gate"] = gate_result
    print(json.dumps(payload, indent=2))

    if gate_result is not None and gate_result.get("catastrophic"):
        print("CATASTROPHIC regression detected; skip full eval for this variant.")
        sys.exit(2)


if __name__ == "__main__":
    main()
