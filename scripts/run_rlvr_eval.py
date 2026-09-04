#!/usr/bin/env python3
"""Run RLVR model evaluation with optional targeted gate check."""

from __future__ import annotations

import argparse
import json
import sys
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


def _filter_categories(
    examples: list[dict],
    categories: list[str] | None,
) -> list[dict]:
    if not categories:
        return examples
    allowed = set(categories)
    return [ex for ex in examples if ex.get("category") in allowed]


def _evaluate_gate(per_task: dict[str, dict], gate_cfg: dict) -> dict:
    checks: dict[str, dict] = {}
    passed = True
    reference = gate_cfg.get("reference", {})
    for key, minimum in gate_cfg.items():
        if key == "reference":
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
    return {"passed": passed, "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RLVR model evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "eval_rlvr.yaml",
    )
    parser.add_argument(
        "--targeted-only",
        action="store_true",
        help="Evaluate only targeted gate categories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging",
    )
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
    examples = _filter_categories(examples, categories)
    if args.limit is not None:
        examples = examples[: args.limit]

    integrity = verify_heldout_integrity(examples, held_out, forbidden)
    print(
        f"Held-out integrity passed: {integrity['example_count']} examples, "
        f"subjects={integrity['confirmed_subjects']}"
    )
    if categories:
        print(f"Targeted categories: {categories}")

    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        adapter_path = str(PROJECT_ROOT / adapter_path)

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
    )

    print(
        f"Evaluating {model_cfg['name']} ({model_cfg.get('variant', 'rlvr')}) "
        f"with adapter={adapter_path} on {len(examples)} examples..."
    )
    summary = run_llm_evaluation(
        examples,
        config,
        system_prompt=prompt_cfg["system"],
        model_name=model_cfg["name"],
        variant=model_cfg.get("variant", "rlvr_QLoRA_BF16"),
        output_dir=output_dir,
    )

    per_task_path = output_dir / "per_task_metrics.json"
    with per_task_path.open() as handle:
        per_task = json.load(handle)

    gate_result = None
    if args.targeted_only and "gate" in cfg:
        gate_result = _evaluate_gate(per_task, cfg["gate"])
        gate_path = output_dir / "gate_result.json"
        with gate_path.open("w") as handle:
            json.dump(gate_result, handle, indent=2)

    print("\nEvaluation complete.")
    payload = {
        "model": summary.model_name,
        "variant": summary.variant,
        "total_examples": summary.total_examples,
        "verifier_pass_rate": summary.verifier_pass_rate,
        "invalid_parse_rate": summary.invalid_parse_rate,
        "empty_refusal_rate": summary.empty_refusal_rate,
        "runtime_s": summary.runtime_s,
        "peak_torch_allocated_mb": summary.peak_torch_allocated_mb,
        "output_dir": str(output_dir),
        "per_task": {
            task: per_task[task]["verifier_pass_rate"]
            for task in sorted(per_task)
        },
    }
    if gate_result is not None:
        payload["gate"] = gate_result
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
