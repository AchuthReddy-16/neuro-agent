#!/usr/bin/env python3
"""Run base VLM held-out multimodal evaluation (Stage F)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.evaluation.llm_eval import load_eval_examples, verify_heldout_integrity
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.multimodal.eval import MultimodalEvalConfig, run_multimodal_evaluation
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run base multimodal VLM evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "multimodal_eval.yaml",
        help="Path to multimodal eval YAML config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke testing",
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
    output_dir = PROJECT_ROOT / output_cfg["dir"]

    held_out = set(eval_cfg["held_out_subjects"])
    forbidden = set(eval_cfg.get("train_subjects", [])) | set(
        eval_cfg.get("validation_subjects", [])
    )

    examples = load_eval_examples(dataset_path)
    if args.limit is not None:
        examples = examples[: args.limit]

    integrity = verify_heldout_integrity(examples, held_out, forbidden)
    print(
        f"Held-out integrity passed: {integrity['example_count']} examples, "
        f"subjects={integrity['confirmed_subjects']}"
    )

    config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=model_cfg["dtype"],
        seed=inf_cfg["seed"],
        do_sample=inf_cfg["do_sample"],
        max_new_tokens=inf_cfg["max_new_tokens"],
        use_cache=inf_cfg["use_cache"],
        temperature=inf_cfg.get("temperature", 0.0),
        top_p=inf_cfg.get("top_p", 1.0),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    print(f"Evaluating {model_cfg['name']} ({model_cfg.get('variant', 'base')}) on {len(examples)} examples...")
    summary = run_multimodal_evaluation(
        examples,
        config,
        MultimodalEvalConfig(
            system_prompt=prompt_cfg["system"],
            model_name=model_cfg["name"],
            variant=model_cfg.get("variant", "base_BF16"),
            output_dir=output_dir,
        ),
    )

    print("\nEvaluation complete.")
    print(
        json.dumps(
            {
                "model": summary.model_name,
                "variant": summary.variant,
                "total_examples": summary.total_examples,
                "verifier_pass_rate": summary.verifier_pass_rate,
                "invalid_parse_rate": summary.invalid_parse_rate,
                "empty_refusal_rate": summary.empty_refusal_rate,
                "runtime_s": summary.runtime_s,
                "peak_torch_allocated_mb": summary.peak_torch_allocated_mb,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
