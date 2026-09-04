#!/usr/bin/env python3
"""Run multimodal corrected SFT checkpoint evaluation (Stage F.1)."""

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
    parser = argparse.ArgumentParser(description="Run multimodal corrected SFT evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "multimodal_sft_corrective.yaml",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    eval_cfg = cfg.get("evaluation", {})
    prompt_cfg = cfg["prompt"]
    output_cfg = cfg.get("eval_output", {"dir": "results/multimodal_corrected_eval"})
    inf_cfg = cfg.get("inference", {})

    dataset_path = PROJECT_ROOT / eval_cfg.get(
        "dataset", "data/processed/vision/multimodal_eval_heldout.jsonl"
    )
    output_dir = PROJECT_ROOT / output_cfg["dir"]
    adapter_path = PROJECT_ROOT / cfg["output"]["checkpoint_dir"] / "final"

    held_out = set(eval_cfg.get("held_out_subjects", ["S026", "S027", "S028", "S029", "S030"]))
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
        dtype=inf_cfg.get("dtype", "bfloat16"),
        seed=inf_cfg.get("seed", 42),
        do_sample=inf_cfg.get("do_sample", False),
        max_new_tokens=inf_cfg.get("max_new_tokens", 32),
        use_cache=inf_cfg.get("use_cache", True),
        temperature=inf_cfg.get("temperature", 0.0),
        top_p=inf_cfg.get("top_p", 1.0),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        adapter_path=str(adapter_path),
    )

    print(f"Evaluating corrected checkpoint on {len(examples)} examples...")
    summary = run_multimodal_evaluation(
        examples,
        config,
        MultimodalEvalConfig(
            system_prompt=prompt_cfg["system"],
            model_name=model_cfg["name"],
            variant="multimodal_sft_corrected",
            output_dir=output_dir,
        ),
    )

    print("\nEvaluation complete.")
    print(json.dumps({"verifier_pass_rate": summary.verifier_pass_rate, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
