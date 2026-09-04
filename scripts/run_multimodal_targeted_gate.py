#!/usr/bin/env python3
"""Stage F.1 Step 6: Targeted gate eval before full 440-example run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.evaluation.llm_eval import load_eval_examples
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.multimodal.dataset import normalize_eval_example
from neuro_agent.multimodal.eval import MultimodalEvalConfig, run_multimodal_evaluation
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs

GATE_TASKS = {
    "waveform": [
        "waveform_highest_rms",
        "waveform_max_rms_numeric",
        "waveform_rms_order",
    ],
    "categorical": [
        "spectrogram_strongest_vs_weakest",
        "spectrogram_dominant_band",
        "topomap_strongest_alpha_mu",
        "band_power_weakest_alpha_mu",
        "psd_dominant_band",
    ],
    "ranking": [
        "psd_band_order",
        "band_power_beta_top3",
        "topomap_beta_top3",
    ],
    "numeric": [
        "psd_peak_frequency",
        "spectrogram_peak_frequency",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal targeted gate evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "multimodal_sft_corrective.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override adapter checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/multimodal_corrected_targeted_gate",
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

    adapter_path = args.checkpoint or (
        PROJECT_ROOT / cfg["output"]["checkpoint_dir"] / "final"
    )

    all_examples = [normalize_eval_example(ex) for ex in load_eval_examples(
        PROJECT_ROOT / eval_cfg["dataset"]
    )]
    gate_families = {f for families in GATE_TASKS.values() for f in families}
    examples = [ex for ex in all_examples if ex["category"] in gate_families]

    config = InferenceConfig(
        model_name=model_cfg["name"],
        dtype=inf_cfg.get("dtype", "bfloat16"),
        seed=inf_cfg["seed"],
        do_sample=inf_cfg["do_sample"],
        max_new_tokens=inf_cfg["max_new_tokens"],
        use_cache=inf_cfg.get("use_cache", True),
        temperature=inf_cfg.get("temperature", 0.0),
        top_p=inf_cfg.get("top_p", 1.0),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        adapter_path=str(adapter_path),
    )

    print(f"Gate eval: {len(examples)} examples from {len(gate_families)} families")
    summary = run_multimodal_evaluation(
        examples,
        config,
        MultimodalEvalConfig(
            system_prompt=prompt_cfg["system"],
            model_name=model_cfg["name"],
            variant="multimodal_sft_corrected_gate",
            output_dir=args.output_dir,
        ),
    )

    per_task = json.loads((args.output_dir / "per_task_metrics.json").read_text())
    per_verifier = json.loads((args.output_dir / "verifier_summary.json").read_text())

    group_rates: dict[str, float] = {}
    for group, families in GATE_TASKS.items():
        rates = []
        for fam in families:
            entry = per_task.get(fam, {})
            if entry.get("count", 0) > 0:
                rates.append(entry.get("verifier_pass_rate", 0.0))
        group_rates[group] = sum(rates) / len(rates) if rates else 0.0

    waveform_ok = group_rates.get("waveform", 0.0) >= 0.35
    categorical_ok = group_rates.get("categorical", 0.0) >= 0.06
    ranking_ok = group_rates.get("ranking", 0.0) >= 0.40
    numeric_ok = group_rates.get("numeric", 0.0) >= 0.15
    gate_pass = waveform_ok and categorical_ok and ranking_ok and numeric_ok

    gate_result = {
        "gate_pass": gate_pass,
        "overall_pass_rate": summary.verifier_pass_rate,
        "group_pass_rates": group_rates,
        "per_verifier": per_verifier,
        "checks": {
            "waveform_recovery": waveform_ok,
            "categorical_recovery": categorical_ok,
            "ranking_preserved": ranking_ok,
            "numeric_preserved": numeric_ok,
        },
        "example_count": len(examples),
        "checkpoint": str(adapter_path),
    }

    with (args.output_dir / "gate_result.json").open("w") as handle:
        json.dump(gate_result, handle, indent=2)

    print(json.dumps(gate_result, indent=2))
    if not gate_pass:
        print("GATE FAILED — stopping before full 440 eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
