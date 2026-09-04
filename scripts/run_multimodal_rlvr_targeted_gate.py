#!/usr/bin/env python3
"""Stage F.2 targeted gate: RLVR vs corrected SFT baselines."""

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
    "set_membership": [
        "band_power_high_beta_set",
        "topomap_high_delta_set",
    ],
    "waveform_numeric": [
        "waveform_max_rms_numeric",
    ],
    "spectrogram_peak_frequency": [
        "spectrogram_peak_frequency",
    ],
}


def _group_rate(per_task: dict, families: list[str]) -> float | None:
    rates = []
    for fam in families:
        entry = per_task.get(fam, {})
        if entry.get("count", 0) > 0:
            rates.append(entry.get("verifier_pass_rate", 0.0))
    if not rates:
        return None
    return sum(rates) / len(rates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal RLVR targeted gate evaluation")
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "multimodal_rlvr.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    gate_cfg = cfg.get("gate", {})

    adapter_path = args.checkpoint or (PROJECT_ROOT / cfg["output"]["checkpoint_dir"] / "final")
    output_dir = args.output_dir or PROJECT_ROOT / gate_cfg.get(
        "output_dir", "results/multimodal_rlvr_targeted_gate"
    )
    baseline_dir = PROJECT_ROOT / gate_cfg.get("baseline_dir", "results/multimodal_corrected_eval")
    baseline_rates = gate_cfg.get("baseline_rates", {})
    cat_floor = float(gate_cfg.get("categorical_floor_pp", 5.0)) / 100.0
    rank_floor = float(gate_cfg.get("ranking_floor_pp", 5.0)) / 100.0
    num_floor = float(gate_cfg.get("numeric_floor_pp", 5.0)) / 100.0

    all_examples = [
        normalize_eval_example(ex)
        for ex in load_eval_examples(PROJECT_ROOT / eval_cfg["dataset"])
    ]
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
            variant="multimodal_rlvr_gate",
            output_dir=output_dir,
        ),
    )

    per_task = json.loads((output_dir / "per_task_metrics.json").read_text())
    per_verifier = json.loads((output_dir / "verifier_summary.json").read_text())

    baseline_per_task = {}
    baseline_verifier = {}
    if (baseline_dir / "per_task_metrics.json").exists():
        baseline_per_task = json.loads((baseline_dir / "per_task_metrics.json").read_text())
    if (baseline_dir / "verifier_summary.json").exists():
        baseline_verifier = json.loads((baseline_dir / "verifier_summary.json").read_text())

    group_rates: dict[str, float | None] = {}
    for group, families in GATE_TASKS.items():
        group_rates[group] = _group_rate(per_task, families)

    rlvr_categorical = per_verifier.get("categorical", {}).get("verifier_pass_rate", 0.0)
    rlvr_ranking = per_verifier.get("ranking", {}).get("verifier_pass_rate", 0.0)
    rlvr_numeric = per_verifier.get("numeric", {}).get("verifier_pass_rate", 0.0)
    rlvr_set = per_verifier.get("set", {}).get("verifier_pass_rate", 0.0)

    base_categorical = baseline_rates.get(
        "categorical", baseline_verifier.get("categorical", {}).get("verifier_pass_rate", 0.0)
    )
    base_ranking = baseline_rates.get(
        "ranking", baseline_verifier.get("ranking", {}).get("verifier_pass_rate", 0.0)
    )
    base_numeric = baseline_rates.get(
        "numeric", baseline_verifier.get("numeric", {}).get("verifier_pass_rate", 0.0)
    )
    base_set = baseline_rates.get("set", baseline_verifier.get("set", {}).get("verifier_pass_rate", 0.0))
    base_waveform = baseline_rates.get(
        "waveform_max_rms_numeric",
        baseline_per_task.get("waveform_max_rms_numeric", {}).get("verifier_pass_rate", 0.0),
    )
    base_spec_peak = baseline_rates.get(
        "spectrogram_peak_frequency",
        baseline_per_task.get("spectrogram_peak_frequency", {}).get("verifier_pass_rate", 0.0),
    )

    rlvr_waveform = per_task.get("waveform_max_rms_numeric", {}).get("verifier_pass_rate", 0.0)
    rlvr_spec_peak = per_task.get("spectrogram_peak_frequency", {}).get("verifier_pass_rate", 0.0)

    categorical_ok = rlvr_categorical >= (base_categorical - cat_floor)
    ranking_ok = rlvr_ranking >= (base_ranking - rank_floor)
    numeric_ok = rlvr_numeric >= (base_numeric - num_floor)
    set_improved = rlvr_set > base_set
    waveform_improved = rlvr_waveform > base_waveform
    spec_peak_improved = rlvr_spec_peak > base_spec_peak

    gate_pass = (
        categorical_ok
        and ranking_ok
        and numeric_ok
        and (set_improved or waveform_improved or spec_peak_improved)
    )

    gate_result = {
        "gate_pass": gate_pass,
        "overall_pass_rate": summary.verifier_pass_rate,
        "group_pass_rates": group_rates,
        "per_verifier": per_verifier,
        "baseline_verifier": baseline_verifier,
        "deltas_vs_corrected": {
            "categorical": rlvr_categorical - base_categorical,
            "ranking": rlvr_ranking - base_ranking,
            "numeric": rlvr_numeric - base_numeric,
            "set": rlvr_set - base_set,
            "waveform_max_rms_numeric": rlvr_waveform - base_waveform,
            "spectrogram_peak_frequency": rlvr_spec_peak - base_spec_peak,
        },
        "checks": {
            "categorical_preserved": categorical_ok,
            "ranking_preserved": ranking_ok,
            "numeric_not_regressed": numeric_ok,
            "set_membership_improved": set_improved,
            "waveform_numeric_improved": waveform_improved,
            "spectrogram_peak_improved": spec_peak_improved,
        },
        "thresholds": {
            "categorical_min": base_categorical - cat_floor,
            "ranking_min": base_ranking - rank_floor,
            "numeric_min": base_numeric - num_floor,
        },
        "example_count": len(examples),
        "checkpoint": str(adapter_path),
    }

    with (output_dir / "gate_result.json").open("w") as handle:
        json.dump(gate_result, handle, indent=2)

    print(json.dumps(gate_result, indent=2))
    if not gate_pass:
        print("GATE FAILED — stopping before full 440 eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
