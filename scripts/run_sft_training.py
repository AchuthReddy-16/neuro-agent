#!/usr/bin/env python3
"""Run QLoRA SFT training for Qwen3-4B (Stage D)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.config import load_yaml
from neuro_agent.paths import CONFIGS_DIR, PROJECT_ROOT, configure_hf_cache, ensure_dirs
from neuro_agent.training.trainer import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QLoRA SFT training")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_DIR / "sft.yaml",
        help="Path to SFT YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_hf_cache()
    ensure_dirs()
    results_dir = PROJECT_ROOT / load_yaml(args.config)["output"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    trainer = SFTTrainer(config, PROJECT_ROOT)
    summary = trainer.train()

    print("\nTraining complete.")
    print(
        json.dumps(
            {
                "model": summary.model_name,
                "train_examples": summary.train_examples,
                "validation_examples": summary.validation_examples,
                "total_steps": summary.total_steps,
                "final_train_loss": summary.final_train_loss,
                "final_eval_loss": summary.final_eval_loss,
                "runtime_s": summary.runtime_s,
                "peak_vram_mb": summary.peak_vram_mb,
                "adapter_size_mb": summary.adapter_size_mb,
                "checkpoint_path": summary.checkpoint_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
